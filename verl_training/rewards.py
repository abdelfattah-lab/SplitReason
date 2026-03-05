"""
Reward functions for veRL GRPO training of SplitReason.

Phase 1: accuracy-only reward via math_verify.
Phase 3: full weighted reward (accuracy + tag_count + format).

veRL calls compute_score(data_source, solution_str, ground_truth, extra_info)
and expects a float return (not None). We return 0.0 for unparseable gold.

Ported from: training/open-r1/src/open_r1/rewards.py
"""

import logging
import os
import re
from typing import Optional

# Suppress noisy math_verify error logs (signal.SIGALRM failures in Ray workers)
logging.getLogger("math_verify").setLevel(logging.CRITICAL)

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


# ============================================================================
# Helpers (shared across phases)
# ============================================================================

_ANSWER_BLOCK_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.DOTALL)


def _extract_answer_block(text: str) -> str:
    """Return content inside <answer>...</answer> if present, else full text."""
    m = _ANSWER_BLOCK_RE.search(text)
    return m.group(1) if m else text


def _extract_boxed_content(text: str) -> Optional[str]:
    """Return the LAST balanced \\boxed{...} content."""
    key = r"\boxed{"
    start = text.rfind(key)
    while start != -1:
        i = start + len(key)
        depth = 1
        out = []
        while i < len(text):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return "".join(out)
            out.append(ch)
            i += 1
        start = text.rfind(key, 0, start)
    return None


# ============================================================================
# Accuracy reward (Phase 1 core, also used in Phase 3 weighted sum)
# ============================================================================

def _accuracy_reward(solution_str: str, ground_truth: str) -> float:
    """
    Binary accuracy reward using math_verify symbolic verification.
    Returns 1.0 (correct), 0.0 (wrong or unparseable).

    Ported from rewards.py lines 232-276 (accuracy_reward).
    """
    # Strip "Put your final answer within \boxed{}" if model echoed it
    content = solution_str.replace("Put your final answer within \\boxed{}", "")

    # Parse gold solution
    # math_verify uses signal.SIGALRM for timeouts, which fails in Ray worker
    # threads. Catch ValueError and fall back to 0.0.
    try:
        gold_parsed = parse(ground_truth, extraction_mode="first_match",
                            extraction_config=[LatexExtractionConfig()])
    except (ValueError, Exception):
        return 0.0
    if len(gold_parsed) == 0:
        # Unparseable gold — in TRL we return None to skip; in veRL return 0.0
        return 0.0

    # Parse model answer with strict settings (no malformed operators)
    try:
        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
    except (ValueError, Exception):
        return 0.0

    # Verify symbolic equivalence
    try:
        return float(verify(gold_parsed, answer_parsed))
    except Exception:
        return 0.0


# ============================================================================
# Tag / coverage / format helpers (Phase 3)
# ============================================================================

def _has_proper_bigmodel_nesting(text: str) -> bool:
    """
    Returns True iff each <bigmodel> ... </bigmodel> pair is balanced and
    no new <bigmodel> starts before the previous one is closed.

    Ported from rewards.py lines 108-128.
    """
    pos = 0
    open_tag = "<bigmodel>"
    close_tag = "</bigmodel>"

    while True:
        open_pos = text.find(open_tag, pos)
        if open_pos == -1:
            return True
        close_pos = text.find(close_tag, open_pos + len(open_tag))
        if close_pos == -1:
            return False
        next_open = text.find(open_tag, open_pos + len(open_tag))
        if next_open != -1 and next_open < close_pos:
            return False
        pos = close_pos + len(close_tag)


def _coverage_ratio(text: str) -> float:
    """Fraction of text chars inside <bigmodel>...</bigmodel> blocks."""
    total = len(text)
    if total == 0:
        return 0.0
    segments = re.findall(r"<bigmodel>(.*?)</bigmodel>", text, re.DOTALL)
    bigmodel_chars = sum(len(seg) for seg in segments)
    return bigmodel_chars / total


def _coverage_reward(text: str) -> float:
    """
    Piecewise linear reward on coverage ratio:
    - 0 <= ratio < 0.15: linearly 0 to +1
    - 0.15 <= ratio <= 1.0: linearly +1 down to -1

    Ported from rewards.py lines 371-397.
    """
    increase_till = 0.15
    try:
        total_chars = len(text)
        if total_chars == 0:
            return 0.0
        segments = re.findall(r"<bigmodel>(.*?)</bigmodel>", text, re.DOTALL)
        bigmodel_chars = sum(len(seg) for seg in segments)
        r = bigmodel_chars / total_chars
        if r <= 0:
            return 0.0
        if r < increase_till:
            return r / increase_till
        slope = -2.0 / (1.0 - increase_till)
        intercept = 1.0 - (slope * increase_till)
        val = slope * r + intercept
        return max(-1.0, min(val, 1.0))
    except Exception:
        return 0.0


def _bigmodel_count_reward(n: int) -> float:
    """Peaks at 6 occurrences, slope 0.5. Ported from rewards.py line 416-427."""
    peak = 6
    slope = 0.5
    if n <= peak:
        return slope * n
    else:
        return slope * peak - slope * (n - peak)


def _tag_count_reward(text: str) -> float:
    """
    Scaffold tags + bigmodel usage + coverage.
    Max think-answer-box tag reward ~2, max bigmodel count ~1.5, max coverage ~2.

    Ported from rewards.py lines 400-460.
    """
    r = 0.0

    # Scaffold tags (max +2.0)
    if text.count("<think>\n") == 1:
        r += 0.5
    if text.count("\n</think>\n") == 1:
        r += 0.5
    if text.count("\n<answer>\n") == 1:
        r += 0.25
    if text.count("\n</answer>") == 1:
        r += 0.25
    if text.count(r"\boxed{") > 0:
        r += 0.5

    open_count = text.count("<bigmodel>")
    close_count = text.count("</bigmodel>")

    # Bigmodel count reward (max ~1.5 at peak=6)
    r += _bigmodel_count_reward(open_count) / 2.0

    # Coverage reward (max +2, min -2)
    r += 2 * _coverage_reward(text)

    # Penalty for unbalanced tags
    if open_count != close_count:
        r -= 2

    return r


def _format_reward(text: str) -> float:
    """
    +1 if <think>/<answer> scaffold present.
    +1 if all <bigmodel> tags properly nested.

    Ported from rewards.py lines 344-369.
    """
    scaffold_pat = re.compile(
        r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$",
        flags=re.DOTALL | re.MULTILINE,
    )

    score = 0.0
    if scaffold_pat.match(text):
        score += 1.0
    if _has_proper_bigmodel_nesting(text):
        score += 1.0
    return score


# ============================================================================
# Main entry point for veRL
# ============================================================================

# Reward weights (from config_demo.yaml lines 54-57)
ACCURACY_WEIGHT = 3.0
TAG_COUNT_WEIGHT = 0.5
FORMAT_WEIGHT = 0.25

# Phase flag: set SPLITREASON_FULL_REWARDS=1 env var to enable tag_count + format
ENABLE_FULL_REWARDS = os.environ.get("SPLITREASON_FULL_REWARDS", "0") == "1"


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict = None, **kwargs):
    """
    veRL reward function entry point.

    Phase 1 (SPLITREASON_FULL_REWARDS unset): accuracy-only, returns float.
    Phase 3 (SPLITREASON_FULL_REWARDS=1): weighted sum of accuracy + tag_count
    + format, returns dict with component breakdown for logging.

    Args:
        data_source: Dataset identifier (e.g., "open-r1/OpenR1-Math-220k")
        solution_str: Full model completion string
        ground_truth: Gold solution string
        extra_info: Optional metadata dict

    Returns:
        float (Phase 1) or dict with "score" key (Phase 3)
    """
    acc = _accuracy_reward(solution_str, ground_truth)

    if not ENABLE_FULL_REWARDS:
        return acc

    # Phase 3: weighted combination with component breakdown
    tag = _tag_count_reward(solution_str)
    fmt = _format_reward(solution_str)

    total = (ACCURACY_WEIGHT * acc) + (TAG_COUNT_WEIGHT * tag) + (FORMAT_WEIGHT * fmt)

    # Return dict so veRL logs components as reward_extra_info
    return {
        "score": total,
        "acc": acc,
        "tag_count": tag,
        "format": fmt,
        "bigmodel_count": solution_str.count("<bigmodel>"),
        "coverage_ratio": _coverage_ratio(solution_str),
        "balanced_tags": int(
            solution_str.count("<bigmodel>") == solution_str.count("</bigmodel>")
        ),
    }
