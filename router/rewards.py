"""
Reward functions for router-based GRPO training.

Default: accuracy-only. Enable routing cost/switching penalty via env vars:
  SPLITREASON_W_ACC (default 1.0)
  SPLITREASON_W_COST (default 0.0)
  SPLITREASON_W_SWITCH (default 0.0)

veRL calls compute_score(data_source, solution_str, ground_truth, extra_info)
and expects a float return (not None). We return 0.0 for unparseable gold.
"""

import logging
import os
import re
from typing import Optional

logging.getLogger("math_verify").setLevel(logging.CRITICAL)

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

# Reward weights from env vars
ACCURACY_WEIGHT = float(os.environ.get("SPLITREASON_W_ACC", "1.0"))
ROUTING_COST_WEIGHT = float(os.environ.get("SPLITREASON_W_COST", "0.0"))
SWITCHING_PENALTY_WEIGHT = float(os.environ.get("SPLITREASON_W_SWITCH", "0.0"))

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


def _accuracy_reward(solution_str: str, ground_truth: str) -> float:
    """
    Binary accuracy reward using math_verify symbolic verification.
    Returns 1.0 (correct), 0.0 (wrong or unparseable).
    """
    content = solution_str.replace("Put your final answer within \\boxed{}", "")

    try:
        gold_parsed = parse(
            ground_truth,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
    except (ValueError, Exception):
        return 0.0
    if len(gold_parsed) == 0:
        return 0.0

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

    try:
        return float(verify(gold_parsed, answer_parsed))
    except Exception:
        return 0.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    **kwargs,
):
    """
    veRL reward function entry point.

    Accuracy-only when SPLITREASON_W_COST=0 and SPLITREASON_W_SWITCH=0.
    Otherwise returns dict with component breakdown.
    """
    acc = _accuracy_reward(solution_str, ground_truth)

    if ROUTING_COST_WEIGHT == 0 and SWITCHING_PENALTY_WEIGHT == 0:
        return acc

    num_yields = extra_info.get("num_yields", 0) if extra_info else 0
    num_switches = extra_info.get("num_switches", 0) if extra_info else 0
    max_yields = extra_info.get("max_rounds", 20) if extra_info else 20

    routing_cost = -num_yields / max(max_yields, 1)
    switching_penalty = -num_switches / max(max_yields, 1)

    total = (
        ACCURACY_WEIGHT * acc
        + ROUTING_COST_WEIGHT * routing_cost
        + SWITCHING_PENALTY_WEIGHT * switching_penalty
    )

    return {
        "score": total,
        "acc": acc,
        "routing_cost": routing_cost,
        "switching_penalty": switching_penalty,
        "num_yields": num_yields,
        "num_switches": num_switches,
    }
