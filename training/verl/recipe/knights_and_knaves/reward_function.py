import re
from typing import Dict, Optional


def _extract_answer(text: str) -> Optional[str]:
    """Return the content inside <answer> tags if present; else last non-empty line."""
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    return lines[-1]


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def compute_score(
    data_source,
    solution_str: str,
    ground_truth: Dict,
    extra_info=None,
    correct_score: float = 1.0,
    incorrect_score: float = 0.0,
    abstain_score: float = 0.0,
    **kwargs,
) -> Dict[str, float]:
    """
    Reward for Knights-and-Knaves logic puzzles.

    Expected ground_truth fields (provide at least one):
      - solution_text: canonical prose answer (e.g., "Alice is a knight, Bob is a knave.")
      - solution_text_format: numbered format version of the above
    Returns a dict with 'score' and 'score_wo_hint_penalty' for compatibility.
    """
    answer = _extract_answer(solution_str or "")
    if not answer:
        return {"score": incorrect_score, "score_wo_hint_penalty": incorrect_score}

    norm_answer = _normalize(answer)
    if "abstain" in norm_answer:
        return {"score": abstain_score, "score_wo_hint_penalty": abstain_score}

    acceptable = []
    for key in ("solution_text", "solution_text_format"):
        if key in ground_truth and isinstance(ground_truth[key], str):
            acceptable.append(_normalize(ground_truth[key]))

    is_correct = norm_answer in acceptable
    if not is_correct:
        is_correct = any(norm_answer in cand or cand in norm_answer for cand in acceptable)

    score = correct_score if is_correct else incorrect_score
    return {"score": score, "score_wo_hint_penalty": score}
