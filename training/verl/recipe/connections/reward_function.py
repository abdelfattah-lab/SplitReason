import re
import random
from typing import Dict, Optional, List


def _extract_answer(text: str) -> Optional[str]:
    """Return the content inside <answer> tags if present; else last non-empty line."""
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    # fallback: grab last non-empty line
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    return lines[-1]


def _normalize(s: str) -> str:
    """Normalize string for comparison: lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", s.strip().lower())


def _parse_groups(answer_str: str) -> Optional[List[List[str]]]:
    """
    Parse answer string like {[W1, W2], [W3, W4]} into list of word lists.
    Returns None if parsing fails.
    """
    try:
        # Remove outer braces if present
        answer_str = answer_str.strip()
        if answer_str.startswith("{") and answer_str.endswith("}"):
            answer_str = answer_str[1:-1].strip()

        # Extract all groups [word1, word2, ...]
        groups = []
        pattern = r"\[(.*?)\]"
        matches = re.findall(pattern, answer_str, re.DOTALL)

        for match in matches:
            # Split on commas and normalize
            words = [w.strip().upper() for w in match.split(",") if w.strip()]
            if words:
                groups.append(sorted(words))

        return groups if groups else None
    except Exception:
        return None


def _validate_groups(predicted_groups: List[List[str]], ground_truth_answers: List[Dict]) -> bool:
    """
    Check if predicted groups match ground truth answers.

    Args:
        predicted_groups: List of sorted word lists
        ground_truth_answers: List of dicts with 'words' key

    Returns:
        True if all groups match (order-independent)
    """
    try:
        # Convert ground truth to sorted word lists
        true_groups = []
        for ans in ground_truth_answers:
            words = ans.get("words", [])
            if isinstance(words, list):
                true_groups.append(sorted([w.upper() for w in words]))

        # Sort both lists for order-independent comparison
        pred_sorted = sorted(predicted_groups)
        true_sorted = sorted(true_groups)

        return pred_sorted == true_sorted
    except Exception:
        return False


def get_num_hints(solution_str: str, hint_pattern: str = "<request>") -> int:
    """Count the number of hint request tags in the solution."""
    return solution_str.count(hint_pattern)


def compute_score(
    data_source,
    solution_str: str,
    ground_truth: Dict,
    extra_info=None,
    method: str = 'strict',
    format_score: float = 0.1,
    correct_score: float = 1.0,
    incorrect_score: float = 0.0,
    abstain_score: float = 0.0,
    reward_abstain: bool = False,
    penalize_hint: bool = False,
    hint_penalty: float = 0.1,
    **kwargs,
) -> Dict[str, float]:
    """
    Reward for NYT Connections-style puzzles with abstention and hint support.

    Args:
        data_source: Data source identifier (unused)
        solution_str: Model's generated solution
        ground_truth: Dict with 'solution_text' and/or 'answers'
        extra_info: Extra info (unused)
        method: Scoring method (unused, for compatibility)
        format_score: Score for properly formatted but incorrect answers
        correct_score: Score for correct answers
        incorrect_score: Score for incorrect/malformed answers
        abstain_score: Score when model abstains
        reward_abstain: Whether to reward abstention
        penalize_hint: Whether to penalize hint usage
        hint_penalty: Penalty per hint (multiplicative)
        **kwargs: Additional args

    Returns:
        Dict with 'score', 'score_wo_hint_penalty', and 'num_hints'

    Expected ground_truth format:
        {
            "solution_text": "{[W1, W2, ...], [W3, W4, ...]}",  # Canonical answer
            "answers": [  # Alternative: list of answer dicts
                {"words": ["W1", "W2", ...], "answerDescription": "..."},
                ...
            ]
        }
    """
    num_hints = get_num_hints(solution_str)

    # Random sampling for debug prints (avoid spam)
    do_print = random.randint(1, 64) == 1

    # Extract answer from solution
    answer = _extract_answer(solution_str or "")

    if do_print:
        print("="*60)
        print(f"Solution excerpt: {solution_str[:200]}...")
        print(f"Extracted answer: {answer}")
        print(f"Num hints: {num_hints}")

    # No answer found
    if not answer:
        if do_print:
            print("No answer found -> incorrect_score")
        return {
            "score": incorrect_score,
            "score_wo_hint_penalty": incorrect_score,
            "num_hints": num_hints
        }

    norm_answer = _normalize(answer)

    # Check for abstention
    if "abstain" in norm_answer or norm_answer == "i abstain":
        if do_print:
            print(f"Model abstained -> abstain_score={abstain_score}")
        return {
            "score": abstain_score,
            "score_wo_hint_penalty": abstain_score,
            "num_hints": num_hints
        }

    # Get ground truth
    gt_solution_text = ground_truth.get("solution_text")
    gt_answers = ground_truth.get("answers")

    # Try strict group matching first (if we have answer dicts)
    if gt_answers and isinstance(gt_answers, list):
        predicted_groups = _parse_groups(answer)
        if predicted_groups:
            is_correct = _validate_groups(predicted_groups, gt_answers)
            if is_correct:
                if do_print:
                    print(f"Correct via group matching!")
                score_wo_penalty = correct_score
                if penalize_hint:
                    score_w_penalty = correct_score * (1 - hint_penalty * num_hints)
                else:
                    score_w_penalty = correct_score
                return {
                    "score": score_w_penalty,
                    "score_wo_hint_penalty": score_wo_penalty,
                    "num_hints": num_hints
                }

    # Fallback: string matching with normalization
    acceptable = []
    if gt_solution_text:
        acceptable.append(gt_solution_text)

    # Also try creating canonical form from answers
    if gt_answers and isinstance(gt_answers, list):
        # Build canonical answer string
        groups_str = []
        for ans in gt_answers:
            words = ans.get("words", [])
            if words:
                words_upper = [w.upper() for w in words]
                groups_str.append("[" + ", ".join(words_upper) + "]")
        if groups_str:
            canonical = "{" + ", ".join(groups_str) + "}"
            acceptable.append(canonical)

    acceptable_norm = [_normalize(a) for a in acceptable if isinstance(a, str)]

    # Check if normalized answer matches any acceptable form
    is_correct = norm_answer in acceptable_norm or any(norm_answer == cand for cand in acceptable_norm)

    # Loose containment fallback
    if not is_correct:
        is_correct = any(norm_answer in cand or cand in norm_answer for cand in acceptable_norm)

    if is_correct:
        if do_print:
            print(f"Correct via string matching!")
        score_wo_penalty = correct_score
        if penalize_hint:
            score_w_penalty = correct_score * (1 - hint_penalty * num_hints)
        else:
            score_w_penalty = correct_score
        return {
            "score": score_w_penalty,
            "score_wo_hint_penalty": score_wo_penalty,
            "num_hints": num_hints
        }

    # Answer was extracted and formatted, but incorrect
    if do_print:
        print(f"Incorrect answer -> format_score={format_score}")

    return {
        "score": format_score,
        "score_wo_hint_penalty": format_score,
        "num_hints": num_hints
    }


def compute_score_abstain(
    data_source,
    solution_str: str,
    ground_truth: Dict,
    extra_info=None,
    **kwargs
) -> Dict[str, float]:
    """
    Variant that rewards abstention.
    Sets abstain_score to a positive value.
    """
    return compute_score(
        data_source,
        solution_str,
        ground_truth,
        extra_info,
        format_score=0.1,
        correct_score=1.0,
        incorrect_score=0.0,
        abstain_score=0.3,  # Positive reward for abstaining
        reward_abstain=True,
        **kwargs
    )


def compute_score_hint(
    data_source,
    solution_str: str,
    ground_truth: Dict,
    extra_info=None,
    **kwargs
) -> Dict[str, float]:
    """
    Variant that penalizes hint usage.
    Reduces score proportionally to number of hints used.
    """
    return compute_score(
        data_source,
        solution_str,
        ground_truth,
        extra_info,
        format_score=0.1,
        correct_score=1.0,
        incorrect_score=0.0,
        abstain_score=0.0,
        penalize_hint=True,
        hint_penalty=0.1,  # 10% penalty per hint
        **kwargs
    )
