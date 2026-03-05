"""Reward functions for competition_math task."""

import re
import random


def extract_answer(solution_str: str) -> str | None:
    """Extract answer from <answer>...</answer> tags."""
    pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(pattern, solution_str, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return None


def has_abstain_tag(solution_str: str) -> bool:
    """Check if the solution ends with the abstention pattern: </think>\\n\\n<abstain>"""
    return solution_str.rstrip().endswith("</think>\n\n<abstain>")


def check_answer(predicted: str, correct: str) -> bool:
    """Check if predicted answer matches correct answer using math-verify."""
    if not predicted or not correct:
        return False

    from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig

    # Primary: parse both as LaTeX (wrapped in $...$)
    try:
        gold_parsed = parse(f"${correct}$", extraction_config=[LatexExtractionConfig()])
    except Exception:
        gold_parsed = []

    try:
        pred_parsed = parse(f"${predicted}$", extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
    except Exception:
        pred_parsed = []

    if gold_parsed and pred_parsed:
        try:
            if verify(gold_parsed, pred_parsed):
                return True
        except Exception:
            pass

    # Fallback: try both as plain expressions
    try:
        gold_plain = parse(correct, extraction_config=[ExprExtractionConfig()])
        pred_plain = parse(predicted, extraction_config=[ExprExtractionConfig()])
        if gold_plain and pred_plain:
            return verify(gold_plain, pred_plain)
    except Exception:
        pass

    return False


def has_malformed_structure(solution_str: str) -> bool:
    """Validate the overall tag structure of the response.

    The response starts inside an open <think> block (from assistant prefix).
    Valid structure:
        ([text]</think><request></request><response>...</response><think>)*
        [text]</think>\\n\\n(<answer>...</answer> | <abstain>)

    Validates:
    - Correct tag sequence (state machine)
    - <request> tags are tight (no content inside)
    - No spurious/duplicate tags (e.g. double </think>)

    Returns:
        True if the structure is malformed, False if valid.
    """
    # Content check: <request> tags must be tight (no content inside)
    if solution_str.count('<request>') != len(re.findall(r'<request></request>', solution_str)):
        return True

    # Extract all structural tags in order
    tag_pattern = r'(</think>|<think>|<request>|</request>|<response>|</response>|<answer>|</answer>|<abstain>)'
    tags = re.findall(tag_pattern, solution_str)

    if not tags:
        return True

    i = 0
    while i < len(tags):
        if tags[i] != '</think>':
            return True
        i += 1

        if i >= len(tags):
            return True

        if tags[i] == '<request>':
            expected = ['<request>', '</request>', '<response>', '</response>', '<think>']
            for expected_tag in expected:
                if i >= len(tags) or tags[i] != expected_tag:
                    return True
                i += 1
        elif tags[i] == '<answer>':
            if i + 1 >= len(tags) or tags[i + 1] != '</answer>':
                return True
            i += 2
            return i != len(tags)
        elif tags[i] == '<abstain>':
            i += 1
            return i != len(tags)
        else:
            return True

    return True


def get_num_hints(solution_str: str) -> int:
    """Count all hint request/response exchanges (including exhausted ones)."""
    responses = re.findall(r'<response>(.*?)</response>', solution_str, re.DOTALL)
    return len(responses)


def compute_score(
    data_source,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict,
    format_score: float = 0.0,
    score: float = 1.0,
    reward_abstain: bool = False,
    abstention_score: float = 0.5,
    penalize_hint: bool = False,
    hint_penalty: float = 0.1,
    hint_bonus: float = 0.0,
    **kwargs,
) -> dict:
    """
    Compute reward score for competition_math task.

    Args:
        data_source: Data source identifier
        solution_str: Model's complete response
        ground_truth: Dict with 'answer' key containing correct answer
        extra_info: Additional context
        format_score: Partial credit for well-formatted but wrong answer
        score: Full score for correct answer
        reward_abstain: Whether to reward abstention
        abstention_score: Score for abstaining
        penalize_hint: Whether to penalize hint usage
        hint_penalty: Penalty per hint used (multiplicative)
        hint_bonus: Bonus added to format_score when hints were used and
            answer is wrong but formatted (default 0.0, no bonus)

    Returns:
        Dict with score and metadata
    """
    correct_answer = ground_truth.get("answer", "")
    do_print = False

    if do_print:
        print(f"--------------------------------")
        print(f"Correct answer: {correct_answer}")
        print(f"Solution: {solution_str[:500]}...")

    # Structural validation: verify entire tag sequence is well-formed
    if has_malformed_structure(solution_str):
        if do_print:
            print(f"Malformed structure detected - awarding 0")
        return {
            "score": 0,
            "score_wo_hint_penalty": 0,
            "num_hints": 0,
            "abstained": False,
            "correct": False,
        }

    num_hints = get_num_hints(solution_str)

    # Check for abstention first
    if reward_abstain and has_abstain_tag(solution_str):
        if do_print:
            print(f"Abstaining - awarding {abstention_score}")
        return {
            "score": abstention_score,
            "score_wo_hint_penalty": abstention_score,
            "num_hints": num_hints,
            "abstained": True,
            "correct": False,
        }

    # Extract predicted answer
    predicted = extract_answer(solution_str)

    if predicted is None:
        if do_print:
            print(f"No answer found")
        return {
            "score": 0,
            "score_wo_hint_penalty": 0,
            "num_hints": num_hints,
            "abstained": False,
            "correct": False,
        }

    # Check correctness
    is_correct = check_answer(predicted, correct_answer)

    if is_correct:
        if do_print:
            print(f"Correct! Predicted: {predicted}")
        base_score = score
        if penalize_hint and num_hints > 0:
            penalized_hints = min(num_hints, 5)
            final_score = base_score * (1 - hint_penalty * penalized_hints)
            final_score = max(final_score, 0)  # Don't go negative
        else:
            final_score = base_score
        return {
            "score": final_score,
            "score_wo_hint_penalty": base_score,
            "num_hints": num_hints,
            "abstained": False,
            "correct": True,
        }
    else:
        final_format_score = format_score
        if hint_bonus > 0 and num_hints > 0:
            penalized_hints = min(num_hints, 5)
            final_format_score = format_score + hint_bonus * penalized_hints
        if do_print:
            print(f"Wrong. Predicted: {predicted}, Expected: {correct_answer}")
            if hint_bonus > 0 and num_hints > 0:
                print(f"  Hint bonus applied: {format_score} + {hint_bonus}*{num_hints} = {final_format_score}")
        return {
            "score": final_format_score,
            "score_wo_hint_penalty": final_format_score,
            "num_hints": num_hints,
            "abstained": False,
            "correct": False,
        }


def compute_score_abstain(
    data_source,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict,
    abstention_score: float = 0.5,
    **kwargs,
) -> dict:
    """
    Reward function that gives partial credit for abstention.

    Abstaining is rewarded with abstention_score (default 0.5),
    which is better than wrong (0) but worse than correct (1).
    """
    return compute_score(
        data_source,
        solution_str,
        ground_truth,
        extra_info,
        reward_abstain=True,
        abstention_score=abstention_score,
        **kwargs,
    )


def compute_score_hint(
    data_source,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict,
    hint_penalty: float = 0.1,
    hint_bonus: float = 0.0,
    format_score: float = 0.1,
    **kwargs,
) -> dict:
    """
    Reward function that penalizes hint usage.

    Each hint used reduces the score by hint_penalty (multiplicative).
    Final score = base_score * (1 - hint_penalty * num_hints)

    If hint_bonus > 0, wrong-but-formatted answers that used hints
    get format_score + hint_bonus (encourages hint exploration).
    """
    return compute_score(
        data_source,
        solution_str,
        ground_truth,
        extra_info,
        penalize_hint=True,
        hint_penalty=hint_penalty,
        hint_bonus=hint_bonus,
        format_score=format_score,
        **kwargs,
    )



