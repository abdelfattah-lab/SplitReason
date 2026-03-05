"""Reward function for ZebraLogic RL training."""

import re
import random


def extract_answer(solution_str):
    """Extract answer from <answer>...</answer> tags."""
    pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(pattern, solution_str, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return None


def has_malformed_structure(solution_str):
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


def check_correctness(predicted, ground_truth):
    """
    Check if predicted answer matches the correct choice.

    Validates:
    1. Letter choice (A, B, C, D...) matches correct answer's position
    2. Or exact text match with the answer

    Args:
        predicted: The extracted answer string
        ground_truth: Dict with 'answer' (correct text) and 'choices' (list)

    Returns:
        (is_correct, metadata_dict)
    """
    if predicted is None:
        return False, {"error": "no_answer_tag"}

    correct_answer = ground_truth["answer"]
    choices = ground_truth["choices"]

    # Find correct answer's letter index
    try:
        correct_idx = choices.index(correct_answer)
        correct_letter = chr(65 + correct_idx)  # A, B, C, D...
    except ValueError:
        correct_letter = None

    # Normalize predicted answer
    predicted_normalized = predicted.lower().strip()

    # Check for exact text match (case-insensitive)
    if predicted_normalized == correct_answer.lower():
        return True, {
            "predicted_answer": predicted,
            "correct_answer": correct_answer,
            "match_type": "exact",
        }

    # Check for letter match (e.g., "A", "A.", "a")
    predicted_letter = predicted.upper().strip().rstrip(".")
    if len(predicted_letter) == 1 and predicted_letter.isalpha():
        if predicted_letter == correct_letter:
            return True, {
                "predicted_answer": predicted,
                "correct_answer": correct_answer,
                "match_type": "letter",
            }

        # Valid letter but wrong choice
        letter_idx = ord(predicted_letter) - 65
        if 0 <= letter_idx < len(choices):
            return False, {
                "predicted_answer": predicted,
                "predicted_choice": choices[letter_idx],
                "correct_answer": correct_answer,
                "error": "wrong_choice",
            }

    # Check if predicted text matches any choice exactly
    for i, choice in enumerate(choices):
        if predicted_normalized == choice.lower():
            is_correct = (choice == correct_answer)
            return is_correct, {
                "predicted_answer": predicted,
                "predicted_choice": choice,
                "correct_answer": correct_answer,
                "match_type": "choice_text" if is_correct else None,
                "error": None if is_correct else "wrong_choice",
            }

    return False, {
        "predicted_answer": predicted,
        "correct_answer": correct_answer,
        "error": "no_match",
    }


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info,
    format_score=0.0,
    score=1.0,
    **kwargs,
):
    """
    Reward function for ZebraLogic.

    Args:
        data_source: Task name
        solution_str: Model's full generation
        ground_truth: Dict with answer, choices, etc.
        extra_info: Additional info (unused)
        format_score: Score for valid letter but wrong choice (default 0.0)
        score: Score for correct answer

    Returns:
        Dict with score and metadata (consistent keys for all outcomes)
    """
    do_print = random.randint(1, 64) == 1

    predicted = extract_answer(solution_str)

    if do_print:
        print(f"--------------------------------")
        print(f"Expected: {ground_truth.get('answer')}")
        print(f"Choices: {ground_truth.get('choices')}")
        print(f"Extracted: {predicted}")
        print(f"Solution: {solution_str[:500]}...")

    # Structural validation: verify entire tag sequence is well-formed
    if has_malformed_structure(solution_str):
        if do_print:
            print(f"Malformed structure detected - awarding 0")
        return {
            "score": 0,
            "correct": False,
            "error": "malformed_structure",
            "predicted_answer": None,
            "abstained": False,
        }

    # Base result with consistent keys
    result = {
        "score": 0,
        "correct": False,
        "error": None,
        "predicted_answer": None,
        "abstained": False,
    }

    if predicted is None:
        if do_print:
            print(f"No answer found")
        result["error"] = "no_answer_tag"
        return result

    is_correct, meta = check_correctness(predicted, ground_truth)

    result["correct"] = is_correct
    result["predicted_answer"] = meta.get("predicted_answer")
    result["error"] = meta.get("error")

    if is_correct:
        result["score"] = score
        if do_print:
            print(f"Correct! Match type: {meta.get('match_type')}")
    elif meta.get("error") == "wrong_choice":
        # Valid letter/choice format but wrong answer - give format_score
        result["score"] = format_score
        if do_print:
            print(f"Wrong choice (valid format). Score: {format_score}")
    else:
        # Invalid format or no match
        result["score"] = 0
        if do_print:
            print(f"Invalid: {meta.get('error')}")

    return result


def compute_score_abstain(
    data_source,
    solution_str,
    ground_truth,
    extra_info,
    format_score=0.0,
    score=1.0,
    abstention_score=0.5,
    **kwargs,
):
    """
    Reward function for ZebraLogic with abstention support.

    Abstention is valid when the rollout ends with </think>\\n\\n<abstain>.

    Args:
        abstention_score: Reward for abstaining (default 0.5)
    """
    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"[reward_kwargs] format_score={format_score}, abstention_score={abstention_score}, score={score}")

    # Structural validation: verify entire tag sequence is well-formed
    if has_malformed_structure(solution_str):
        if do_print:
            print(f"Malformed structure detected - awarding 0")
        return {
            "score": 0,
            "correct": False,
            "error": "malformed_structure",
            "predicted_answer": None,
            "abstained": False,
        }

    # Extract the answer first
    predicted = extract_answer(solution_str)

    # Check for abstention: rollout must end with </think>\n\n<abstain>
    if solution_str.rstrip().endswith("</think>\n\n<abstain>"):
        if do_print:
            print(f"Abstaining (via </think>\\n\\n<abstain>) - awarding {abstention_score}")
        return {
            "score": abstention_score,
            "correct": False,
            "error": None,
            "predicted_answer": None,
            "abstained": True,
        }

    # Otherwise, use standard scoring
    return compute_score(
        data_source,
        solution_str,
        ground_truth,
        extra_info,
        format_score=format_score,
        score=score,
        **kwargs,
    )
