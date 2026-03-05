"""Reward function for Chess Puzzles RL training."""

import re
import random


def extract_answer(solution_str):
    """Extract answer from <answer>...</answer> tags."""
    pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(pattern, solution_str, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return None


def check_correctness(predicted, ground_truth):
    """
    Check if predicted move matches the winning move.

    Validates:
    1. Move is in valid UCI format (e.g., "e2e4", "e7e8q")
    2. Move matches the expected winning move
    """
    if predicted is None:
        return False, {"error": "no_answer_tag"}

    # Clean and normalize the answer (remove spaces, lowercase)
    predicted_move = predicted.strip().lower().replace(" ", "")
    expected_move = ground_truth["winning_move"].strip().lower()

    # Check if predicted move is in valid UCI format
    uci_pattern = r'^[a-h][1-8][a-h][1-8][qrbn]?$'
    if not re.match(uci_pattern, predicted_move):
        return False, {"error": "invalid_uci_format", "predicted_move": predicted}

    # Check if move matches expected winning move
    is_correct = (predicted_move == expected_move)

    if is_correct:
        return True, {"predicted_move": predicted_move}
    else:
        return False, {"error": "wrong_move", "predicted_move": predicted_move}


def get_num_hints(solution_str):
    """Count valid hint requests (<request></request> with nothing inside)."""
    return len(re.findall(r'<request></request>', solution_str))


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
    Reward function for Chess Puzzles.

    Args:
        data_source: Task name
        solution_str: Model's full generation
        ground_truth: Dict with winning_move, fen, etc.
        extra_info: Additional info (unused)
        format_score: Score for valid UCI format but wrong move (default 0.0)
        score: Score for correct answer

    Returns:
        Dict with score and metadata (consistent keys for all outcomes)
    """
    do_print = random.randint(1, 64) == 1

    predicted = extract_answer(solution_str)

    if do_print:
        print(f"--------------------------------")
        print(f"Expected move: {ground_truth.get('winning_move')}")
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
            "predicted_move": None,
            "abstained": False,
        }

    # Base result with consistent keys
    result = {
        "score": 0,
        "correct": False,
        "error": None,
        "predicted_move": None,
        "abstained": False,
    }

    if predicted is None:
        if do_print:
            print(f"No answer found")
        result["error"] = "no_answer_tag"
        return result

    is_correct, meta = check_correctness(predicted, ground_truth)

    result["correct"] = is_correct
    result["predicted_move"] = meta.get("predicted_move")
    result["error"] = meta.get("error")

    if is_correct:
        result["score"] = score
        if do_print:
            print(f"Correct!")
    elif meta.get("error") == "wrong_move":
        # Valid UCI format but wrong move - give format_score
        result["score"] = format_score
        if do_print:
            print(f"Wrong move (valid format). Score: {format_score}")
    else:
        # Invalid format
        result["score"] = 0
        if do_print:
            print(f"Invalid format: {meta.get('error')}")

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
    Reward function for Chess Puzzles with abstention support.

    Args:
        abstention_score: Reward for abstaining (default 0.5)
    """
    do_print = random.randint(1, 64) == 1

    # Debug: print kwargs on first few calls
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
            "predicted_move": None,
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
            "predicted_move": None,
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


def compute_score_hint(
    data_source,
    solution_str,
    ground_truth,
    extra_info,
    format_score=0.0,
    score=1.0,
    hint_penalty=0.1,
    **kwargs,
):
    """
    Reward function for Chess Puzzles with hint penalty.

    Penalizes hint usage: final_score = score * (1 - hint_penalty * num_hints)

    Args:
        hint_penalty: Multiplicative penalty per hint (default 0.1).
    """
    do_print = random.randint(1, 64) == 1

    # Structural validation: verify entire tag sequence is well-formed
    if has_malformed_structure(solution_str):
        if do_print:
            print(f"Malformed structure detected - awarding 0")
        return {
            "score": 0,
            "score_wo_hint_penalty": 0,
            "num_hints": 0,
            "correct": False,
            "error": "malformed_structure",
            "predicted_move": None,
            "abstained": False,
            "malformed": True,
        }

    num_hints = get_num_hints(solution_str)

    # Get base score from compute_score
    result = compute_score(
        data_source,
        solution_str,
        ground_truth,
        extra_info,
        format_score=format_score,
        score=score,
        **kwargs,
    )

    # Apply hint penalty to correct answers
    base_score = result["score"]
    if result["correct"] and num_hints > 0:
        result["score"] = score * (1 - hint_penalty * num_hints)

    result["score_wo_hint_penalty"] = base_score
    result["num_hints"] = num_hints
    result["malformed"] = False

    if do_print and num_hints > 0:
        print(f"Hints used: {num_hints}, penalty: {hint_penalty}, "
              f"score: {base_score} -> {result['score']}")

    return result
