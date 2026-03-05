import re
import random
import ast
import operator
from collections import Counter


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    """if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    solution_str = solution_str.split('\n')[-1]"""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer

def abstain_equation(equation_str):
    """Check if the equation indicates abstention."""
    if equation_str is None:
        return False
    return equation_str.strip().lower() == "i abstain"


def has_abstain_tag(solution_str):
    """Check if the solution ends with the abstention pattern: </think>\\n\\n<abstain>"""
    return solution_str.rstrip().endswith("</think>\n\n<abstain>")


def validate_equation(equation_str, available_numbers):
    """Validate that equation uses all and only the numbers in available_numbers, with exact counts."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]

        # Cast to Python ints (numpy int64 from parquet can cause Counter mismatch)
        available_numbers = [int(n) for n in available_numbers]

        # Compare exact counts
        return Counter(numbers_in_eq) == Counter(available_numbers)
    except:
        return False


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().=\s]+$'
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")
        
        def strip_trailing_result(equation_str):
            """
            If the equation ends with '= <number>' (optionally with whitespace), remove it.
            """
            return re.sub(r'\s*=\s*\d+\s*$', '', equation_str)

        # Evaluate the equation with restricted globals and locals
        result = eval(strip_trailing_result(equation_str), {"__builtins__": None}, {})
        return result
    except Exception as e:
        return None
    
def get_num_hints(solution_str):
    """Count all hint request/response exchanges (including exhausted ones)."""
    responses = re.findall(r'<response>(.*?)</response>', solution_str, re.DOTALL)
    return len(responses)


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
        return True  # No tags at all

    # Validate tag sequence with a state machine
    # Expected: (</think> <request> </request> <response> </response> <think>)* </think> (<answer> </answer> | <abstain>)
    i = 0
    while i < len(tags):
        # Must start each cycle with </think> (closing current think block)
        if tags[i] != '</think>':
            return True
        i += 1

        if i >= len(tags):
            return True  # Ended after </think> without terminal tag

        if tags[i] == '<request>':
            # Hint exchange: <request> </request> <response> </response> <think>
            expected = ['<request>', '</request>', '<response>', '</response>', '<think>']
            for expected_tag in expected:
                if i >= len(tags) or tags[i] != expected_tag:
                    return True
                i += 1
            # Loop back to expect next </think>
        elif tags[i] == '<answer>':
            # Terminal: <answer> </answer>
            if i + 1 >= len(tags) or tags[i + 1] != '</answer>':
                return True
            i += 2
            return i != len(tags)  # Malformed if extra tags after
        elif tags[i] == '<abstain>':
            i += 1
            return i != len(tags)  # Malformed if extra tags after
        else:
            return True  # Unexpected tag after </think>

    return True  # Ran out of tags without proper termination


def compute_score(data_source, solution_str, ground_truth, extra_info, method='strict', format_score=0.1, score=1., reward_abstain=False, abstention_score=0.3, penalize_hint=False, hint_penalty=0.2, hint_bonus=0.0, **kwargs):
    """The scoring function for countdown
    """
    #format_score = 0
    target = ground_truth['target']
    numbers = ground_truth['numbers']

    equation = extract_solution(solution_str=solution_str)
    do_print = False

    """if len(solution_str.split()) < 200:
        return 0"""

    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    # Structural validation: verify entire tag sequence is well-formed
    if has_malformed_structure(solution_str):
        if do_print:
            print(f"Malformed structure detected - awarding 0")
        return {"score": 0, "score_wo_hint_penalty": 0, "num_hints": 0, "abstained": False, "malformed": True, "correct": False}

    num_hints = get_num_hints(solution_str)

    # Check for abstention first (before equation extraction check)
    if reward_abstain and has_abstain_tag(solution_str):
        if do_print:
            print(f"Abstaining (via <abstain> tag) - awarding {abstention_score}")
        return {
            "score": abstention_score,
            "score_wo_hint_penalty": abstention_score,
            "num_hints": num_hints,
            "abstained": True,
            "malformed": False,
            "correct": False,
        }

    if equation is None:
        if do_print:
            print(f"No equation found or length too short")
        return {"score": 0, "score_wo_hint_penalty": 0, "num_hints": num_hints, "abstained": False, "malformed": False, "correct": False}

    # Check for abstention via answer content (legacy support)
    if reward_abstain and abstain_equation(equation):
        if do_print:
            print(f"Abstaining (via answer) - awarding {abstention_score}")
        return {
            "score": abstention_score,
            "score_wo_hint_penalty": abstention_score,
            "num_hints": num_hints,
            "abstained": True,
            "malformed": False,
            "correct": False,
        }

    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation")
        return {"score": format_score, "score_wo_hint_penalty": format_score, "num_hints": num_hints, "abstained": False, "malformed": False, "correct": False}

    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return {"score": format_score, "score_wo_hint_penalty": format_score, "num_hints": num_hints, "abstained": False, "malformed": False, "correct": False}

        if abs(result - target) < 1e-5:  # Account for floating point precision
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            if penalize_hint:
                penalized_hints = min(num_hints, 5)
                final_score = score * (1 - hint_penalty * penalized_hints)
            else:
                final_score = score
            return {"score": final_score, "score_wo_hint_penalty": score, "num_hints": num_hints, "abstained": False, "malformed": False, "correct": True}
        else:
            final_format_score = format_score
            if hint_bonus > 0 and num_hints > 0:
                penalized_hints = min(num_hints, 5)
                final_format_score = format_score + hint_bonus * penalized_hints
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
                if hint_bonus > 0 and num_hints > 0:
                    print(f"  Hint bonus applied: {format_score} + {hint_bonus}*{num_hints} = {final_format_score}")
            return {"score": final_format_score, "score_wo_hint_penalty": final_format_score, "num_hints": num_hints, "abstained": False, "malformed": False, "correct": False}
    except:
        if do_print:
            print(f"Error evaluating equation")
        return {"score": format_score, "score_wo_hint_penalty": format_score, "num_hints": num_hints, "abstained": False, "malformed": False, "correct": False}



def compute_score_abstain(data_source, solution_str, ground_truth, extra_info, method='strict', format_score=0.1, score=1., abstention_score=0.5, **kwargs):
    """The scoring function for countdown that rewards abstention.

    Args:
        abstention_score: Reward for abstaining (default 0.5, between wrong and correct).
    """
    return compute_score(
        data_source, solution_str, ground_truth, extra_info,
        method=method, format_score=format_score, score=score,
        reward_abstain=True, abstention_score=abstention_score, **kwargs
    )

def compute_score_hint(data_source, solution_str, ground_truth, extra_info, method='strict', format_score=0.1, score=1., hint_penalty=0.1, hint_bonus=0.0, **kwargs):
    """The scoring function for countdown that penalizes hint usage.

    Args:
        hint_penalty: Multiplicative penalty per hint (default 0.1).
            Final score = accuracy * (1 - hint_penalty * num_hints)
        hint_bonus: Bonus added to format_score when hints were used and
            answer is wrong but formatted (default 0.0, no bonus).
    """
    return compute_score(
        data_source, solution_str, ground_truth, extra_info,
        method=method, format_score=format_score, score=score,
        penalize_hint=True, hint_penalty=hint_penalty, hint_bonus=hint_bonus, **kwargs
    )