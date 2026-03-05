import re
import os
from datasets import Dataset, load_dataset
from random import randint, seed, choice
from typing import List, Tuple, Optional
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse

import random
import operator
import json
from typing import List, Union, Tuple

OPS = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': lambda a, b: a // b if b != 0 and a % b == 0 else None
}
OP_SYMBOLS = list(OPS.keys())

class ExprNode:
    def __init__(self, value: Union[int, str], left=None, right=None):
        self.value = value  # int or operator
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.left is None and self.right is None

    def __str__(self):
        if self.is_leaf():
            return str(self.value)
        return f"({str(self.left)} {self.value} {str(self.right)})"
    
def find_operands_for_target(
    target: int,
    op_symbol: str,
    number_range: Tuple[int, int] = (1, 50),
    max_attempts: int = 1000
) -> Optional[Tuple[int, int]]:
    """
    Given a target number and an operator, find two numbers a and b such that:
    a <op> b == target, with a, b in number_range.
    
    Returns (a, b) if possible, else None.
    """
    op_func = OPS[op_symbol]
    lo, hi = number_range

    for _ in range(max_attempts):
        if op_symbol == '+':
            a = random.randint(lo, hi)
            b = target - a
        elif op_symbol == '-':
            a = random.randint(lo, hi)
            b = a - target
        elif op_symbol == '*':
            if target == 0:  # trivial case
                a = 0
                b = random.randint(lo, hi)
            else:
                b_candidates = [b for b in range(lo, hi + 1) if b != 0 and target % b == 0]
                if not b_candidates:
                    continue
                b = random.choice(b_candidates)
                a = target // b
        elif op_symbol == '/':
            b_candidates = [b for b in range(lo, hi + 1) if b != 0]
            if not b_candidates:
                continue
            b = random.choice(b_candidates)
            a = target * b
        else:
            raise ValueError(f"Unknown operator: {op_symbol}")

        # Check if both a and b are within range
        if lo <= a <= hi and lo <= b <= hi:
            # Verify to be sure
            result = op_func(a, b)
            if result == target:
                return a, b

    return None, None

def build_random_expr_tree(num_numbers, target: int, number_range, max_attempts=100) -> ExprNode:
    if num_numbers == 1:
        return ExprNode(target)

    split = random.randint(1, num_numbers - 1)
    left_nums = split
    right_nums = num_numbers - split

    random.shuffle(OP_SYMBOLS)
    for op_idx in range(4):
        op = OP_SYMBOLS[op_idx]
        left_target, right_target = find_operands_for_target(target, op, number_range)
        if left_target == None:
            continue
        left_tree = build_random_expr_tree(left_nums, left_target, number_range)
        right_tree = build_random_expr_tree(right_nums, right_target, number_range)

        return ExprNode(op, left_tree, right_tree)

    return None

def extract_leaf_numbers(node: ExprNode) -> List[int]:
    if node.is_leaf():
        return [node.value]
    return extract_leaf_numbers(node.left) + extract_leaf_numbers(node.right)


def evaluate_expr_tree(node: ExprNode) -> Union[int, None]:
    if node.is_leaf():
        return node.value
    left_val = evaluate_expr_tree(node.left)
    right_val = evaluate_expr_tree(node.right)
    result = OPS[node.value](left_val, right_val)
    return result

def extract_hints(node: ExprNode) -> List[str]:
    """
    Extracts all intermediate expression hints from an ExprNode,
    ordered from leaves upward (i.e., bottom-up).
    """
    if node.is_leaf():
        return []

    hints = []
    hints += extract_hints(node.left)
    hints += extract_hints(node.right)
    hints.append(str(node))  # include current expression at this node
    return hints


def generate_example(
    number_range=(1, 50),
    target_range=(10, 90),
    num_numbers=4,
    max_attempts=100
) -> Tuple[int, List[int], str]:

    target = random.randint(*target_range)
    tree = build_random_expr_tree(num_numbers, target, number_range)
    if tree == None:
        return None, None, None
    result = evaluate_expr_tree(tree)
    numbers = extract_leaf_numbers(tree)
    hints = extract_hints(tree)

    print(result)
    print(numbers)
    print(str(tree))
    print(hints)

    return result, numbers, str(tree), hints


    """for _ in range(max_attempts):
        numbers = [random.randint(*number_range) for _ in range(num_numbers)]
        random.shuffle(numbers)
        tree = build_random_expr_tree(numbers)
        result = evaluate_expr_tree(tree)
        if result is not None and target_range[0] <= result <= target_range[1]:
            return result, numbers, str(tree)
    return None, None, None"""


def generate_dataset(n: int, num_numbers=4) -> List[dict]:
    data = []
    while len(data) < n:
        target, numbers, expr, hints = generate_example(num_numbers=num_numbers)
        if target:
            if len(set(numbers + [target])) == num_numbers + 1:
                print(target)
                print(numbers)
                data.append({
                    "target": target,
                    "numbers": numbers,
                    "solution_expr": expr,
                    "hint_exprs": hints
                })
    return data

def save_as_jsonl(data, path):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def make_prefix(dp, template_type, abstention=False, hint=False, penalty=False):
    target = dp['target']
    numbers = dp['numbers']
    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        system = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process and then provides the user with the answer."""
        question = f"""Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. You should show your work and how you try different options within the <think> </think> tags. You should return the final equation with these numbers within <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."""
        if abstention:
            question += " If you believe you will not be able to solve the problem at any point of your thinking or reasoning process, you can return <answer> I abstain </answer>"
        if hint:
            question += " If you believe you will not be able to solve the problem, you can ask for a hint by outputting <request></request>."
        assistant = f"""<think> Let me solve this step by step."""
    elif template_type == 'qwen-instruct':
        print("THis template type needs to be implemented")
        exit()
    return system, question, assistant

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/mnt/home/tg436/countdown-expt/data_countdown_scratch_take2/')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--num_samples', type=int, default=100000)
    parser.add_argument('--num_operands', type=int, default=6)
    parser.add_argument('--max_target', type=int, default=1000)
    parser.add_argument('--min_number', type=int, default=1)
    parser.add_argument('--max_number', type=int, default=100)
    parser.add_argument('--train_size', type=int, default=327680)
    parser.add_argument('--test_size', type=int, default=1024)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--data_mix', type=str, default='{"4":"0.5", "5":"0.5"}')
    parser.add_argument('--abstention', action='store_true', help="Include abstention text in prompt.")
    parser.add_argument('--penalty', action='store_true', help="Include penalty text in prompt.")
    parser.add_argument('--hint', action='store_true', help="Include hint text in prompt.")

    args = parser.parse_args()

    data_source = 'countdown'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size
    

    mix = json.loads(args.data_mix)

    def make_map_fn(split):
        def process_fn(example, idx):
            #system, question, assistant = make_prefix_penalty(example, template_type=args.template_type)
            system, question, assistant = make_prefix(example, template_type=args.template_type, abstention=args.abstention, hint=args.hint)
            solution = {
                "target": example['target'],
                "numbers": example['numbers'],
                "solution_expr": example['solution_expr'],
                "hints_expr": example["hint_exprs"]
            }
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": question,
                },
                {
                    "role": "assistant",
                    "content": assistant,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn

    data_train = []
    data_test = []
    for num_numbers in mix.keys():
        
        dataset_size_train = int(TRAIN_SIZE * float(mix[num_numbers]))
        dataset_size_test = int(TEST_SIZE * float(mix[num_numbers]))
        num_numbers = int(num_numbers)

        print(dataset_size_train)

        data_train_curr = generate_dataset(dataset_size_train, num_numbers=num_numbers)
        data_test_curr = generate_dataset(dataset_size_test, num_numbers=num_numbers)

        data_train += data_train_curr
        data_test += data_test_curr

    data_train = Dataset.from_list(data_train).shuffle(seed=42)
    data_test = Dataset.from_list(data_test).shuffle(seed=42)

    data_train = data_train.map(function=make_map_fn('train'), with_indices=True)
    data_test = data_test.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    suffix = '_'.join(mix.keys())
    arg_extra = []
    if args.abstention:
        arg_extra.append('ab')
    if args.hint: 
        arg_extra.append('hint')
    midfix = '_'.join(arg_extra)
    local_dir_path = os.path.join(local_dir, midfix + '__' + suffix)

    if not os.path.isdir(local_dir_path):
        os.mkdir(local_dir_path)

    data_train.to_parquet(os.path.join(local_dir_path, 'train.parquet'))
    data_test.to_parquet(os.path.join(local_dir_path, 'test.parquet'))

    """num_numbers=7
    data = generate_dataset(10000, num_numbers=num_numbers)
    save_as_jsonl(data, f"./countdown_data/{num_numbers}.json")"""