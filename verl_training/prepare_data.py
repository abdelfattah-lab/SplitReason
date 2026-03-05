"""
Convert math datasets to veRL parquet format.

Supports:
  - open-r1/OpenR1-Math-220k (filtered by correctness_llama)
  - qwedsacf/competition_math (no filtering needed)

Maps each example to the veRL schema: data_source, prompt (chat format),
ability, reward_model, extra_info.

Writes parquet via pandas (not HF datasets.to_parquet) to avoid embedding
incompatible HF metadata that older datasets versions can't read.

Usage:
    python verl_training/prepare_data.py \
        --output_dir ~/data/splitreason \
        [--phase 1|2] \
        [--dataset_name qwedsacf/competition_math]
"""

import argparse
import json
import os

import datasets
import pandas as pd

# System prompts ----------------------------------------------------------
# Phase 1: vanilla GRPO, no bigmodel instructions
SYSTEM_PROMPT_PHASE1 = (
    "You are a helpful AI Assistant that provides well-reasoned and detailed "
    "responses. You first think about the reasoning process as an internal "
    "monologue and then provide the user with the answer. Respond in the "
    "following format: <think>\n...\n</think>\n<answer>\n...\n</answer>. \n "
    r"Put your final answer within \boxed{}."
)

# Phase 2+: includes bigmodel tag instructions
SYSTEM_PROMPT_PHASE2 = (
    "You are a helpful AI Assistant that provides well-reasoned and detailed "
    "responses. You first think about the reasoning process as an internal "
    "monologue and then provide the user with the answer. Respond in the "
    "following format: <think>\n...\n</think>\n<answer>\n...\n</answer>. \n "
    r"Put your final answer within \boxed{}. "
    "You always use <bigmodel>...</bigmodel> to mark parts of the reasoning "
    "process that are important."
)


def build_rows(hf_dataset, split: str, system_prompt: str, data_source: str = "open-r1/OpenR1-Math-220k") -> list[dict]:
    """Convert HF dataset split to list of veRL-format dicts."""
    rows = []
    for idx, example in enumerate(hf_dataset):
        question = example["problem"]
        solution = example.get("solution", "")

        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        rows.append({
            "data_source": data_source,
            "prompt": prompt,
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution,
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "solution": solution,
                "problem": question,
            },
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Prepare veRL parquet data from OpenR1-Math-220k")
    parser.add_argument("--output_dir", type=str, default=os.path.expanduser("~/data/splitreason"))
    parser.add_argument(
        "--phase",
        type=int,
        default=1,
        choices=[1, 2],
        help="Phase 1: no bigmodel instructions; Phase 2+: with bigmodel instructions",
    )
    parser.add_argument("--dataset_name", type=str, default="qwedsacf/competition_math")
    args = parser.parse_args()

    system_prompt = SYSTEM_PROMPT_PHASE1 if args.phase == 1 else SYSTEM_PROMPT_PHASE2

    print(f"Loading dataset: {args.dataset_name}")
    dataset = datasets.load_dataset(args.dataset_name)

    # Dataset-specific filtering
    if "correctness_llama" in dataset["train"].column_names:
        print("Filtering by correctness_llama ...")
        dataset = dataset.filter(
            lambda ex: (
                isinstance(ex["correctness_llama"], list)
                and len(ex["correctness_llama"]) > 0
                and all(ex["correctness_llama"])
            ),
            num_proc=os.cpu_count(),
        )

    if "level" in dataset["train"].column_names:
        print("Filtering to Level 5 problems only ...")
        dataset = dataset.filter(
            lambda ex: ex["level"] == "Level 5",
            num_proc=os.cpu_count(),
        )

    os.makedirs(args.output_dir, exist_ok=True)

    # Convert and write train split via pandas
    print(f"Converting to veRL format (phase {args.phase}) ...")
    train_rows = build_rows(dataset["train"], "train", system_prompt, data_source=args.dataset_name)
    train_df = pd.DataFrame(train_rows)
    train_path = os.path.join(args.output_dir, f"train_phase{args.phase}.parquet")
    train_df.to_parquet(train_path)
    print(f"Wrote {len(train_df)} train examples to {train_path}")

    # If there's a test split, process it too
    if "test" in dataset:
        test_rows = build_rows(dataset["test"], "test", system_prompt, data_source=args.dataset_name)
        test_df = pd.DataFrame(test_rows)
        test_path = os.path.join(args.output_dir, f"test_phase{args.phase}.parquet")
        test_df.to_parquet(test_path)
        print(f"Wrote {len(test_df)} test examples to {test_path}")


if __name__ == "__main__":
    main()
