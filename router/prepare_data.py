"""
Convert math datasets to veRL parquet format for router training.

Uses Phase 1 system prompt (no <bigmodel> instructions) since the router
makes routing decisions externally — the model never sees bigmodel tags.

Usage:
    python router/prepare_data.py --output_dir ~/data/splitreason
"""

import argparse
import os

import datasets
import pandas as pd

# System prompt: vanilla reasoning, no bigmodel instructions
SYSTEM_PROMPT = (
    "You are a helpful AI Assistant that provides well-reasoned and detailed "
    "responses. You first think about the reasoning process as an internal "
    "monologue and then provide the user with the answer. Respond in the "
    "following format: <think>\n...\n</think>\n<answer>\n...\n</answer>. \n "
    r"Put your final answer within \boxed{}."
)


def build_rows(hf_dataset, split: str, data_source: str = "qwedsacf/competition_math") -> list[dict]:
    """Convert HF dataset split to list of veRL-format dicts."""
    rows = []
    for idx, example in enumerate(hf_dataset):
        question = example["problem"]
        solution = example.get("solution", "")

        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
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
    parser = argparse.ArgumentParser(description="Prepare veRL parquet data for router training")
    parser.add_argument("--output_dir", type=str, default=os.path.expanduser("~/data/splitreason"))
    parser.add_argument("--dataset_name", type=str, default="qwedsacf/competition_math")
    args = parser.parse_args()

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

    # Remove MATH500 eval problems to avoid contamination
    print("Removing MATH500 (simplescaling/openaimath) overlap ...")
    math500 = datasets.load_dataset("simplescaling/openaimath", split="test")
    math500_problems = set(ex["problem"].strip() for ex in math500)
    before = len(dataset["train"])
    dataset = dataset.filter(
        lambda ex: ex["problem"].strip() not in math500_problems,
        num_proc=os.cpu_count(),
    )
    after = len(dataset["train"])
    print(f"  Removed {before - after} overlapping problems ({before} -> {after})")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Converting to veRL format (router, no bigmodel instructions) ...")
    train_rows = build_rows(dataset["train"], "train", data_source=args.dataset_name)
    train_df = pd.DataFrame(train_rows)
    train_path = os.path.join(args.output_dir, "train_router.parquet")
    train_df.to_parquet(train_path)
    print(f"Wrote {len(train_df)} train examples to {train_path}")

    if "test" in dataset:
        test_rows = build_rows(dataset["test"], "test", data_source=args.dataset_name)
        test_df = pd.DataFrame(test_rows)
        test_path = os.path.join(args.output_dir, "test_router.parquet")
        test_df.to_parquet(test_path)
        print(f"Wrote {len(test_df)} test examples to {test_path}")


if __name__ == "__main__":
    main()
