#!/usr/bin/env python3
"""
Verify that the Connections RL data is properly formatted for VERL training.
"""
import sys
import json
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    import pyarrow.parquet as pq
except ImportError:
    print("Error: pandas and pyarrow required. Install with: pip install pandas pyarrow")
    sys.exit(1)


def verify_parquet_file(filepath: Path, data_type: str = "train"):
    """Verify the structure of a parquet file."""
    print(f"\n{'='*60}")
    print(f"Verifying {data_type} data: {filepath}")
    print('='*60)

    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return False

    try:
        # Read parquet
        df = pd.read_parquet(filepath)

        print(f"✓ File loaded successfully")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {df.columns.tolist()}")

        # Check required columns
        required_cols = ['prompt', 'reward_model', 'index']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return False

        print(f"✓ All required columns present")

        # Check first row
        first_row = df.iloc[0]

        # Check prompt format (should be list of messages)
        prompt = first_row['prompt']

        # Handle numpy array conversion (parquet may serialize lists as arrays)
        import numpy as np
        if isinstance(prompt, np.ndarray):
            prompt = prompt.tolist()

        if not isinstance(prompt, list):
            print(f"❌ 'prompt' should be a list of messages, got {type(prompt)}")
            return False

        if len(prompt) == 0:
            print(f"❌ 'prompt' list is empty")
            return False

        if not isinstance(prompt[0], dict):
            print(f"❌ 'prompt' messages should be dicts, got {type(prompt[0])}")
            return False

        if 'role' not in prompt[0] or 'content' not in prompt[0]:
            print(f"❌ 'prompt' messages should have 'role' and 'content' keys")
            print(f"   Found keys: {prompt[0].keys()}")
            return False

        print(f"✓ Prompt format is valid (list of message dicts)")
        print(f"  First message role: {prompt[0]['role']}")
        print(f"  First message content preview: {prompt[0]['content'][:100]}...")

        # Check reward_model format
        reward_model = first_row['reward_model']

        # If it's a JSON string, parse it
        if isinstance(reward_model, str):
            try:
                reward_model = json.loads(reward_model)
            except json.JSONDecodeError:
                print(f"❌ 'reward_model' is a string but not valid JSON")
                return False

        if not isinstance(reward_model, dict):
            print(f"❌ 'reward_model' should be a dict, got {type(reward_model)}")
            return False

        if 'ground_truth' not in reward_model:
            print(f"❌ 'reward_model' should have 'ground_truth' key")
            print(f"   Found keys: {reward_model.keys()}")
            return False

        ground_truth = reward_model['ground_truth']
        if not isinstance(ground_truth, dict):
            print(f"❌ 'ground_truth' should be a dict, got {type(ground_truth)}")
            return False

        # Check for solution_text or answers
        has_solution = 'solution_text' in ground_truth
        has_answers = 'answers' in ground_truth

        if not (has_solution or has_answers):
            print(f"❌ 'ground_truth' should have 'solution_text' or 'answers'")
            print(f"   Found keys: {ground_truth.keys()}")
            return False

        print(f"✓ Reward model format is valid")
        if has_solution:
            print(f"  Solution text preview: {ground_truth['solution_text'][:100]}...")
        if has_answers:
            print(f"  Number of answer groups: {len(ground_truth['answers'])}")
            if ground_truth['answers']:
                print(f"  First group: {ground_truth['answers'][0]}")

        # Check index
        index = first_row['index']
        # Accept Python int/float and numpy numeric types
        if not isinstance(index, (int, float, np.integer, np.floating)):
            print(f"❌ 'index' should be numeric, got {type(index)}")
            return False

        print(f"✓ Index is valid (value: {index})")

        # Sample a few more rows to ensure consistency
        sample_size = min(5, len(df))
        print(f"\n✓ Checking {sample_size} sample rows...")

        for i in range(sample_size):
            row = df.iloc[i]

            # Handle prompt
            row_prompt = row['prompt']
            if isinstance(row_prompt, np.ndarray):
                row_prompt = row_prompt.tolist()
            if not isinstance(row_prompt, list):
                print(f"❌ Row {i}: prompt is not a list")
                return False

            # Handle reward_model
            row_reward = row['reward_model']
            if isinstance(row_reward, str):
                try:
                    row_reward = json.loads(row_reward)
                except json.JSONDecodeError:
                    print(f"❌ Row {i}: reward_model is not valid JSON")
                    return False
            if not isinstance(row_reward, dict):
                print(f"❌ Row {i}: reward_model is not a dict")
                return False

        print(f"✓ All sampled rows have correct format")

        print(f"\n{'='*60}")
        print(f"✅ {data_type.upper()} DATA VERIFIED SUCCESSFULLY")
        print(f"{'='*60}")

        return True

    except Exception as e:
        print(f"❌ Error reading file: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    data_dir = root_dir / "data" / "connections" / "artifacts"

    train_file = data_dir / "rl_train.parquet"
    val_file = data_dir / "rl_val.parquet"

    print("="*60)
    print("VERL Connections Data Verification")
    print("="*60)
    print(f"Data directory: {data_dir}")

    train_ok = verify_parquet_file(train_file, "train")
    val_ok = verify_parquet_file(val_file, "validation")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    print(f"Train data: {'✅ PASS' if train_ok else '❌ FAIL'}")
    print(f"Val data:   {'✅ PASS' if val_ok else '❌ FAIL'}")
    print("="*60)

    if train_ok and val_ok:
        print("\n✅ All data files are properly formatted for VERL training!")
        print("\nYou can now run:")
        print("  cd verl/recipe/connections")
        print("  bash run_connections.sh")
        return 0
    else:
        print("\n❌ Some data files have issues. Please fix them before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
