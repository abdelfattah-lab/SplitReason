#!/usr/bin/env bash
set -euo pipefail

# Driver script for RL on Knights & Knaves puzzles.
# Expected data format: each row has `prompt` (list of chat messages) and ground-truth fields
# such as `solution_text` / `solution_text_format` used by the reward function. Convert your
# JSONL to parquet/arrow with chat-formatted prompts before running.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/../.. && pwd)"

DATA_TRAIN="${DATA_TRAIN:-$ROOT_DIR/data/knights_and_knaves_rl/train.parquet}"
DATA_VAL="${DATA_VAL:-$ROOT_DIR/data/knights_and_knaves_rl/val.parquet}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-knights_and_knaves_rl}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/$EXPERIMENT_NAME}"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$DATA_TRAIN" \
  data.val_files="$DATA_VAL" \
  data.prompt_key=prompt \
  data.max_prompt_length=2048 \
  data.max_response_length=1024 \
  custom_reward_function.path="$ROOT_DIR/verl/recipe/knights_and_knaves/reward_function.py" \
  custom_reward_function.name=compute_score \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_micro_batch_size=32 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  trainer.project_name=knights_and_knaves \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.rollout_data_dir="$OUTPUT_DIR" \
  trainer.save_freq=50 \
  trainer.test_freq=50 \
  trainer.total_training_steps=200
