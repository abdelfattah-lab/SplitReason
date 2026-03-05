#!/usr/bin/env bash
set -euo pipefail

# Disable FlashAttention2 - use eager attention instead
export TRANSFORMERS_ATTN_IMPLEMENTATION=eager

# Driver script for RL on Connections puzzles with pay-per-search/abstention support.
# Based on the countdown recipe pattern with epsilon-greedy exploration and hint penalties.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/connections/artifacts}"

# Data paths
DATA_TRAIN="${DATA_TRAIN:-$DATA_DIR/rl_train.parquet}"
DATA_VAL="${DATA_VAL:-$DATA_DIR/rl_val.parquet}"

# Model configuration
MODEL_FAMILY="${MODEL_FAMILY:-Qwen}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-1.5B}"
MODEL_PATH="${MODEL_PATH:-$MODEL_FAMILY/$MODEL_NAME}"

# Training hyperparameters
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
LR="${LR:-1e-6}"

# Pay-per-search / abstention settings
ALLOW_HINT="${ALLOW_HINT:-true}"
USE_EPSILON="${USE_EPSILON:-true}"
EPSILON_A="${EPSILON_A:-0.6}"       # Epsilon parameter for exploration
EPSILON_K="${EPSILON_K:-12}"        # Epsilon decay parameter

# Reward function selection
# Options: compute_score (basic), compute_score_abstain (rewards abstention), compute_score_hint (penalizes hints)
REWARD_FUNCTION="${REWARD_FUNCTION:-compute_score_hint}"

# Experiment naming
TIMESTAMP="$(date +%F)"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${TIMESTAMP}_${MODEL_NAME}_lr${LR}_bs${TRAIN_BATCH_SIZE}_hint${ALLOW_HINT}_eps${USE_EPSILON}_eA${EPSILON_A}_ek${EPSILON_K}_${REWARD_FUNCTION}}"

# Output configuration
OUTPUT_PATH_BASE="${OUTPUT_PATH_BASE:-$ROOT_DIR/outputs/connections}"
OUTPUT_PATH="${OUTPUT_PATH:-$OUTPUT_PATH_BASE/$EXPERIMENT_NAME}"

# Create output directory
mkdir -p "$OUTPUT_PATH"
mkdir -p "$OUTPUT_PATH/logs"

echo "=========================================="
echo "VERL RL Training - Connections"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Train data: $DATA_TRAIN"
echo "Val data: $DATA_VAL"
echo "Output: $OUTPUT_PATH"
echo "Experiment: $EXPERIMENT_NAME"
echo "Reward function: $REWARD_FUNCTION"
echo "Allow hints: $ALLOW_HINT"
echo "Epsilon-greedy: $USE_EPSILON (A=$EPSILON_A, k=$EPSILON_K)"
echo "=========================================="

# Run VERL PPO trainer
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$DATA_TRAIN" \
  data.val_files="$DATA_VAL" \
  data.prompt_key=prompt \
  data.train_batch_size=$TRAIN_BATCH_SIZE \
  data.val_batch_size=256 \
  data.max_prompt_length=1024 \
  data.max_response_length=2048 \
  custom_reward_function.path="$ROOT_DIR/verl/recipe/connections/reward_function.py" \
  custom_reward_function.name=$REWARD_FUNCTION \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.optim.lr=$LR \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size=32 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
  +actor_rollout_ref.model.override_config.attn_implementation=eager \
  +critic.model.override_config.attn_implementation=eager \
  algorithm.kl_ctrl.kl_coef=0.001 \
  allow_hint=$ALLOW_HINT \
  trainer.use_epsilon=$USE_EPSILON \
  trainer.epsilon_k=$EPSILON_K \
  trainer.epsilon_A=$EPSILON_A \
  trainer.logger=['wandb'] \
  trainer.project_name=connections-abstention \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir="$OUTPUT_PATH" \
  trainer.rollout_data_dir="$OUTPUT_PATH" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=25 \
  trainer.test_freq=25 \
  trainer.resume_mode=disable \
  trainer.log_val_generations=25 \
  trainer.total_training_steps=200 \
  2>&1 | tee "$OUTPUT_PATH/output.log"

echo "Training complete. Output saved to: $OUTPUT_PATH"
