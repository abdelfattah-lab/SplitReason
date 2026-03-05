#!/usr/bin/env bash
# Phase 2: GRPO with Speculative Reasoning (Pipelined Sync SPMD + Big Model HTTP)
#
# GPU allocation:
#   GPU 0-3: 32B big model (standalone vLLM server, TP=4)
#   GPU 4:   1.5B veRL colocated training+rollout
#
# Prerequisites:
#   1. pip install verl requests
#   2. python verl_training/prepare_data.py --phase 2 --output_dir ~/data/splitreason
#   3. python verl_training/prepare_tokenizer.py --output_dir ~/data/splitreason/tokenizer
#   4. Launch the 32B big model server in a separate tmux pane (see below)
#
# Step 1 — Launch big model server (separate tmux pane):
#   CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m vllm.entrypoints.openai.api_server \
#       --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
#       --tensor-parallel-size 4 --gpu-memory-utilization 0.90 \
#       --port 8002 --max-model-len 16384 --dtype bfloat16
#
# Step 2 — Run this script:
#   bash verl_training/run_phase2.sh

set -euo pipefail

# NCCL settings for multi-GPU stability
export TORCH_NCCL_ENABLE_MONITORING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=120000
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=216000
export NCCL_P2P_DISABLE=1

# GPU selection: 1 GPU for training+rollout (colocated)
export CUDA_VISIBLE_DEVICES=4

# WandB config
export WANDB_API_KEY=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
export WANDB_ENTITY=anthonyf1223-cornell-university
unset TRITON_INTERPRET
export RAY_DEDUP_LOGS=0

# Big model server config (read by speculative_rollout.py)
export SPLITREASON_BIG_MODEL_HOST=localhost
export SPLITREASON_BIG_MODEL_PORT=8002
export SPLITREASON_BIG_MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B

DATA_DIR="${HOME}/data/splitreason"
TRAIN_FILE="${DATA_DIR}/train_phase2.parquet"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ ! -f "${TRAIN_FILE}" ]; then
    echo "ERROR: ${TRAIN_FILE} not found. Run prepare_data.py --phase 2 first."
    exit 1
fi

# Verify big model server is reachable
if ! curl -s "http://${SPLITREASON_BIG_MODEL_HOST}:${SPLITREASON_BIG_MODEL_PORT}/v1/models" > /dev/null 2>&1; then
    echo "WARNING: Big model server not reachable at ${SPLITREASON_BIG_MODEL_HOST}:${SPLITREASON_BIG_MODEL_PORT}"
    echo "Make sure to launch the 32B vLLM server first (see script header)."
fi

# Ensure the speculative_rollout module is importable
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TRAIN_FILE}" \
    data.train_batch_size=48 \
    data.max_prompt_length=512 \
    data.max_response_length=8320 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    \
    actor_rollout_ref.model.path=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-SpeculativeReasoner \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.02 \
    actor_rollout_ref.actor.ppo_mini_batch_size=48 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.04 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.max_model_len=8832 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8832 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=10 \
    +actor_rollout_ref.rollout.speculative_reasoning=True \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    algorithm.use_kl_in_reward=False \
    \
    custom_reward_function.path=verl_training/rewards.py \
    custom_reward_function.name=compute_score \
    \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_training_steps=400 \
    trainer.total_epochs=3 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=SplitReason_veRL \
    trainer.experiment_name=phase2_speculative_grpo \
    trainer.rollout_data_dir=outputs/phase2_generations
