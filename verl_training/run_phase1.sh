#!/usr/bin/env bash
# Phase 1: Vanilla GRPO on veRL (accuracy-only reward, no speculative reasoning)
#
# GPU allocation: GPU 0-1 reserved for big model (not used in Phase 1),
#                 GPU 2-7 for 1.5B veRL colocated training+rollout.
#
# Prerequisites:
#   1. pip install verl
#   2. python verl_training/prepare_data.py --phase 1 --output_dir ~/data/splitreason
#   3. python verl_training/prepare_tokenizer.py --output_dir ~/data/splitreason/tokenizer
#
# Usage:
#   bash verl_training/run_phase1.sh

set -euo pipefail

# NCCL settings for multi-GPU stability
export TORCH_NCCL_ENABLE_MONITORING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=120000
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=216000
export NCCL_P2P_DISABLE=1

# WandB config
export WANDB_API_KEY=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
export WANDB_ENTITY=anthonyf1223-cornell-university

# GPU selection: 4 GPUs for generation parallelism (each runs its own vLLM server)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Explicitly unset TRITON_INTERPRET — it was previously set to fix triton
# "0 active drivers" in Ray non-GPU actors, but deepspeed>=0.16.4 resolves that.
# TRITON_INTERPRET=1 breaks both vLLM CUDA graphs and flash_attn triton kernels.
unset TRITON_INTERPRET
export RAY_DEDUP_LOGS=0

DATA_DIR="${HOME}/data/splitreason"
TRAIN_FILE="${DATA_DIR}/train_phase1.parquet"

if [ ! -f "${TRAIN_FILE}" ]; then
    echo "ERROR: ${TRAIN_FILE} not found. Run prepare_data.py first."
    exit 1
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TRAIN_FILE}" \
    data.train_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    \
    actor_rollout_ref.model.path=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-SpeculativeReasoner \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.02 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
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
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=10 \
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
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_training_steps=400 \
    trainer.total_epochs=3 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=SplitReason_veRL \
    trainer.experiment_name=phase1_vanilla_grpo \
    trainer.rollout_data_dir=outputs/phase1_generations
