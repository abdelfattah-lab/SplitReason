#!/bin/bash
#SBATCH --job-name=count_h         # Name of the job
#SBATCH --output=logs/output_%j.log    # Stdout (%j = job ID)
#SBATCH --error=logs/error_%j.log      # Stderr
#SBATCH --time=24:00:00                # Time limit: HH:MM:SS
#SBATCH --mail-user=tg436@cornell.edu    # Your email address
#SBATCH --mail-type=END,FAIL                 # Send email on job END or FAIL
#SBATCH --partition=priority,cornell           # Partition to use (change if needed)
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=4              # CPUs per task
#SBATCH --gres=gpu:1                   # Request 1 GPU (optional)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate verlEnv

N_GPUS=1

VLLM_ATTENTION_BACKEND=XFORMERS

MODEL_PATH=Qwen/Qwen2.5-3B
DATA_PATH=/mnt/home/tg436/countdown-expt/data/
TRAIN_BATCH_SIZE=512
LR=1e-6
EXPERIMENT_NAME=$(date +%F)'_'$MODEL_PATH'_'$LR'_'$TRAIN_BATCH_SIZE
OUTPUT_PATH_BASE=/mnt/lustre/cornell/tg436/countdown-expt/models/
OUTPUT_PATH=$OUTPUT_PATH_BASE'/'$EXPERIMENT_NAME

ray start --head --port=6379
export RAY_ADDRESS="auto"

conda activate verlEnv

HYDRA_FULL_ERROR=1  python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_PATH/train.parquet \
    data.val_files=$DATA_PATH/test.parquet \
    custom_reward_function.path=recipe/countdown/reward_function.py \
    custom_reward_function.name=compute_score \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=1312 \
    data.max_prompt_length=256 \
    data.max_response_length=1024 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['wandb'] \
    trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$OUTPUT_PATH \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=200 \
    trainer.log_val_generations=100 \
    trainer.rollout_data_dir=$OUTPUT_PATH \
    trainer.project_name=countdown \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_training_steps=200 2>&1 | tee $OUTPUT_PATH_BASE'/output.log'
