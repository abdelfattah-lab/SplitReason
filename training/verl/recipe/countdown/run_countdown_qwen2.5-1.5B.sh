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


RAY_TMPDIR=/tmp/ray_$SLURM_JOBID
RAY_PORT=6384
ray start --head --port=$RAY_PORT --temp-dir=$RAY_TMPDIR
export RAY_ADDRESS=127.0.0.1:$RAY_PORT


MODEL_FAMILY=Qwen #deepseek-ai
MODEL_NAME=Qwen2.5-3B #DeepSeek-R1-Distill-Qwen-1.5B
MODEL_PATH=$MODEL_FAMILY/$MODEL_NAME
#MODEL_PATH=/mnt/home/tg436/countdown-expt/models/2025-08-18_Qwen2.5-3B-Instruct_1e-6_512_hintFalse_eTrue_eA0.9_ek6hint__4_5/global_step_50/actor/huggingface
DATA_PATH=/mnt/home/tg436/countdown-expt/data_countdown_scratch_take2/hint__4_5/
TRAIN_BATCH_SIZE=512
LR=1e-6
allow_hint=True
use_epsilon=True
epsilon_A=0.6
epsilon_k=12
EXPERIMENT_NAME=$(date +%F)'_'$MODEL_NAME'_'$LR'_'$TRAIN_BATCH_SIZE'_hint'$allow_hint'_e'$use_epsilon'_eA'$epsilon_A'_ek'$epsilon_k'hint__4_5_treesearch_penalty_micro32'
#EXPERIMENT_NAME=$(date +%F)'_'$MODEL_NAME'_'$LR'_'$TRAIN_BATCH_SIZE'_hint'$allow_hint'_e'$use_epsilon'bell_hint__4_5'
OUTPUT_PATH_BASE=/mnt/home/tg436/countdown-expt/models
OUTPUT_PATH=$OUTPUT_PATH_BASE'/'$EXPERIMENT_NAME

mkdir $OUTPUT_PATH
HYDRA_FULL_ERROR=1  python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_PATH/train.parquet \
    data.val_files=$DATA_PATH/test.parquet \
    custom_reward_function.path=/mnt/home/tg436/countdown-expt/verl/recipe/countdown/reward_function.py \
    custom_reward_function.name=compute_score_hint \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=1312 \
    data.max_prompt_length=256 \
    data.max_response_length=1024 \
    actor_rollout_ref.rollout.n=4 \
    allow_hint=$allow_hint \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.ppo_micro_batch_size=32 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.use_epsilon=$use_epsilon \
    trainer.epsilon_k=$epsilon_k \
    trainer.epsilon_A=$epsilon_A \
    trainer.logger=['wandb'] \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$OUTPUT_PATH \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.resume_mode=disable \
    trainer.log_val_generations=25 \
    trainer.rollout_data_dir=$OUTPUT_PATH \
    trainer.project_name=countdown-newdata \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_training_steps=150 2>&1 | tee $OUTPUT_PATH'/output.log'