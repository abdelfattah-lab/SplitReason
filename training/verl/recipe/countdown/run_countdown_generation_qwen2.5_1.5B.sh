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

MODEL_FAMILY=Qwen
MODEL_NAME=Qwen2.5-1.5B
#MODEL_PATH=$MODEL_FAMILY/$MODEL_NAME #/mnt/home/tg436/countdown-expt/models/2025-07-27_Qwen/Qwen2.5-1.5B_1e-6_512_abstention_instruction/global_step_50/actor/huggingface #
MODEL_PATH=/mnt/home/tg436/countdown-expt/models/2025-07-31_Qwen2.5-1.5B_1e-6_512__4_5_basic100_abstention/global_step_150/actor/huggingface
DATA_PATH=/mnt/home/tg436/countdown-expt/data_countdown_scratch/hint__4_5
#OUTPUT_PATH_BASE=/mnt/home/tg436/countdown-expt/models
OUTPUT_PATH=$MODEL_PATH'/validation/'

export RAY_TMPDIR=/tmp/ray_$SLURM_JOBID
RAY_PORT=6375
ray start --head --port=$RAY_PORT --temp-dir=$RAY_TMPDIR
export RAY_ADDRESS=127.0.0.1:$RAY_PORT

HYDRA_FULL_ERROR=1  python3 -m verl.trainer.main_generation \
    data.path=$DATA_PATH/test.parquet \
    data.output_path=$OUTPUT_PATH \
    model.path=$MODEL_PATH \
    data.batch_size=4 \
    data.n_samples=4 \
    rollout.prompt_length=256 \
    rollout.response_length=1024 \
    trainer.n_gpus_per_node=1 allow_hint=True \
    trainer.nnodes=1 2>&1 | tee $OUTPUT_PATH'/output.log'
