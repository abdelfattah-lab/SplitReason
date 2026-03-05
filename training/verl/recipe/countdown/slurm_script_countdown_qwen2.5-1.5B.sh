#!/bin/bash
#SBATCH --job-name=count_h         # Name of the job
#SBATCH --output=logs/output_%j.log    # Stdout (%j = job ID)
#SBATCH --error=logs/error_%j.log      # Stderr
#SBATCH --time=01:00:00                # Time limit: HH:MM:SS
#SBATCH --partition=standard           # Partition to use (change if needed)
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=4              # CPUs per task
#SBATCH --mem=8G                       # Memory per node
#SBATCH --gres=gpu:1                   # Request 1 GPU (optional)

# Optional: load modules
# module load python/3.10 cuda/11.8

# Optional: activate Conda or venv
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

# Run your script
python3 my_script.py --arg1 val1
