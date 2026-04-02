#!/bin/bash
#SBATCH --job-name=py
#SBATCH --output=py_output.out
#SBATCH --error=py_error.out
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=volta:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G

source /project/pakhrin/venv/bin/activate

module add python

# Run python script
/project/pakhrin/venv/bin/torchrun /project/pakhrin/simonsha/qwen3-vl-lora/qwen3-vl-lora.py --r 32 --batch_size 2 
