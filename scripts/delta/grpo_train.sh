#!/bin/bash
#SBATCH --job-name=nmars-grpo
#SBATCH --account=tma3
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --mail-user=tma2@nd.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/delta/grpo_%j.out
#SBATCH --error=logs/delta/grpo_%j.err

# Usage: sbatch scripts/delta/grpo_train.sh configs/train/grpo_template.yaml
CONFIG=${1:-configs/train/grpo_template.yaml}

source scripts/delta/bashrc.sh

torchrun \
  --nproc_per_node=4 \
  --master_port=29500 \
  $(which swift) rlhf \
  --config "$CONFIG"
