#!/bin/bash
#SBATCH --job-name=nmars-sft
#SBATCH --account=bgdn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --mail-user=tma2@nd.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/%x_%j.out

# Usage: sbatch scripts/delta/sft_train.sh configs/train/sft_template.yaml
CONFIG=${1:-configs/train/sft_template.yaml}

source scripts/delta/bashrc.sh

torchrun \
  --nproc_per_node=4 \
  --master_port=29500 \
  $(which swift) sft \
  --config "$CONFIG"
