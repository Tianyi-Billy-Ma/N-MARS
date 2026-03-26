#!/bin/bash
# Shared environment setup for UIUC Delta (SLURM)
# Source this at the top of every job script: source scripts/delta/bashrc.sh

# module load python/3.11
module load aws-ofi-nccl/1.14.2

# Activate uv-managed virtual environment
source ./.venv/bin/activate

# Multi-GPU comms
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=4

export USERNAME="tma3"
export PROJECT_CODE="bgdn"
export HDD_DIR="/work/hdd/${PROJECT_CODE}/${USERNAME}"
export NVME_DIR="/work/nvme/${PROJECT_CODE}/${USERNAME}"
export HF_HOME="/projects/${PROJECT_CODE}/${USERNAME}/.cache/huggingface"
export WANDB_DIR="/projects/${PROJECT_CODE}/${USERNAME}/.cache/wandb"

# Ensure cache directories exist
mkdir -p "${HF_HOME}" "${WANDB_DIR}"

# source ./.venv/bin/activate

export PYTHONDONTWRITEBYTECODE=1
# W&B
export WANDB_PROJECT=n-mars
export WANDB_ENTITY="mtybilly"
