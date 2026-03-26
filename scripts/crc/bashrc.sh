#!/bin/bash
# Shared environment setup for Notre Dame CRC (SGE)
# Source this at the top of every job script: source scripts/crc/bashrc.sh

# module load python/3.11

export USERNAME="tma3"
export PROJECT_CODE="bgdn"
export HDD_DIR="/work/hdd/${PROJECT_CODE}/${USERNAME}"
export NVME_DIR="/work/nvme/${PROJECT_CODE}/${USERNAME}"
export HF_HOME="/work/hdd/${PROJECT_CODE}/${USERNAME}/huggingface"
export WANDB_DIR="/work/hdd/${PROJECT_CODE}/${USERNAME}/wandb"

# source ./.venv/bin/activate

export PYTHONDONTWRITEBYTECODE=1
export WANDB_PROJECT="n-mars"
export WANDB_ENTITY="mtybilly"

# Activate uv-managed virtual environment
source ./venv/bin/activate

# Multi-GPU comms
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=4
