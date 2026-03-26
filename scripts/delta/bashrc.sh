#!/bin/bash
# Shared environment setup for UIUC Delta (SLURM)
# Source this at the top of every job script: source scripts/delta/bashrc.sh

module reset
module load python/3.11

# Activate uv-managed virtual environment
source "$(dirname "$BASH_SOURCE")"/../../.venv/bin/activate

# W&B
export WANDB_PROJECT=n-mars

# Multi-GPU comms
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=4
