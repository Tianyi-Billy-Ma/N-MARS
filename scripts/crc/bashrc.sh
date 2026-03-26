#!/bin/bash
# Shared environment setup for Notre Dame CRC (SGE)
# Source this at the top of every job script: source scripts/crc/bashrc.sh

# module load python/3.11

source ~/.bashrc

export HF_HOME="$CACHE_DIR/huggingface"
export WANDB_DIR="$CACHE_DIR/wandb"

export PYTHONDONTWRITEBYTECODE=1
export WANDB_PROJECT="n-mars"
export WANDB_ENTITY="mtybilly"

# Activate uv-managed virtual environment
cd $CACHE_DIR/N-MARS
source ./.venv/bin/activate
