#!/bin/bash
#$ -N self-backtrack
#$ -A tma2
#$ -q gpu@@yye7_lab
#$ -l gpu_card=1
#$ -pe smp 8
#$ -l h_rt=24:00:00
#$ -M tma2@nd.edu
#$ -m abe
#$ -o logs/crc/self_backtrack_$JOB_ID.out
#$ -e logs/crc/self_backtrack_$JOB_ID.err
#$ -cwd

# Self-Backtracking baseline (arXiv:2502.04404) — Llama-3.2-1B on GSM8K
# Full pipeline: data construction -> training (3 epochs) -> greedy eval -> backtracking eval
#
# Usage: qsub scripts/crc/self_backtrack.sh
# Estimated time: ~10h on single A40 (1.1h train + 15min greedy + 8h backtrack eval)

set -euo pipefail

source scripts/crc/bashrc.sh

MODEL=meta-llama/Llama-3.2-1B
DATA_DIR=data/self_backtracking/gsm8k
OUTPUT_DIR=outputs/self-backtrack-llama3.2-1b-gsm8k

echo "=== Job started: $(date) ==="
echo "Host: $(hostname)"
nvidia-smi
echo ""

# ---------------------------------------------------------------
# Step 1: Build training data (D_op + D_back)
# ---------------------------------------------------------------
echo "=== Step 1: Building training data ==="
python -m baselines.self_backtracking.build_data \
    --output_dir ${DATA_DIR} \
    --error_rate 0.5 \
    --seed 42
echo "Data construction done: $(date)"
echo ""

# ---------------------------------------------------------------
# Step 2: Train (dual-loss SFT, 3 epochs)
# ---------------------------------------------------------------
echo "=== Step 2: Training ==="
WANDB_MODE=online python -m baselines.self_backtracking.train \
    --model_name_or_path ${MODEL} \
    --data_dir ${DATA_DIR}/hf_dataset \
    --output_dir ${OUTPUT_DIR} \
    --num_epochs 3 \
    --batch_size 16 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --max_length 2048 \
    --seed 42 \
    --bf16 \
    --wandb_project n-mars \
    --wandb_run_name self-backtrack-llama3.2-1b-gsm8k
echo "Training done: $(date)"
echo ""

# ---------------------------------------------------------------
# Step 3a: Greedy evaluation
# ---------------------------------------------------------------
echo "=== Step 3a: Greedy evaluation ==="
python -m baselines.self_backtracking.evaluate \
    --model_path ${OUTPUT_DIR} \
    --output_path ${OUTPUT_DIR}/eval_greedy.json \
    --greedy_only \
    --max_new_tokens 512 \
    --seed 42
echo "Greedy eval done: $(date)"
echo ""

# ---------------------------------------------------------------
# Step 3b: Backtracking evaluation (b=1, n=32)
# ---------------------------------------------------------------
echo "=== Step 3b: Backtracking evaluation (b=1, n=32) ==="
python -m baselines.self_backtracking.evaluate \
    --model_path ${OUTPUT_DIR} \
    --output_path ${OUTPUT_DIR}/eval_backtrack_b1_n32.json \
    --b 1 --n 32 \
    --max_new_tokens 512 \
    --seed 42
echo "Backtracking eval done: $(date)"
echo ""

echo "=== Job completed: $(date) ==="
