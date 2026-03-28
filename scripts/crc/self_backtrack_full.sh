#!/bin/bash
#$ -N sb-full
#$ -A tma2
#$ -q gpu@@yye7_lab
#$ -l gpu_card=1
#$ -pe smp 8
#$ -l h_rt=48:00:00
#$ -M tma2@nd.edu
#$ -m abe
#$ -o logs/crc/self_backtrack_full_$JOB_ID.out
#$ -e logs/crc/self_backtrack_full_$JOB_ID.err
#$ -cwd

# Self-Backtracking full pipeline (Stage 1 + Stage 2) — Llama-3.2-1B on GSM8K
# Stage 1: Data build + dual-loss SFT
# Stage 2: 3 rounds of expert iteration (generate + filter correct + retrain)
# Final: Greedy eval on GSM8K
#
# Estimated time: ~30-35h on single A40
#   Stage 1: ~1.5h (data + SFT)
#   Stage 2: ~27h (3 rounds x ~9h each: 8h gen + 1h train)
#   Eval: ~30min per iteration
#
# Usage: qsub scripts/crc/self_backtrack_full.sh

set -euo pipefail

source scripts/crc/bashrc.sh

MODEL=meta-llama/Llama-3.2-1B
DATA_DIR=data/self_backtracking/gsm8k
STAGE1_DIR=outputs/self-backtrack-llama3.2-1b-gsm8k
STAGE2_DIR=outputs/self-backtrack-ei-llama3.2-1b-gsm8k

echo "=== Job started: $(date) ==="
echo "Host: $(hostname)"
nvidia-smi
echo ""

# ---------------------------------------------------------------
# Stage 1: Build data + dual-loss SFT
# ---------------------------------------------------------------
echo "=== Stage 1: Building training data ==="
python -m baselines.self_backtracking.build_data \
    --output_dir ${DATA_DIR} \
    --error_rate 0.5 \
    --seed 42
echo "Data construction done: $(date)"

echo "=== Stage 1: Training ==="
WANDB_MODE=online python -m baselines.self_backtracking.train \
    --model_name_or_path ${MODEL} \
    --data_dir ${DATA_DIR}/hf_dataset \
    --output_dir ${STAGE1_DIR} \
    --num_epochs 3 \
    --batch_size 16 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --max_length 2048 \
    --seed 42 \
    --bf16 \
    --wandb_project n-mars \
    --wandb_run_name self-backtrack-stage1-llama3.2-1b-gsm8k
echo "Stage 1 done: $(date)"
echo ""

# ---------------------------------------------------------------
# Stage 2: Expert Iteration (3 rounds)
# ---------------------------------------------------------------
echo "=== Stage 2: Expert Iteration (3 rounds) ==="
python -m baselines.self_backtracking.expert_iteration \
    --model_path ${STAGE1_DIR} \
    --base_model ${MODEL} \
    --output_dir ${STAGE2_DIR} \
    --num_iterations 3 \
    --b 1 --n 32 \
    --max_new_tokens 512 \
    --temperature 0.7 \
    --ei_epochs 3 \
    --ei_batch_size 16 \
    --ei_lr 1e-5 \
    --lora_rank 8 \
    --max_length 2048 \
    --bf16 \
    --seed 42
echo "Stage 2 done: $(date)"
echo ""

echo "=== Job completed: $(date) ==="
