#!/bin/bash
#$ -N sr-msft
#$ -A tma2
#$ -q gpu@@yye7_lab
#$ -l gpu_card=1
#$ -pe smp 8
#$ -l h_rt=12:00:00
#$ -M tma2@nd.edu
#$ -m abe
#$ -o logs/crc/self_reflect_msft_$JOB_ID.out
#$ -e logs/crc/self_reflect_msft_$JOB_ID.err
#$ -cwd

# Self-Reflect (mSFT) — NL correction, error tokens masked
# Llama-3.2-1B on GSM8K. Estimated time: ~2h on single A40
#
# Usage: qsub scripts/crc/self_reflect_msft.sh

set -euo pipefail

source scripts/crc/bashrc.sh

MODEL=meta-llama/Llama-3.2-1B
DATA_DIR=data/self_reflect/gsm8k
OUTPUT_DIR=outputs/self-reflect-msft-llama3.2-1b-gsm8k

echo "=== Job started: $(date) ==="
echo "Host: $(hostname)"
nvidia-smi
echo ""

# ---------------------------------------------------------------
# Step 1: Build training data (replace BACKTRACK with NL markers)
# ---------------------------------------------------------------
echo "=== Step 1: Building training data ==="
python -m baselines.self_reflect.build_data \
    --output_dir ${DATA_DIR} \
    --dataset_config p0.1_n10 \
    --seed 42
echo "Data construction done: $(date)"
echo ""

# ---------------------------------------------------------------
# Step 2: Train (mSFT — mask error tokens before NL markers)
# ---------------------------------------------------------------
echo "=== Step 2: Training Self-Reflect (mSFT) ==="
WANDB_MODE=online python -m baselines.self_reflect.train \
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
    --mask_errors \
    --wandb_project n-mars \
    --wandb_run_name self-reflect-msft-llama3.2-1b-gsm8k
echo "Training done: $(date)"
echo ""

# ---------------------------------------------------------------
# Step 3: Greedy evaluation on GSM8K
# ---------------------------------------------------------------
echo "=== Step 3: Evaluation ==="
python -m baselines.self_reflect.evaluate \
    --model_path ${OUTPUT_DIR} \
    --output_path ${OUTPUT_DIR}/eval_greedy.json \
    --task gsm8k \
    --max_new_tokens 512 \
    --seed 42
echo "Eval done: $(date)"
echo ""

echo "=== Job completed: $(date) ==="
