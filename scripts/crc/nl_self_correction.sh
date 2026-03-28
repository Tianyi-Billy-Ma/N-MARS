#!/bin/bash
#$ -N nl-self-correction
#$ -A tma2
#$ -q gpu@@yye7_lab
#$ -l gpu_card=1
#$ -pe smp 8
#$ -l h_rt=12:00:00
#$ -M tma2@nd.edu
#$ -m abe
#$ -o logs/crc/nl_self_correction_$JOB_ID.out
#$ -e logs/crc/nl_self_correction_$JOB_ID.err
#$ -cwd

# NL Self-Correction Baseline (EXP-5) — Llama-3.2-1B on GSM8K
# Tests whether natural-language revision markers can replace the explicit
# UNDO/BACKTRACK token used in N-MARS (reviewer DLhi-W1 rebuttal baseline).
#
# Full pipeline: build NL-augmented data -> train NL-SFT -> train NL-mSFT
#                -> evaluate both variants on GSM8K test set
#
# Usage: qsub scripts/crc/nl_self_correction.sh
# Estimated time: ~10h on single A40

set -euo pipefail

source scripts/crc/bashrc.sh

DATA_DIR=data/nl_self_correction
SFT_OUTPUT=outputs/nl-sft-llama3.2-1b-gsm8k
MSFT_OUTPUT=outputs/nl-msft-llama3.2-1b-gsm8k

echo "=== Job started: $(date) ==="
echo "Host: $(hostname)"
nvidia-smi
echo ""

# ---------------------------------------------------------------
# Step 1: Build NL-augmented training data
# ---------------------------------------------------------------
echo "=== Step 1: Building NL-augmented training data ==="
python -m n_mars.scripts.nl_self_correction \
    --stage build_data \
    --output_dir ${DATA_DIR}
echo "Data construction done: $(date)"
echo ""

# ---------------------------------------------------------------
# Step 2a: Train NL-SFT (loss on all response tokens)
# ---------------------------------------------------------------
echo "=== Step 2a: Training NL-SFT ==="
WANDB_MODE=online python -m n_mars.scripts.nl_self_correction \
    --stage train \
    --variant sft \
    --data_dir ${DATA_DIR} \
    --output_dir ${SFT_OUTPUT} \
    --num_epochs 3 \
    --batch_size 16 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --max_length 2048 \
    --seed 42 \
    --bf16 \
    --wandb_project n-mars \
    --wandb_run_name nl-sft-llama3.2-1b-gsm8k
echo "NL-SFT training done: $(date)"
echo ""

# ---------------------------------------------------------------
# Step 2b: Train NL-mSFT (error tokens masked to -100)
# ---------------------------------------------------------------
echo "=== Step 2b: Training NL-mSFT ==="
WANDB_MODE=online python -m n_mars.scripts.nl_self_correction \
    --stage train \
    --variant msft \
    --data_dir ${DATA_DIR} \
    --output_dir ${MSFT_OUTPUT} \
    --num_epochs 3 \
    --batch_size 16 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --max_length 2048 \
    --seed 42 \
    --bf16 \
    --wandb_project n-mars \
    --wandb_run_name nl-msft-llama3.2-1b-gsm8k
echo "NL-mSFT training done: $(date)"
echo ""

# ---------------------------------------------------------------
# Step 3a: Evaluate NL-SFT on GSM8K
# ---------------------------------------------------------------
echo "=== Step 3a: Evaluating NL-SFT ==="
python -m n_mars.scripts.nl_self_correction \
    --stage evaluate \
    --model_path ${SFT_OUTPUT} \
    --output_path ${SFT_OUTPUT}/eval.json \
    --max_new_tokens 512 \
    --seed 42
echo "NL-SFT eval done: $(date)"
echo ""

# ---------------------------------------------------------------
# Step 3b: Evaluate NL-mSFT on GSM8K
# ---------------------------------------------------------------
echo "=== Step 3b: Evaluating NL-mSFT ==="
python -m n_mars.scripts.nl_self_correction \
    --stage evaluate \
    --model_path ${MSFT_OUTPUT} \
    --output_path ${MSFT_OUTPUT}/eval.json \
    --max_new_tokens 512 \
    --seed 42
echo "NL-mSFT eval done: $(date)"
echo ""

echo "=== Job completed: $(date) ==="
