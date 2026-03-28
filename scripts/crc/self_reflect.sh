#!/bin/bash
#$ -N self-reflect
#$ -A tma2
#$ -q gpu@@yye7_lab
#$ -l gpu_card=1
#$ -pe smp 8
#$ -l h_rt=24:00:00
#$ -M tma2@nd.edu
#$ -m abe
#$ -o logs/crc/self_reflect_$JOB_ID.out
#$ -e logs/crc/self_reflect_$JOB_ID.err
#$ -cwd

# Self-Reflect baseline — Llama-3.2-1B on GSM8K
# Two variants: SFT (all tokens supervised) and mSFT (error tokens masked)
# Pipeline: data build -> SFT train+eval -> mSFT train+eval
# Estimated time: ~4h on single A40
#
# Usage: qsub scripts/crc/self_reflect.sh

set -euo pipefail

source scripts/crc/bashrc.sh

MODEL=meta-llama/Llama-3.2-1B
DATA_DIR=data/self_reflect/gsm8k
OUTPUT_SFT=outputs/self-reflect-sft-llama3.2-1b-gsm8k
OUTPUT_MSFT=outputs/self-reflect-msft-llama3.2-1b-gsm8k

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
# Step 2a: Train Self-Reflect (SFT — no error masking)
# ---------------------------------------------------------------
echo "=== Step 2a: Training Self-Reflect (SFT) ==="
WANDB_MODE=online python -m baselines.self_reflect.train \
    --model_name_or_path ${MODEL} \
    --data_dir ${DATA_DIR}/hf_dataset \
    --output_dir ${OUTPUT_SFT} \
    --num_epochs 3 \
    --batch_size 16 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --max_length 2048 \
    --seed 42 \
    --bf16 \
    --wandb_project n-mars \
    --wandb_run_name self-reflect-sft-llama3.2-1b-gsm8k
echo "SFT training done: $(date)"
echo ""

# ---------------------------------------------------------------
# Step 2b: Train Self-Reflect (mSFT — mask error tokens)
# ---------------------------------------------------------------
echo "=== Step 2b: Training Self-Reflect (mSFT) ==="
WANDB_MODE=online python -m baselines.self_reflect.train \
    --model_name_or_path ${MODEL} \
    --data_dir ${DATA_DIR}/hf_dataset \
    --output_dir ${OUTPUT_MSFT} \
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
echo "mSFT training done: $(date)"
echo ""

# ---------------------------------------------------------------
# Step 3a: Evaluate Self-Reflect (SFT)
# ---------------------------------------------------------------
echo "=== Step 3a: Eval Self-Reflect (SFT) ==="
python -m baselines.self_reflect.evaluate \
    --model_path ${OUTPUT_SFT} \
    --output_path ${OUTPUT_SFT}/eval_greedy.json \
    --task gsm8k \
    --max_new_tokens 512 \
    --seed 42
echo "SFT eval done: $(date)"
echo ""

# ---------------------------------------------------------------
# Step 3b: Evaluate Self-Reflect (mSFT)
# ---------------------------------------------------------------
echo "=== Step 3b: Eval Self-Reflect (mSFT) ==="
python -m baselines.self_reflect.evaluate \
    --model_path ${OUTPUT_MSFT} \
    --output_path ${OUTPUT_MSFT}/eval_greedy.json \
    --task gsm8k \
    --max_new_tokens 512 \
    --seed 42
echo "mSFT eval done: $(date)"
echo ""

echo "=== Job completed: $(date) ==="
