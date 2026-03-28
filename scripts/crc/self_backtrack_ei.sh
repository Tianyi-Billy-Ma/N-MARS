#!/bin/bash
#$ -N sb-ei
#$ -A tma2
#$ -q gpu@@yye7_lab
#$ -l gpu_card=1
#$ -pe smp 8
#$ -l h_rt=48:00:00
#$ -M tma2@nd.edu
#$ -m abe
#$ -o logs/crc/self_backtrack_ei_$JOB_ID.out
#$ -e logs/crc/self_backtrack_ei_$JOB_ID.err
#$ -cwd

# Self-Backtracking Stage 2 only: Expert Iteration (3 rounds)
# Uses already-trained Stage 1 model.
# Estimated time: ~30h on single A40 (3 rounds x ~10h each)
#
# Usage: qsub scripts/crc/self_backtrack_ei.sh

set -euo pipefail

source scripts/crc/bashrc.sh

MODEL=meta-llama/Llama-3.2-1B
STAGE1_DIR=outputs/self-backtrack-llama3.2-1b-gsm8k
STAGE2_DIR=outputs/self-backtrack-ei-llama3.2-1b-gsm8k

echo "=== Job started: $(date) ==="
echo "Host: $(hostname)"
nvidia-smi
echo ""

# Verify Stage 1 model exists
if [ ! -d "${STAGE1_DIR}" ]; then
    echo "ERROR: Stage 1 model not found at ${STAGE1_DIR}"
    exit 1
fi

echo "=== Expert Iteration (3 rounds) ==="
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
echo "Expert iteration done: $(date)"
echo ""

echo "=== Job completed: $(date) ==="
