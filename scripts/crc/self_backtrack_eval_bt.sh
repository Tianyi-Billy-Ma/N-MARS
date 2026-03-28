#!/bin/bash
#$ -N sb-bt-eval
#$ -A tma2
#$ -q gpu@@yye7_lab
#$ -l gpu_card=1
#$ -pe smp 8
#$ -l h_rt=24:00:00
#$ -M tma2@nd.edu
#$ -m abe
#$ -o logs/crc/self_backtrack_eval_bt_$JOB_ID.out
#$ -e logs/crc/self_backtrack_eval_bt_$JOB_ID.err
#$ -cwd

# Self-Backtracking backtracking eval (b=1, n=32) on GSM8K
# Uses already-trained model at outputs/self-backtrack-llama3.2-1b-gsm8k/
# Estimated time: ~8h on single A40
#
# Usage: qsub scripts/crc/self_backtrack_eval_bt.sh

set -euo pipefail

source scripts/crc/bashrc.sh

OUTPUT_DIR=outputs/self-backtrack-llama3.2-1b-gsm8k

echo "=== Job started: $(date) ==="
echo "Host: $(hostname)"
nvidia-smi
echo ""

# Verify model exists
if [ ! -d "${OUTPUT_DIR}" ]; then
    echo "ERROR: Model not found at ${OUTPUT_DIR}"
    exit 1
fi

echo "=== Backtracking evaluation (b=1, n=32) ==="
python -m baselines.self_backtracking.evaluate \
    --model_path ${OUTPUT_DIR} \
    --output_path ${OUTPUT_DIR}/eval_backtrack_b1_n32.json \
    --b 1 --n 32 \
    --max_new_tokens 512 \
    --seed 42
echo "Backtracking eval done: $(date)"
echo ""

echo "=== Job completed: $(date) ==="
