#!/bin/bash
#$ -N inference-cost
#$ -A tma2
#$ -q gpu@@yye7_lab
#$ -l gpu_card=1
#$ -pe smp 8
#$ -l h_rt=4:00:00
#$ -M tma2@nd.edu
#$ -m abe
#$ -o logs/crc/inference_cost_$JOB_ID.out
#$ -e logs/crc/inference_cost_$JOB_ID.err
#$ -cwd

# EXP-1: Inference Cost Analysis — N-MARS UNDO token overhead vs SFT baseline
#
# Usage:
#   qsub scripts/crc/inference_cost.sh <model_path> <output_path> [baseline_model_path] [task]
#
# Examples:
#   qsub scripts/crc/inference_cost.sh \
#       outputs/nmars-llama3.1-8b-gsm8k \
#       outputs/nmars-inference-cost.json
#
#   qsub scripts/crc/inference_cost.sh \
#       outputs/nmars-llama3.1-8b-gsm8k \
#       outputs/nmars-inference-cost.json \
#       outputs/sft-llama3.1-8b-gsm8k \
#       gsm8k

MODEL_PATH=${1:?MODEL_PATH required}
OUTPUT_PATH=${2:?OUTPUT_PATH required}
BASELINE_MODEL_PATH=${3:-}
TASK=${4:-gsm8k}

set -euo pipefail

source scripts/crc/bashrc.sh

echo "=== Job started: $(date) ==="
echo "Host: $(hostname)"
nvidia-smi
echo ""
echo "Model:    ${MODEL_PATH}"
echo "Output:   ${OUTPUT_PATH}"
echo "Task:     ${TASK}"
if [ -n "${BASELINE_MODEL_PATH}" ]; then
    echo "Baseline: ${BASELINE_MODEL_PATH}"
fi
echo ""

BASELINE_FLAG=""
if [ -n "${BASELINE_MODEL_PATH}" ]; then
    BASELINE_FLAG="--baseline_model_path ${BASELINE_MODEL_PATH}"
fi

python -m n_mars.scripts.inference_cost \
    --model_path "${MODEL_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --task "${TASK}" \
    --max_new_tokens 512 \
    --seed 42 \
    ${BASELINE_FLAG}

echo ""
echo "=== Job completed: $(date) ==="
