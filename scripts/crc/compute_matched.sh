#!/bin/bash
#$ -N compute-matched
#$ -A tma2
#$ -q gpu@@yye7_lab
#$ -l gpu_card=1
#$ -pe smp 8
#$ -l h_rt=4:00:00
#$ -M tma2@nd.edu
#$ -m abe
#$ -o logs/crc/compute_matched_$JOB_ID.out
#$ -e logs/crc/compute_matched_$JOB_ID.err
#$ -cwd

# EXP-2: Compute-Matched Comparison — SFT self-consistency at N-MARS token budget
#
# Usage:
#   qsub scripts/crc/compute_matched.sh <model_path> <nmars_results> <output_path> [k_values] [task]
#
# Examples:
#   qsub scripts/crc/compute_matched.sh \
#       outputs/sft-llama3.1-8b-gsm8k \
#       outputs/nmars-inference-cost.json \
#       outputs/compute-matched-results.json
#
#   qsub scripts/crc/compute_matched.sh \
#       outputs/sft-llama3.1-8b-gsm8k \
#       outputs/nmars-inference-cost.json \
#       outputs/compute-matched-results.json \
#       1,2,4,8,16 \
#       gsm8k

MODEL_PATH=${1:?MODEL_PATH required}
NMARS_RESULTS=${2:?NMARS_RESULTS required}
OUTPUT_PATH=${3:?OUTPUT_PATH required}
K_VALUES=${4:-1,2,4,8}
TASK=${5:-gsm8k}

set -euo pipefail

source scripts/crc/bashrc.sh

echo "=== Job started: $(date) ==="
echo "Host: $(hostname)"
nvidia-smi
echo ""
echo "Model:         ${MODEL_PATH}"
echo "N-MARS results: ${NMARS_RESULTS}"
echo "Output:        ${OUTPUT_PATH}"
echo "K values:      ${K_VALUES}"
echo "Task:          ${TASK}"
echo ""

python -m n_mars.scripts.compute_matched \
    --model_path "${MODEL_PATH}" \
    --nmars_results "${NMARS_RESULTS}" \
    --output_path "${OUTPUT_PATH}" \
    --k_values "${K_VALUES}" \
    --task "${TASK}" \
    --max_new_tokens 512 \
    --temperature 0.7 \
    --seed 42

echo ""
echo "=== Job completed: $(date) ==="
