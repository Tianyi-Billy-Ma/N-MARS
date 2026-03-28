#!/bin/bash
#$ -N grad-cosine
#$ -A tma2
#$ -q gpu@@yye7_lab
#$ -l gpu_card=1
#$ -pe smp 8
#$ -l h_rt=04:00:00
#$ -M tma2@nd.edu
#$ -m abe
#$ -o logs/crc/grad_cosine_$JOB_ID.out
#$ -e logs/crc/grad_cosine_$JOB_ID.err
#$ -cwd

# Gradient cosine similarity analysis for Proposition 2.1 validation.
# Full parameter training, Llama-3.2-1B, 3 epochs, bs=8 on single A40.
# Estimated time: ~25 min.
#
# Usage: qsub scripts/crc/gradient_cosine.sh

set -euo pipefail

source scripts/crc/bashrc.sh

echo "=== Job started: $(date) ==="
echo "Host: $(hostname)"
nvidia-smi
echo ""

python -m n_mars.scripts.gradient_cosine \
    --model_name_or_path meta-llama/Llama-3.2-1B \
    --dataset_config p0.1_n10 \
    --output_dir outputs/gradient_cosine \
    --num_steps 2522 \
    --log_interval 50 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --max_length 1024 \
    --lora_rank 0 \
    --seed 42

echo "=== Job completed: $(date) ==="
