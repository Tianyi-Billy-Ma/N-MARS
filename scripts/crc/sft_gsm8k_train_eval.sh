#!/bin/bash
#$ -M tma2@nd.edu
#$ -m abe
#$ -pe smp 32
#$ -q gpu@@yye7_lab
#$ -o logs/$JOB_NAME_$JOB_ID.out
#$ -j y
#$ -V
#$ -l gpu_card=4
#$ -N sft-gsm8k
#$ -cwd

source scripts/crc/bashrc.sh

MODEL_DIR=outputs/sft-llama3.2-1b-gsm8k
CONFIG=configs/train/sft_gsm8k_llama3.2-1b.yaml

# ── Stage 1: SFT Training ─────────────────────────────────────────────────
echo "=== Stage 1: SFT Training ==="
torchrun \
  --nproc_per_node=4 \
  --master_port=29500 \
  $(which swift) sft \
  --config "$CONFIG"

echo "=== SFT Training Complete ==="

# ── Stage 2: Eval on GSM8K ────────────────────────────────────────────────
echo "=== Stage 2: Eval on GSM8K ==="
accelerate launch -m lm_eval \
  --model hf \
  --model_args pretrained=${MODEL_DIR} \
  --tasks gsm8k \
  --batch_size auto:32 \
  --output_path ${MODEL_DIR}/eval_gsm8k \
  --wandb_args project=${WANDB_PROJECT},name=gsm8k-sft-llama3.2-1b,group=lm_eval \
  --seed 42 \
  --log_samples \
  --confirm_run_unsafe_code

echo "=== Eval Complete ==="
