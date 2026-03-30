#!/bin/bash
#$ -M tma2@nd.edu
#$ -m abe
#$ -pe smp 32
#$ -q gpu@@yye7_lab
#$ -o logs/$JOB_NAME_$JOB_ID.out
#$ -j y
#$ -V
#$ -l gpu_card=4
#$ -N sft-full
#$ -cwd

# Usage: qsub scripts/crc/full.sh <config_path> [extra args...]
# Example: qsub scripts/crc/full.sh configs/train/sft_gsm8k_llama3.2-1b.yaml
# Example: qsub scripts/crc/full.sh configs/train/sft_gsm8k_llama3.2-1b.yaml --mode train
# Example: qsub scripts/crc/full.sh configs/train/sft_gsm8k_llama3.2-1b.yaml --model_name_or_path meta-llama/Llama-3.1-8B --output_dir outputs/sft-llama3.1-8b-gsm8k

CONFIG=${1:?CONFIG required}
shift  # remaining args passed through

source scripts/crc/bashrc.sh

torchrun \
  --nproc_per_node=4 \
  --master_port=29500 \
  -m n_mars.train.sft \
  --config "$CONFIG" \
  "$@"
