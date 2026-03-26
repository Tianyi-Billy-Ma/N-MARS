#!/bin/bash
#$ -N nmars-grpo
#$ -A tma2
#$ -q gpu
#$ -l gpu=4
#$ -l h_rt=24:00:00
#$ -pe mpi-8 4
#$ -l h_vmem=32G
#$ -M tma2@nd.edu
#$ -m abe
#$ -o logs/crc/grpo_$JOB_ID.out
#$ -e logs/crc/grpo_$JOB_ID.err
#$ -cwd

# Usage: qsub scripts/crc/grpo_train.sh configs/train/grpo_template.yaml
CONFIG=${1:-configs/train/grpo_template.yaml}

source scripts/crc/bashrc.sh

torchrun \
  --nproc_per_node=4 \
  --master_port=29500 \
  $(which swift) rlhf \
  --config "$CONFIG"
