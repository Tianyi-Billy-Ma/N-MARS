#!/bin/bash
#$ -N nmars-sft
#$ -A tma2
#$ -q gpu
#$ -l gpu=4
#$ -l h_rt=12:00:00
#$ -pe mpi-8 4
#$ -l h_vmem=32G
#$ -M tma2@nd.edu
#$ -m abe
#$ -o logs/crc/sft_$JOB_ID.out
#$ -e logs/crc/sft_$JOB_ID.err
#$ -cwd

# Usage: qsub scripts/crc/sft_train.sh configs/train/sft_template.yaml
CONFIG=${1:-configs/train/sft_template.yaml}

source scripts/crc/bashrc.sh

torchrun \
  --nproc_per_node=4 \
  --master_port=29500 \
  $(which swift) sft \
  --config "$CONFIG"
