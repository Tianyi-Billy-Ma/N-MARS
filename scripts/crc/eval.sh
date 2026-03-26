#!/bin/bash
#$ -N nmars-eval
#$ -A tma2
#$ -q gpu
#$ -l gpu=4
#$ -l h_rt=4:00:00
#$ -l h_vmem=64G
#$ -M tma2@nd.edu
#$ -m abe
#$ -o logs/crc/eval_$JOB_ID.out
#$ -e logs/crc/eval_$JOB_ID.err
#$ -cwd

# Usage: qsub scripts/crc/eval.sh <model_path> <task_name> <run_suffix>
# Example: qsub scripts/crc/eval.sh outputs/llama3.1-8b-sft-20240326 gsm8k llama3.1-8b-sft
MODEL_PATH=${1:?MODEL_PATH required}
TASK_NAME=${2:?TASK_NAME required}
RUN_SUFFIX=${3:?RUN_SUFFIX required}   # used in output path and wandb run name

source scripts/crc/bashrc.sh

OUTPUT_PATH=outputs/${RUN_SUFFIX}

lm_eval \
  --model hf \
  --model_args pretrained=${MODEL_PATH},parallelize=True \
  --tasks ${TASK_NAME} \
  --batch_size 128 \
  --output_path ${OUTPUT_PATH} \
  --wandb_args project=${WANDB_PROJECT},name=${TASK_NAME}-${RUN_SUFFIX},group=lm_eval \
  --seed 42 \
  --log_samples \
  --apply_chat_template
