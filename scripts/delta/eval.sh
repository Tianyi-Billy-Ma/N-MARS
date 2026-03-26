#!/bin/bash
#SBATCH --job-name=nmars-eval
#SBATCH --account=tma3
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --mail-user=tma2@nd.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/delta/eval_%j.out
#SBATCH --error=logs/delta/eval_%j.err

# Usage: sbatch scripts/delta/eval.sh <model_path> <task_name> <run_suffix>
# Example: sbatch scripts/delta/eval.sh outputs/llama3.1-8b-sft-20240326 gsm8k llama3.1-8b-sft
MODEL_PATH=${1:?MODEL_PATH required}
TASK_NAME=${2:?TASK_NAME required}
RUN_SUFFIX=${3:?RUN_SUFFIX required}   # used in output path and wandb run name

source scripts/delta/bashrc.sh

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
