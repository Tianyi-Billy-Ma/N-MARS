#!/bin/bash
#SBATCH --job-name=nmars-eval
#SBATCH --account=bgdn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --mail-user=tma2@nd.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/%x_%j.out

# Usage: sbatch scripts/delta/eval.sh <model_path> <task_name> <run_suffix> [max_gen_toks] [num_fewshot]
# Example: sbatch scripts/delta/eval.sh outputs/llama3.1-8b-sft-20240326 gsm8k llama3.1-8b-sft
# Example: sbatch scripts/delta/eval.sh outputs/llama3.1-8b-sft-20240326 gsm8k llama3.1-8b-sft 512 4
MODEL_PATH=${1:?MODEL_PATH required}
TASK_NAME=${2:?TASK_NAME required}
RUN_SUFFIX=${3:?RUN_SUFFIX required}   # used in output path and wandb run name
MAX_GEN_TOKS=${4:-}                     # optional, default 256 (lm_eval default)
NUM_FEWSHOT=${5:-}                      # optional, uses task default if not set

source scripts/delta/bashrc.sh

OUTPUT_PATH=outputs/${RUN_SUFFIX}

MAX_GEN_TOKS_FLAG=""
if [ -n "${MAX_GEN_TOKS}" ]; then
  MAX_GEN_TOKS_FLAG="--gen_kwargs max_gen_toks=${MAX_GEN_TOKS}"
fi

NUM_FEWSHOT_FLAG=""
if [ -n "${NUM_FEWSHOT}" ]; then
  NUM_FEWSHOT_FLAG="--num_fewshot ${NUM_FEWSHOT}"
fi

accelerate launch -m lm_eval \
  --model hf \
  --model_args pretrained=${MODEL_PATH} \
  --tasks ${TASK_NAME} \
  --batch_size auto:32 \
  --output_path ${OUTPUT_PATH} \
  --wandb_args project=${WANDB_PROJECT},name=${TASK_NAME}-${RUN_SUFFIX},group=lm_eval \
  --seed 42 \
  --log_samples \
  --confirm_run_unsafe_code \
  ${MAX_GEN_TOKS_FLAG} \
  ${NUM_FEWSHOT_FLAG}
