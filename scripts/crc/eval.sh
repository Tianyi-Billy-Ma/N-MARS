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

# Usage: qsub scripts/crc/eval.sh <model_path> <task_name> <run_suffix> [max_gen_toks] [num_fewshot]
# Example: qsub scripts/crc/eval.sh outputs/llama3.1-8b-sft-20240326 gsm8k llama3.1-8b-sft
# Example: qsub scripts/crc/eval.sh outputs/llama3.1-8b-sft-20240326 gsm8k llama3.1-8b-sft 512 4
MODEL_PATH=${1:?MODEL_PATH required}
TASK_NAME=${2:?TASK_NAME required}
RUN_SUFFIX=${3:?RUN_SUFFIX required}   # used in output path and wandb run name
MAX_GEN_TOKS=${4:-}                     # optional, default 256 (lm_eval default)
NUM_FEWSHOT=${5:-}                      # optional, uses task default if not set

source scripts/crc/bashrc.sh

OUTPUT_PATH=outputs/${RUN_SUFFIX}

MAX_GEN_TOKS_FLAG=""
if [ -n "${MAX_GEN_TOKS}" ]; then
  MAX_GEN_TOKS_FLAG="--max_gen_toks ${MAX_GEN_TOKS}"
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
  ${MAX_GEN_TOKS_FLAG} \
  ${NUM_FEWSHOT_FLAG}
