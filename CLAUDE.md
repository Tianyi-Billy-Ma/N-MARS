# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

N-MARS (Non-Monotonic Autoregressive Sequence Models) is an LLM training and evaluation research project. It uses **ms-swift** for SFT/GRPO fine-tuning and **lm_eval** for benchmarking, with experiments tracked via **Weights & Biases** (project: `n-mars`).

## Package Management

This project uses **uv**. Always use `uv` — never `pip` directly.

```bash
uv sync                    # install all dependencies
uv sync --dev              # include dev dependencies
uv add <package>           # add a runtime dependency
uv add --dev <package>     # add a dev dependency
uv run <command>           # run a command in the venv
```

## Common Commands

### Training (ms-swift)

```bash
# SFT (single GPU)
uv run swift sft --config configs/train/sft_template.yaml

# SFT (multi-GPU, 4 GPUs)
uv run torchrun --nproc_per_node=4 $(which swift) sft --config configs/train/sft_template.yaml

# GRPO
uv run torchrun --nproc_per_node=4 $(which swift) rlhf --config configs/train/grpo_template.yaml
```

### Evaluation (lm_eval)

Eval is driven by CLI args, not YAML configs. YAML files under `configs/eval/` are reference documentation only.

```bash
# args: <model_path> <task_name> <run_suffix>
uv run accelerate launch -m lm_eval \
  --model hf \
  --model_args pretrained=<model_path> \
  --tasks gsm8k \
  --batch_size auto:32 \
  --output_path outputs/<run_suffix> \
  --wandb_args project=n-mars,name=gsm8k-<run_suffix>,group=lm_eval \
  --seed 42 --log_samples
```

`accelerate launch` runs DDP across all visible GPUs (each GPU holds a full model copy and processes different batches). On HPC, pass the three positional args to the eval scripts (see HPC section).

### Linting / Tests

```bash
uv run ruff check src/          # lint
uv run ruff format src/         # format
uv run pytest                   # run tests
```

## Output Naming Convention

All experiment outputs go under `outputs/` using the slug pattern:

```
outputs/<model_name>-<dataset_name>-<YYYYMMDD_HHMMSS>/
```

Examples:
- `outputs/llama3.1-8b-gsm8k-20240326_143000/`
- `outputs/qwen2.5-7b-numina-grpo-20240326_143000/`

Both `output_dir` (ms-swift) and `output_path` (lm_eval) in YAML configs must follow this pattern. The W&B `run_name` should match the slug.

## Repository Structure

```
configs/
  train/         # ms-swift YAML configs (sft_template.yaml, grpo_template.yaml)
  eval/          # lm_eval YAML configs (gsm8k.yaml, mbpp.yaml, math500.yaml)
external/        # third-party reference codebases (read-only, never import directly)
outputs/         # experiment results (gitignored except .gitkeep)
scripts/
  delta/         # SLURM scripts for UIUC Delta HPC
  crc/           # SLURM scripts for Notre Dame CRC
src/n_mars/      # installable Python package
logs/            # SLURM stdout/stderr (create before submitting)
```

### `external/` Convention

- **Purpose:** Store cloned third-party codebases for reference only.
- **Isolation:** Code under `src/` must NEVER import from `external/`. The two directories are strictly isolated.
- **Reuse:** If you need to use code from `external/`, copy it into `src/` and adapt it there.
- **Read-only:** Do not modify files inside `external/` repos.

## HPC Job Submission

### Notre Dame CRC (SGE — uses `qsub`)

- Account: `tma2`, Queue: `gpu`
- Email: `tma2@nd.edu`

```bash
qsub scripts/crc/sft_train.sh configs/train/sft_template.yaml
qsub scripts/crc/grpo_train.sh configs/train/grpo_template.yaml
qsub scripts/crc/eval.sh configs/eval/gsm8k.yaml
```

### UIUC Delta (SLURM — uses `sbatch`)

- Account: `bgdn-delta-gpu`, Partition: `gpuA100x4` (or `gpuA40x4`)
- Email: `tma2@nd.edu`

```bash
sbatch scripts/delta/sft_train.sh configs/train/sft_template.yaml
sbatch scripts/delta/grpo_train.sh configs/train/grpo_template.yaml
sbatch scripts/delta/eval.sh configs/eval/gsm8k.yaml
```

Before submitting on either cluster, create the log directories:
```bash
mkdir -p logs/crc logs/delta
```

## Config Conventions

- All configs are YAML.
- Training configs (`configs/train/`) are ms-swift format — use `swift sft` or `swift rlhf`.
- Eval configs (`configs/eval/`) are lm_eval format — use `lm_eval --config`.
- Always set `report_to: wandb` and `run_name` matching the output slug in training configs.
- Always set `wandb_args.project: n-mars` and `wandb_args.name` matching the slug in eval configs.

## Target Models

All experiments use **base models** (not instruct variants):

| Model | HuggingFace ID |
|-------|---------------|
| Llama-3.2-1B | `meta-llama/Llama-3.2-1B` |
| Llama-3.1-8B | `meta-llama/Llama-3.1-8B` |
| Qwen3-4B | `Qwen/Qwen3-4B` |

Use these IDs in the `model` field of training configs and `model_args.pretrained` in eval configs.

## Hard Rules

- **No `__pycache__` folders.** Never create `__pycache__` directories. After running `uv` or any Python command during your workflow, check whether `__pycache__` folders were created and remove them if they exist (`find . -type d -name __pycache__ -exec rm -rf {} +`).

## Key Libraries

| Library | Purpose | Docs |
|---------|---------|------|
| ms-swift | SFT & GRPO fine-tuning | https://swift.readthedocs.io |
| lm_eval | Standardised benchmarking | https://github.com/EleutherAI/lm-evaluation-harness |
| wandb | Experiment tracking | https://docs.wandb.ai |
