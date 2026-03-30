"""SFT training arguments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class SFTArguments:
    # Mode
    mode: str = "full"  # train, eval, full

    # Model
    model_name_or_path: str = "meta-llama/Llama-3.2-1B"

    # LoRA
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: str = "all-linear"

    # Dataset
    dataset: str = "gsm8k"
    max_length: int = 2048

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Eval
    eval_tasks: str = "gsm8k"
    eval_batch_size: str = "auto:32"
    num_fewshot: int | None = None
    max_gen_toks: int | None = None

    # Output
    output_dir: str = "outputs/sft-llama3.2-1b-gsm8k"

    # Logging
    logging_steps: int = 10
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    report_to: str = "wandb"
    run_name: str | None = None
    wandb_project: str = "n-mars"

    # Misc
    seed: int = 42
    dataloader_num_workers: int = 4

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SFTArguments":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def override_from_cli(self, cli_args: dict):
        for k, v in cli_args.items():
            if v is not None and hasattr(self, k):
                setattr(self, k, v)
