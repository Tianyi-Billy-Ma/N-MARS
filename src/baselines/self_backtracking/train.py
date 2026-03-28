"""Self-Backtracking baseline training script.

Implements dual-loss SFT from arXiv:2502.04404 (LAMDASZ-ML/Self-Backtracking).

Two data splits are trained jointly:
- D_op: gold optimal CoT paths (standard SFT, prompt masked)
- D_back: traces with error step + <backtrack> token + correction
          (prompt masked AND the error step between second-last and last newline masked)

Usage:
    python -m baselines.self_backtracking.train \
        --model_name_or_path meta-llama/Llama-3.2-1B \
        --data_dir data/self_backtracking/gsm8k \
        --output_dir outputs/self-backtrack-llama3.2-1b-gsm8k \
        --num_epochs 3 --batch_size 16 --learning_rate 1e-4 \
        --lora_rank 8 --max_length 2048 --seed 42 --bf16
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)

BACKTRACK_TOKEN = "<backtrack>"
RESPONSE_MARKER = "###Response:\n"


def build_tokenize_fn(tokenizer, max_length: int):
    """Return a tokenize function that applies the correct masking for each sample.

    Masking rules (from LAMDASZ-ML/Self-Backtracking):
    1. Mask prompt (everything up to and including '###Response:\\n').
    2. For D_back samples (text contains '<backtrack>'): additionally mask the
       error step, which sits between the second-last and last newline in the text.
    """

    def tokenize(example: dict) -> dict:
        text: str = example["text"]

        encoding = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        input_ids = encoding["input_ids"]
        labels = list(input_ids)  # mutable copy

        # --- 1. Mask the prompt (everything before ###Response:\n) ---
        response_start = text.find(RESPONSE_MARKER)
        if response_start != -1:
            pre_response_text = text[: response_start + len(RESPONSE_MARKER)]
        else:
            # Fallback: mask nothing of the response if marker is absent
            pre_response_text = ""

        if pre_response_text:
            pre_response_ids = tokenizer(pre_response_text, add_special_tokens=False)["input_ids"]
            n_prompt = len(pre_response_ids)
            for i in range(min(n_prompt, len(labels))):
                labels[i] = -100

        # --- 2. For D_back samples: mask the error step ---
        if BACKTRACK_TOKEN in text:
            last_index = text.rfind("\n")
            second_last_index = text.rfind("\n", 0, last_index)

            if second_last_index != -1 and last_index != -1:
                before_ids = tokenizer(text[: second_last_index + 1], add_special_tokens=False)[
                    "input_ids"
                ]
                between_ids = tokenizer(text[: last_index + 1], add_special_tokens=False)[
                    "input_ids"
                ]

                start_mask = len(before_ids)
                end_mask = len(between_ids)
                for i in range(start_mask, min(end_mask, len(labels))):
                    labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"],
            "labels": labels,
        }

    return tokenize


def load_dataset(data_dir: Path):
    """Load a HuggingFace Dataset saved to disk from data_dir."""
    dataset = load_from_disk(str(data_dir))
    logger.info("Loaded dataset: %s", dataset)
    return dataset


def build_model_and_tokenizer(
    model_name_or_path: str,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    bf16: bool,
):
    """Load model and tokenizer, add <backtrack> special token, apply LoRA."""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Register <backtrack> as a special token
    tokenizer.add_special_tokens({"additional_special_tokens": [BACKTRACK_TOKEN]})

    torch_dtype = torch.bfloat16 if bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        use_cache=False,  # required for gradient checkpointing
    )

    # Resize embeddings to accommodate the new special token
    model.resize_token_embeddings(len(tokenizer))

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules="all-linear",
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Self-Backtracking baseline SFT (arXiv:2502.04404)"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Directory containing a saved HuggingFace Dataset with columns: "
        "text, has_backtrack, split",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Where to write checkpoints and final model",
    )
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Effective batch size (per-device × gradient_accumulation_steps)",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Per-device batch size; gradient_accumulation computed automatically",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha (scaling factor)",
    )
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Train in bfloat16",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="n-mars",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name (defaults to output_dir stem)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )

    set_seed(args.seed)

    # --- Model and tokenizer ---
    model, tokenizer = build_model_and_tokenizer(
        model_name_or_path=args.model_name_or_path,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bf16=args.bf16,
    )

    # --- Dataset ---
    dataset = load_dataset(args.data_dir)

    # Support both a plain Dataset and a DatasetDict with a "train" split
    if hasattr(dataset, "keys"):
        train_dataset = dataset["train"]
    else:
        train_dataset = dataset

    tokenize_fn = build_tokenize_fn(tokenizer, max_length=args.max_length)
    tokenized = train_dataset.map(
        tokenize_fn,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing",
    )

    # --- Gradient accumulation ---
    num_devices = max(torch.cuda.device_count(), 1)
    per_device_bs = args.per_device_train_batch_size
    grad_accum = max(1, args.batch_size // (per_device_bs * num_devices))
    logger.info(
        "num_devices=%d  per_device_bs=%d  grad_accum=%d  effective_bs=%d",
        num_devices,
        per_device_bs,
        grad_accum,
        per_device_bs * num_devices * grad_accum,
    )

    # --- W&B run name ---
    wandb_run_name = args.wandb_run_name or Path(args.output_dir).name

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=args.bf16,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=args.seed,
        report_to="wandb",
        run_name=wandb_run_name,
        # Disable default data collator padding to max-length (we use Seq2Seq collator)
        remove_unused_columns=False,
    )

    # Set W&B project via env var if not already set
    import os

    if "WANDB_PROJECT" not in os.environ:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        pad_to_multiple_of=8,
        padding=True,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    logger.info("Training complete. Model saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
