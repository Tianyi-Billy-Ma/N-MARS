"""Self-Reflect baseline training script.

Standard SFT on augmented traces where <|BACKTRACK|> is replaced with
natural language reflection markers. No error masking — all tokens
(including errors) are supervised.

Usage:
    python -m baselines.self_reflect.train \
        --model_name_or_path meta-llama/Llama-3.2-1B \
        --data_dir data/self_reflect/gsm8k \
        --output_dir outputs/self-reflect-llama3.2-1b-gsm8k \
        --num_epochs 3 --batch_size 16 --learning_rate 1e-4 \
        --lora_rank 8 --max_length 2048 --seed 42 --bf16
"""

from __future__ import annotations

import argparse
import logging
import os
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

PROMPT_MARKER = "Answer:\n"


def build_tokenize_fn(tokenizer, max_length: int):
    """Standard SFT tokenization — mask only the prompt, supervise everything else."""

    def tokenize(example: dict) -> dict:
        text: str = example["text"]

        encoding = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        input_ids = encoding["input_ids"]
        labels = list(input_ids)

        # Mask prompt tokens (everything up to "Answer:\n")
        marker_pos = text.find(PROMPT_MARKER)
        if marker_pos != -1:
            pre_text = text[: marker_pos + len(PROMPT_MARKER)]
            pre_ids = tokenizer(pre_text, add_special_tokens=False)["input_ids"]
            for i in range(min(len(pre_ids), len(labels))):
                labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"],
            "labels": labels,
        }

    return tokenize


def build_model_and_tokenizer(
    model_name: str,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    bf16: bool,
):
    """Load model and tokenizer, apply LoRA. No special tokens needed."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, use_cache=False,
    )

    if lora_rank > 0:
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
        description="Self-Reflect baseline SFT"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, required=True,
    )
    parser.add_argument(
        "--data_dir", type=Path, required=True,
        help="Directory with saved HuggingFace Dataset (text, has_reflection, split)",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="n-mars")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    set_seed(args.seed)

    model, tokenizer = build_model_and_tokenizer(
        args.model_name_or_path,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bf16=args.bf16,
    )

    dataset = load_from_disk(str(args.data_dir))
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

    num_devices = max(torch.cuda.device_count(), 1)
    per_device_bs = args.per_device_train_batch_size
    grad_accum = max(1, args.batch_size // (per_device_bs * num_devices))
    logger.info(
        "num_devices=%d per_device_bs=%d grad_accum=%d effective_bs=%d",
        num_devices, per_device_bs, grad_accum,
        per_device_bs * num_devices * grad_accum,
    )

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
        remove_unused_columns=False,
    )

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
