"""Model loading utilities with optional LoRA."""
from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_name_or_path: str,
    use_lora: bool = False,
    lora_rank: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: str = "all-linear",
    bf16: bool = True,
    gradient_checkpointing: bool = True,
):
    """Load model and tokenizer, optionally with LoRA."""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        use_cache=False,
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model  # noqa: PLC0415

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def save_model(model, tokenizer, output_dir: str, use_lora: bool = False):
    """Save model and tokenizer. Merges LoRA weights if applicable."""
    from pathlib import Path  # noqa: PLC0415

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if use_lora:
        # Merge LoRA weights and save full model
        merged = model.merge_and_unload()
        merged.save_pretrained(output_dir)
    else:
        model.save_pretrained(output_dir)

    tokenizer.save_pretrained(output_dir)
    logger.info("Model saved to %s", output_dir)
