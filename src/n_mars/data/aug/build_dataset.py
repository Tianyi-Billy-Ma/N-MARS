"""CLI script to build the D_aug augmented dataset from GSM8K.

Loads the GSM8K training split from HuggingFace, tokenizes each sample, applies
one of three augmentation strategies, and saves the result as both a HuggingFace
Dataset (arrow) and a JSONL file.

Augmentation methods
--------------------
alignment
    Generates a completion from a base model (specified by ``--model_path``),
    then aligns it against the reference answer using Myers' diff (Section 3.1).
stochastic
    Randomly injects error tokens into the reference sequence (paper Appendix).
hard
    Uses an alternative GSM8K problem (same template, different numbers) as
    the hard-error source (paper Appendix).  Falls back to stochastic injection
    when no suitable alternative is available.

Usage::

    python -m n_mars.augmentation.build_dataset \\
        --output_dir data/augmentation/gsm8k \\
        --augmentation_method alignment \\
        --model_path meta-llama/Llama-3.1-8B \\
        --max_samples 1000 \\
        --seed 42
"""

import argparse
import json
import random
from pathlib import Path

from datasets import Dataset, load_dataset

from n_mars.augmentation.hard_sample import hard_sample_augment
from n_mars.augmentation.sequence_alignment import build_augmented_sequence
from n_mars.augmentation.stochastic_injection import stochastic_augment

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UNDO_TOKEN = "<UNDO>"
PROMPT_TEMPLATE = "###Question: {question}\n###Response:\n"

# ---------------------------------------------------------------------------
# GSM8K parsing  (mirrors build_data.py in the self_backtracking baseline)
# ---------------------------------------------------------------------------


def parse_gsm8k_answer(answer_field: str) -> tuple[list[str], str]:
    """Split a GSM8K answer field into reasoning steps and final answer.

    The answer field looks like::

        Step one text\\nStep two text\\n#### 42

    Returns:
        steps: list of non-empty lines before the #### delimiter.
        final_answer: the numeric string after ####.
    """
    parts = answer_field.split("####")
    cot_text = parts[0].rstrip()
    final_answer = parts[1].strip() if len(parts) > 1 else ""
    steps = [line for line in cot_text.splitlines() if line.strip()]
    return steps, final_answer


def build_reference_text(question: str, steps: list[str], final_answer: str) -> str:
    """Construct full reference text (prompt + CoT + answer)."""
    body = "\n".join(steps)
    return f"{PROMPT_TEMPLATE.format(question=question)}{body}\nThe answer is {final_answer}."


# ---------------------------------------------------------------------------
# Model-based completion generation (alignment method only)
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(model_path: str):
    """Load a HuggingFace causal LM and its tokenizer.

    The tokenizer is extended with the ``<UNDO>`` special token if it is not
    already present.

    Returns:
        tokenizer: AutoTokenizer instance.
        model:     AutoModelForCausalLM instance (float16, device=cuda if available).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if UNDO_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [UNDO_TOKEN]})

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    return tokenizer, model


def generate_completion(
    prompt: str,
    tokenizer,
    model,
    max_new_tokens: int = 256,
) -> str:
    """Generate a single greedy completion for *prompt* using *model*.

    Args:
        prompt:         Input prompt string.
        tokenizer:      HuggingFace tokenizer.
        model:          HuggingFace causal LM.
        max_new_tokens: Maximum number of tokens to generate.

    Returns:
        Generated text (prompt excluded).
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=False)


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------


def get_undo_token_id(tokenizer) -> int:
    """Return the token ID for ``<UNDO>``, adding it if necessary."""
    vocab = tokenizer.get_vocab()
    if UNDO_TOKEN not in vocab:
        tokenizer.add_special_tokens({"additional_special_tokens": [UNDO_TOKEN]})
    return tokenizer.convert_tokens_to_ids(UNDO_TOKEN)


def ids_to_text_with_undo(token_ids: list[int], tokenizer) -> str:
    """Decode token IDs back to text, keeping ``<UNDO>`` as a literal string."""
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def mask_to_str(mask: list[int]) -> str:
    """Compact binary mask representation as a comma-joined string."""
    return ",".join(map(str, mask))


# ---------------------------------------------------------------------------
# Per-sample augmentation
# ---------------------------------------------------------------------------


def augment_alignment(
    question: str,
    steps: list[str],
    final_answer: str,
    tokenizer,
    model,
    undo_token_id: int,
) -> dict:
    """Apply sequence-alignment augmentation to a single GSM8K sample.

    Generates a completion from *model*, tokenizes both the completion and the
    reference answer, computes matching blocks, and builds y*.

    Returns a dict with keys: ``text``, ``augmented_text``, ``mask``,
    ``augmentation_method``.
    """
    reference_text = build_reference_text(question, steps, final_answer)
    prompt = PROMPT_TEMPLATE.format(question=question)

    # Generate completion from policy model
    completion = generate_completion(prompt, tokenizer, model)
    generated_text = prompt + completion

    # Tokenize (IDs only, no special prompt tokens added again)
    ref_ids = tokenizer.encode(reference_text, add_special_tokens=False)
    gen_ids = tokenizer.encode(generated_text, add_special_tokens=False)

    aug_ids, mask = build_augmented_sequence(ref_ids, gen_ids, undo_token_id)
    augmented_text = ids_to_text_with_undo(aug_ids, tokenizer)

    return {
        "text": reference_text,
        "augmented_text": augmented_text,
        "mask": mask_to_str(mask),
        "augmentation_method": "alignment",
    }


def augment_stochastic(
    question: str,
    steps: list[str],
    final_answer: str,
    tokenizer,
    undo_token_id: int,
    rng: random.Random,
    num_insertions: int = 2,
    max_error_len: int = 5,
) -> dict:
    """Apply stochastic error injection to a single GSM8K sample."""
    reference_text = build_reference_text(question, steps, final_answer)
    ref_ids = tokenizer.encode(reference_text, add_special_tokens=False)

    aug_ids, mask = stochastic_augment(
        reference=ref_ids,
        undo_token_id=undo_token_id,
        vocab_size=len(tokenizer),
        num_insertions=num_insertions,
        max_error_len=max_error_len,
        rng=rng,
    )
    augmented_text = ids_to_text_with_undo(aug_ids, tokenizer)

    return {
        "text": reference_text,
        "augmented_text": augmented_text,
        "mask": mask_to_str(mask),
        "augmentation_method": "stochastic",
    }


def augment_hard(
    question: str,
    steps: list[str],
    final_answer: str,
    alternative_row: dict | None,
    tokenizer,
    undo_token_id: int,
    rng: random.Random,
) -> dict:
    """Apply hard-sample augmentation to a single GSM8K sample.

    Uses *alternative_row* (another GSM8K sample) as the hard-error source.
    Falls back to stochastic injection when *alternative_row* is None.
    """
    reference_text = build_reference_text(question, steps, final_answer)
    ref_ids = tokenizer.encode(reference_text, add_special_tokens=False)

    if alternative_row is not None:
        alt_steps, alt_final = parse_gsm8k_answer(alternative_row["answer"])
        alt_text = build_reference_text(
            alternative_row["question"], alt_steps, alt_final
        )
        alt_ids = tokenizer.encode(alt_text, add_special_tokens=False)
        aug_ids, mask = hard_sample_augment(ref_ids, alt_ids, undo_token_id)
        method = "hard"
    else:
        # Fallback: stochastic injection
        aug_ids, mask = stochastic_augment(
            reference=ref_ids,
            undo_token_id=undo_token_id,
            vocab_size=len(tokenizer),
            num_insertions=2,
            max_error_len=5,
            rng=rng,
        )
        method = "hard_fallback_stochastic"

    augmented_text = ids_to_text_with_undo(aug_ids, tokenizer)

    return {
        "text": reference_text,
        "augmented_text": augmented_text,
        "mask": mask_to_str(mask),
        "augmentation_method": method,
    }


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def build_dataset(
    augmentation_method: str,
    model_path: str | None,
    max_samples: int | None,
    seed: int,
) -> list[dict]:
    """Load GSM8K and construct the D_aug samples.

    Args:
        augmentation_method: One of ``"alignment"``, ``"stochastic"``, ``"hard"``.
        model_path:          HF model path (required for ``"alignment"``).
        max_samples:         If set, only process this many training samples.
        seed:                Random seed for reproducibility.

    Returns:
        List of dicts with keys ``text``, ``augmented_text``, ``mask``,
        ``augmentation_method``.
    """
    rng = random.Random(seed)

    gsm8k = load_dataset("gsm8k", "main")
    train_data = gsm8k["train"]

    if max_samples is not None:
        train_data = train_data.select(range(min(max_samples, len(train_data))))

    # Load model only for alignment method
    tokenizer = None
    model = None
    undo_token_id: int = -1

    if augmentation_method == "alignment":
        if model_path is None:
            raise ValueError("--model_path is required for augmentation_method=alignment")
        print(f"Loading model from {model_path} ...")
        tokenizer, model = load_model_and_tokenizer(model_path)
        undo_token_id = get_undo_token_id(tokenizer)
    else:
        # For stochastic and hard, we only need a tokenizer for encoding/decoding.
        # Use a lightweight default tokenizer; caller should set --model_path to
        # the tokenizer matching their training setup.
        if model_path is None:
            raise ValueError(
                "--model_path (tokenizer path) is required even for non-alignment methods "
                "so that token IDs match the training tokenizer."
            )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if UNDO_TOKEN not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": [UNDO_TOKEN]})
        undo_token_id = tokenizer.convert_tokens_to_ids(UNDO_TOKEN)

    samples: list[dict] = []
    rows = list(train_data)

    for idx, row in enumerate(rows):
        question = row["question"]
        steps, final_answer = parse_gsm8k_answer(row["answer"])

        if augmentation_method == "alignment":
            sample = augment_alignment(
                question, steps, final_answer, tokenizer, model, undo_token_id
            )
        elif augmentation_method == "stochastic":
            sample = augment_stochastic(
                question, steps, final_answer, tokenizer, undo_token_id, rng
            )
        elif augmentation_method == "hard":
            # Pick a different row as the alternative (circular shift by 1)
            alt_idx = (idx + 1) % len(rows)
            alternative_row = rows[alt_idx] if alt_idx != idx else None
            sample = augment_hard(
                question, steps, final_answer, alternative_row, tokenizer, undo_token_id, rng
            )
        else:
            raise ValueError(f"Unknown augmentation_method: {augmentation_method!r}")

        samples.append(sample)

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(rows)} samples")

    return samples


# ---------------------------------------------------------------------------
# Save utilities
# ---------------------------------------------------------------------------


def save_dataset(samples: list[dict], output_dir: Path) -> None:
    """Save *samples* as a HuggingFace Dataset (arrow) and JSONL.

    Args:
        samples:    List of augmented sample dicts.
        output_dir: Output directory (created if it does not exist).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_dataset = Dataset.from_list(samples)
    hf_dataset.save_to_disk(str(output_dir / "hf_dataset"))

    jsonl_path = output_dir / "data.jsonl"
    with open(jsonl_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Saved {len(samples)} samples")
    print(f"  HuggingFace dataset: {output_dir / 'hf_dataset'}")
    print(f"  JSONL: {jsonl_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build D_aug augmented dataset from GSM8K for N-MARS training."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help=(
            "HuggingFace model path or ID. Required for 'alignment' (used as "
            "the policy model for generation). For 'stochastic'/'hard', only "
            "the tokenizer is used — pass the model whose tokenizer you will "
            "train with."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/augmentation/gsm8k"),
        help="Directory to write outputs (default: data/augmentation/gsm8k)",
    )
    parser.add_argument(
        "--augmentation_method",
        choices=["alignment", "stochastic", "hard"],
        default="alignment",
        help="Augmentation strategy to apply (default: alignment)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit the number of training samples processed (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Building D_aug dataset from GSM8K")
    print(f"  augmentation_method={args.augmentation_method}")
    print(f"  model_path={args.model_path}")
    print(f"  output_dir={args.output_dir}")
    print(f"  max_samples={args.max_samples}, seed={args.seed}")

    samples = build_dataset(
        augmentation_method=args.augmentation_method,
        model_path=args.model_path,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    save_dataset(samples, args.output_dir)


if __name__ == "__main__":
    main()
