"""Build Self-Reflect training data from N-MARS augmented traces.

Replaces <|BACKTRACK|> tokens with natural language reflection markers.
Unlike N-MARS, error tokens are NOT masked — standard SFT on all tokens.
At inference, the full output (including errors + NL marker + correction)
is generated, and only the final answer is extracted.

Uses mtybilly/GSM8K-Random-All dataset.

Usage:
    python -m baselines.self_reflect.build_data \
        --output_dir data/self_reflect/gsm8k \
        --dataset_config p0.1_n10 \
        --seed 42
"""

import argparse
import json
import re
from pathlib import Path

from datasets import Dataset, load_dataset

DATASET_ID = "mtybilly/GSM8K-Random-All"
SRC_BACKTRACK = "<|BACKTRACK|>"
NL_MARKER = "Wait, that is incorrect. Let me reconsider."

PROMPT_TEMPLATE = "Question: {query}\nAnswer:\n"


def convert_backtrack_to_nl(text: str, n_backtrack: int) -> str:
    """Replace a run of N <|BACKTRACK|> tokens with a single NL marker.

    The BACKTRACK tokens indicate N tokens to erase, but in the NL
    setting we cannot erase — we just insert the reflection marker
    once and let the correction follow.
    """
    # Replace all consecutive BACKTRACK runs with a single NL marker
    pattern = re.compile(re.escape(SRC_BACKTRACK) + r"(?:\s*" + re.escape(SRC_BACKTRACK) + r")*")
    return pattern.sub(f" {NL_MARKER}\n", text)


def build_samples(
    dataset_config: str, seed: int,
) -> list[dict]:
    """Load augmented traces and convert to Self-Reflect format."""
    ds = load_dataset(DATASET_ID, dataset_config, split="train")

    samples = []
    for row in ds:
        query = row["query"]
        response = row["response"]
        backtrack_response = row["backtrack_response"]

        # D_op: optimal path (no backtracking) — same as N-MARS
        op_text = f"{PROMPT_TEMPLATE.format(query=query)}{response}"
        samples.append({
            "text": op_text,
            "has_reflection": False,
            "split": "op",
        })

        # D_reflect: replace BACKTRACK with NL marker
        if SRC_BACKTRACK in backtrack_response:
            nl_response = convert_backtrack_to_nl(backtrack_response, 0)
            reflect_text = (
                f"{PROMPT_TEMPLATE.format(query=query)}{nl_response}"
            )
            samples.append({
                "text": reflect_text,
                "has_reflection": True,
                "split": "reflect",
            })

    return samples


def save_samples(samples: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_dataset = Dataset.from_list(samples)
    hf_dataset.save_to_disk(str(output_dir / "hf_dataset"))

    jsonl_path = output_dir / "data.jsonl"
    with open(jsonl_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    n_op = sum(1 for s in samples if s["split"] == "op")
    n_ref = sum(1 for s in samples if s["split"] == "reflect")
    print(f"Saved {n_op} D_op + {n_ref} D_reflect = {len(samples)} total")
    print(f"  HuggingFace dataset: {output_dir / 'hf_dataset'}")
    print(f"  JSONL: {jsonl_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Self-Reflect training data from N-MARS traces"
    )
    parser.add_argument(
        "--output_dir", type=Path,
        default=Path("data/self_reflect/gsm8k"),
    )
    parser.add_argument(
        "--dataset_config", type=str, default="p0.1_n10",
        choices=["p0.1_n10", "p1_n1", "p1_n3"],
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Building Self-Reflect data from N-MARS augmented traces")
    print(f"  config={args.dataset_config}, seed={args.seed}")
    print(f"  NL marker: '{NL_MARKER}'")
    print(f"  output_dir={args.output_dir}")

    samples = build_samples(args.dataset_config, args.seed)
    save_samples(samples, args.output_dir)


if __name__ == "__main__":
    main()
