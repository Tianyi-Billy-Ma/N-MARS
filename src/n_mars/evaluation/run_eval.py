"""Convenience evaluation script that wraps lm_eval with N-MARS defaults.

Imports the NMARSUndoFilter (which registers it via @register_filter) before
invoking lm_eval's Python API, so the ``nmars_undo`` filter is available to
any task YAML loaded during evaluation.

Usage
-----
# GSM8K with <UNDO> post-processing (default)
python -m n_mars.evaluation.run_eval \\
    --model_path outputs/nmars-llama3.2-1b-gsm8k-grpo \\
    --task nmars_gsm8k --seed 42

# Vanilla GSM8K (no UNDO processing)
python -m n_mars.evaluation.run_eval \\
    --model_path outputs/nmars-llama3.2-1b-gsm8k-grpo \\
    --task gsm8k --seed 42

# Limit to 100 examples for quick debugging
python -m n_mars.evaluation.run_eval \\
    --model_path outputs/nmars-llama3.2-1b-gsm8k-grpo \\
    --task nmars_gsm8k --limit 100 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def _tasks_dir() -> Path:
    """Return the path containing the n_mars custom task YAMLs."""
    return Path(__file__).parent / "tasks"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run lm_eval with N-MARS defaults and UNDO post-processing"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model or HuggingFace model ID",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="nmars_gsm8k",
        help="lm_eval task name (default: nmars_gsm8k)",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default=None,
        help="Suffix for output directory under outputs/. Auto-generated if omitted.",
    )
    parser.add_argument(
        "--batch_size",
        type=str,
        default="auto:32",
        help="Batch size passed to lm_eval (default: auto:32)",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        help="Number of few-shot examples (default: task default)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of evaluation examples (for debugging)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    # Register the N-MARS custom model BEFORE lm_eval runs evaluation.
    # The @register_model decorator fires on import, adding "nmars_hf" to
    # lm_eval's global model registry.  This model subclass applies token-level
    # stack_postprocess (<UNDO> removal) before decoding to text.
    from lm_eval import simple_evaluate
    from lm_eval.tasks import TaskManager

    import n_mars.evaluation.nmars_model  # noqa: F401  # registers nmars_hf model

    # Derive output suffix
    if args.output_suffix:
        suffix = args.output_suffix
    else:
        model_slug = Path(args.model_path).name
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"{model_slug}-{args.task}-{ts}"

    output_path = Path("outputs") / suffix
    output_path.mkdir(parents=True, exist_ok=True)

    tasks_dir = _tasks_dir()
    task_manager = TaskManager(include_path=str(tasks_dir))

    print(f"Running task: {args.task}")
    print(f"Model: {args.model_path}")
    print(f"Output: {output_path}")

    results = simple_evaluate(
        model="nmars_hf",
        model_args=f"pretrained={args.model_path}",
        tasks=[args.task],
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
        log_samples=True,
        task_manager=task_manager,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
    )

    if results is None:
        print("Evaluation returned no results.", file=sys.stderr)
        sys.exit(1)

    # Save results to JSON
    results_file = output_path / "results.json"
    with open(results_file, "w") as fh:
        json.dump(results, fh, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    # Print summary
    if hasattr(results, "results"):
        task_results = results.results
    elif isinstance(results, dict) and "results" in results:
        task_results = results["results"]
    else:
        task_results = {}

    for task_name, metrics in task_results.items():
        print(f"\n=== {task_name} ===")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")


if __name__ == "__main__":
    main()
