"""Custom lm_eval model wrapper for N-MARS with token-level <UNDO> post-processing.

Subclasses lm_eval's HFLM to intercept raw generated token IDs and apply
``stack_postprocess`` (push normal tokens, pop on ``<UNDO>``) *before*
decoding to text.  This is the correct token-level approach — the alternative
text-level filter (``nmars_filter.py``) is only an approximation because a
single whitespace-delimited word may span multiple BPE tokens.

Usage with ``run_eval.py``::

    python -m n_mars.evaluation.run_eval \\
        --model_path outputs/nmars-llama3.2-1b-gsm8k-grpo \\
        --task nmars_gsm8k --seed 42

Or directly with lm_eval (after importing this module to register the model)::

    lm_eval --model nmars_hf \\
        --model_args pretrained=<model_path> \\
        --tasks gsm8k --batch_size auto:32 --seed 42
"""

from __future__ import annotations

from typing import Iterator

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

from n_mars.inference.decoder import stack_postprocess

UNDO_TOKEN = "<UNDO>"


@register_model("nmars_hf", "nmars-hf")
class NMARSHFModel(HFLM):
    """HFLM subclass that applies token-level <UNDO> stack post-processing.

    After the base model generates raw token IDs (which may contain ``<UNDO>``
    tokens), this wrapper resolves them via the stack-based algorithm before
    decoding to text.  The rest of the lm_eval pipeline (answer extraction,
    metric computation) then operates on the clean, post-processed text.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Resolve the <UNDO> token ID in the (already-loaded) tokenizer.
        vocab = self.tokenizer.get_vocab()
        if UNDO_TOKEN in vocab:
            self._undo_token_id: int | None = self.tokenizer.convert_tokens_to_ids(
                UNDO_TOKEN
            )
        else:
            # Token not in vocab — model was not trained with <UNDO>.
            # Fall back to no-op post-processing.
            self._undo_token_id = None

    def tok_decode(
        self, tokens: list[int] | Iterator[int], skip_special_tokens: bool = True
    ) -> str:
        """Decode token IDs, applying stack-based <UNDO> removal first.

        This override is the single point through which lm_eval converts
        generated token IDs to text (used in ``generate_until``).  By
        post-processing here, every downstream consumer (filters, metrics)
        automatically receives clean text.
        """
        if isinstance(tokens, int):
            # Single token ID (e.g., EOS decoding) — pass through unchanged.
            return super().tok_decode(tokens, skip_special_tokens=skip_special_tokens)

        token_list = list(tokens)
        if self._undo_token_id is not None:
            token_list = stack_postprocess(token_list, self._undo_token_id)
        return super().tok_decode(token_list, skip_special_tokens=skip_special_tokens)
