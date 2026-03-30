"""Custom lm_eval filter for N-MARS <UNDO> post-processing.

lm_eval decodes model outputs to text before filtering.  We implement a
text-level stack post-processor: the text is split into whitespace-delimited
segments, each ``<UNDO>`` pops the previous segment, and the remaining
segments are rejoined.

This approximates the token-level stack_postprocess from
``n_mars.inference.decoder`` and is sufficient for answer extraction from
GSM8K / MATH responses where answers live at the word/number level.

Usage with lm_eval ``--include_path``:
    lm_eval ... --include_path src/n_mars/eval
The task YAML references this filter by its registered name ``nmars_undo``.
"""

from __future__ import annotations

from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter

UNDO_TOKEN = "<UNDO>"


def _text_stack_postprocess(text: str, undo_token: str = UNDO_TOKEN) -> str:
    """Apply stack-based post-processing to a decoded text string.

    Splits on whitespace, treats each ``undo_token`` occurrence as a pop
    operation that removes the previous word from the stack.

    Example:
        "The answer is 5 <UNDO> 4" -> "The answer is 4"
    """
    words = text.split()
    stack: list[str] = []
    for word in words:
        if word == undo_token:
            if stack:
                stack.pop()
        else:
            stack.append(word)
    return " ".join(stack)


@register_filter("nmars_undo")
class NMARSUndoFilter(Filter):
    """lm_eval filter that removes <UNDO> tokens via stack-based post-processing.

    Registered as ``nmars_undo`` for use in task YAML configs.

    Args:
        undo_token: The literal undo token string (default: ``"<UNDO>"``).
    """

    def __init__(self, undo_token: str = UNDO_TOKEN, **kwargs) -> None:
        self.undo_token = undo_token
        super().__init__(**kwargs)

    def apply(self, resps: list, docs: list) -> list:
        """Post-process model responses by applying stack <UNDO> removal.

        Args:
            resps: List of response lists (each element is a list of strings
                   for a single document, as returned by lm_eval).
            docs: Corresponding document dicts (unused here).

        Returns:
            Filtered responses with the same structure as ``resps``.
        """
        filtered = []
        for resp_group in resps:
            filtered.append(
                [_text_stack_postprocess(r, self.undo_token) for r in resp_group]
            )
        return filtered
