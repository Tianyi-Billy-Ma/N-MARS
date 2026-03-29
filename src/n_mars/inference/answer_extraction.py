"""Answer extraction utilities for N-MARS evaluation.

Supports GSM8K (numeric answers) and MATH (LaTeX boxed answers).
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------

_GSM8K_PATTERNS = [
    re.compile(r"####\s*(\-?[\d,]+(?:\.\d+)?)"),
    re.compile(r"[Tt]he answer is\s*[:\$]?\s*(\-?[\d,]+(?:\.\d+)?)"),
    re.compile(r"[Aa]nswer[:\s]+\$?(\-?[\d,]+(?:\.\d+)?)"),
]


def extract_gsm8k_answer(text: str) -> str | None:
    """Extract a numeric answer from GSM8K-style text.

    Matches patterns like:
      - "#### 42"
      - "The answer is 42"
      - "Answer: 42"

    Returns the numeric string (commas stripped) or None.
    """
    for pat in _GSM8K_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).replace(",", "").strip()
    return None


# ---------------------------------------------------------------------------
# MATH (LaTeX boxed)
# ---------------------------------------------------------------------------

_BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")


def extract_math_answer(text: str) -> str | None:
    """Extract answer from MATH-format text (LaTeX \\boxed{} answers).

    Returns the contents of the last \\boxed{} expression, or None.
    """
    matches = _BOXED_PATTERN.findall(text)
    if matches:
        return matches[-1].strip()
    return None


# ---------------------------------------------------------------------------
# Normalized comparison
# ---------------------------------------------------------------------------

def _normalize(answer: str) -> str:
    """Strip whitespace, commas, dollar signs, and leading zeros."""
    s = answer.replace(",", "").replace("$", "").strip()
    # Remove leading zeros for integer-like strings
    try:
        # Normalize via float to handle "42.0" == "42"
        f = float(s)
        if f == int(f):
            return str(int(f))
        return str(f)
    except ValueError:
        return s.lower()


def answers_match(pred: str | None, gold: str) -> bool:
    """Return True if pred and gold represent the same answer.

    Applies normalization: strips commas, dollar signs, compares as floats
    when possible, falls back to case-insensitive string comparison.

    Args:
        pred: Predicted answer string, or None.
        gold: Gold answer string.

    Returns:
        True if the answers match after normalization.
    """
    if pred is None:
        return False
    try:
        return float(_normalize(pred)) == float(_normalize(gold))
    except ValueError:
        return _normalize(pred) == _normalize(gold)
