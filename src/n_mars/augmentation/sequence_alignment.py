"""Core sequence alignment augmentation for N-MARS (paper Section 3.1).

Given a reference token sequence y and a model-generated sequence ŷ, this module:
1. Computes maximal matching blocks M(y, ŷ) using difflib.SequenceMatcher
   (equivalent to Myers' O(ND) diff algorithm).
2. Identifies error indices E — positions in ŷ not covered by any matching block.
3. Constructs the augmented sequence y* following Eq 5:
       (..., matched_p_i, error_e_i, UNDO×k_i, correction_c_i, ...)
   where k_i = |e_i| (number of error tokens between consecutive matched blocks).
4. Builds a binary mask M where M_t=0 for error-span positions, M_t=1 elsewhere
   (matched tokens, UNDO tokens, correction tokens). The mask is used in mSFT to
   suppress loss on tokens that the model will later learn to erase.

Usage::

    from n_mars.augmentation.sequence_alignment import build_augmented_sequence

    aug_ids, mask = build_augmented_sequence(
        reference=[1, 2, 3, 4, 5],
        generated=[1, 2, 99, 4, 5],
        undo_token_id=32000,
    )
"""

import difflib
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class MatchingBlock(NamedTuple):
    """A maximal matching block between reference and generated sequences.

    Attributes:
        ref_start: Start index in the reference sequence.
        gen_start: Start index in the generated sequence.
        length:    Number of tokens in the block.
    """

    ref_start: int
    gen_start: int
    length: int


# ---------------------------------------------------------------------------
# Matching block computation
# ---------------------------------------------------------------------------


def compute_matching_blocks(
    reference: list[int],
    generated: list[int],
) -> list[MatchingBlock]:
    """Return maximal matching blocks between *reference* and *generated*.

    Uses :class:`difflib.SequenceMatcher` which implements Myers' O(ND) diff
    algorithm and returns non-overlapping, non-adjacent matching blocks in
    order of appearance.

    The terminal sentinel block (length=0) returned by SequenceMatcher is
    stripped so every returned block has ``length > 0``.

    Args:
        reference: Token IDs of the reference (gold) sequence.
        generated: Token IDs of the model-generated sequence.

    Returns:
        List of :class:`MatchingBlock` tuples ``(ref_start, gen_start, length)``,
        sorted by ``gen_start``.
    """
    matcher = difflib.SequenceMatcher(None, reference, generated, autojunk=False)
    blocks = [
        MatchingBlock(ref_start=b.a, gen_start=b.b, length=b.size)
        for b in matcher.get_matching_blocks()
        if b.size > 0
    ]
    return blocks


# ---------------------------------------------------------------------------
# Augmented sequence construction
# ---------------------------------------------------------------------------


def build_augmented_sequence(
    reference: list[int],
    generated: list[int],
    undo_token_id: int,
) -> tuple[list[int], list[int]]:
    """Build augmented sequence y* and binary mask from reference and generated.

    The augmented sequence interleaves matched tokens from *generated*, error
    spans (tokens in *generated* that differ from *reference*), UNDO tokens,
    and correction tokens (matching tokens from *reference* that follow each
    error span).

    Structure per divergence region (Eq 5 of the paper)::

        [matched_prefix] [error_span] [<UNDO>×k] [correction]

    where ``k = len(error_span)``.  The mask is::

        matched_prefix → 1 (include in loss)
        error_span     → 0 (exclude from loss — model learns to erase these)
        <UNDO>×k       → 1 (include in loss — model learns to emit UNDO)
        correction     → 1 (include in loss — model learns the correct tokens)

    If *generated* perfectly matches *reference* (no error spans), the function
    returns ``(list(generated), [1] * len(generated))``.

    Args:
        reference:      Token IDs of the gold reference sequence.
        generated:      Token IDs of the model-generated sequence.
        undo_token_id:  Integer ID of the ``<UNDO>`` special token.

    Returns:
        A tuple ``(augmented_ids, mask)`` where:
        - ``augmented_ids``: token IDs of the augmented sequence y*.
        - ``mask``: binary list of the same length; 0 marks error positions,
          1 marks all other positions.
    """
    blocks = compute_matching_blocks(reference, generated)

    # We'll build the output by walking through matching blocks and gaps.
    augmented: list[int] = []
    mask: list[int] = []

    gen_cursor = 0  # how far we've consumed in generated
    ref_cursor = 0  # how far we've consumed in reference

    for block in blocks:
        gen_start = block.gen_start
        gen_end = gen_start + block.length
        ref_start = block.ref_start
        ref_end = ref_start + block.length

        # --- Error span: generated tokens between previous block and this one ---
        error_span = generated[gen_cursor:gen_start]
        k = len(error_span)

        if k > 0:
            # Append error tokens with mask=0
            augmented.extend(error_span)
            mask.extend([0] * k)

            # Append k UNDO tokens with mask=1
            augmented.extend([undo_token_id] * k)
            mask.extend([1] * k)

            # Append correction tokens from reference (Eq 5)
            correction = reference[ref_cursor:ref_start]
            augmented.extend(correction)
            mask.extend([1] * len(correction))

        # --- Matched tokens: emit from generated (identical to reference) ---
        matched = generated[gen_start:gen_end]
        augmented.extend(matched)
        mask.extend([1] * len(matched))

        gen_cursor = gen_end
        ref_cursor = ref_end

    # --- Trailing error span after last matching block ---
    trailing_error = generated[gen_cursor:]
    k = len(trailing_error)
    if k > 0:
        augmented.extend(trailing_error)
        mask.extend([0] * k)

        augmented.extend([undo_token_id] * k)
        mask.extend([1] * k)

        # Correction: remaining reference tokens after last matched block
        tail_correction = reference[ref_cursor:]
        augmented.extend(tail_correction)
        mask.extend([1] * len(tail_correction))

    assert len(augmented) == len(mask), (
        f"Length mismatch: augmented={len(augmented)}, mask={len(mask)}"
    )
    return augmented, mask
