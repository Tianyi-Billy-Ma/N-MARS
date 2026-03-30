"""Hard sample augmentation for N-MARS (paper Appendix).

GSM-Symbolic style augmentation: given a reference sequence and an *alternative*
sequence derived from the same problem template but with different numerical
values, we use the alternative's divergent tokens as hard error spans.

At each divergence point between reference and alternative:
  1. Emit the matched prefix tokens (mask=1).
  2. Emit the alternative's divergent tokens as the error span (mask=0).
  3. Emit ``<UNDO>×k`` tokens (mask=1) where k = len(error_span).
  4. Emit the reference's correct tokens at that position (mask=1).

This produces harder training examples than stochastic injection because the
error tokens are semantically plausible (they come from a valid but wrong
problem instantiation) rather than arbitrary vocabulary items.

Usage::

    from n_mars.augmentation.hard_sample import hard_sample_augment

    aug_ids, mask = hard_sample_augment(
        reference=[1, 2, 3, 4, 5],
        alternative=[1, 2, 99, 100, 5],
        undo_token_id=32000,
    )
"""

import difflib


def hard_sample_augment(
    reference: list[int],
    alternative: list[int],
    undo_token_id: int,
) -> tuple[list[int], list[int]]:
    """Build a hard-sample augmented sequence from reference and alternative.

    Computes maximal matching blocks between *reference* and *alternative*
    (using :class:`difflib.SequenceMatcher`).  At each gap between matched
    blocks, the alternative's tokens form the error span; the reference's
    tokens at the same position form the correction.

    The output sequence structure at each divergence region is::

        [matched_prefix] [alt_error_span] [<UNDO>×k] [ref_correction]

    where ``k = max(len(alt_error_span), len(ref_correction))`` so that
    the UNDO count covers the full error span length.

    The mask is:
    - 0 at ``alt_error_span`` positions (errors to be erased).
    - 1 at all other positions (matched, UNDO, corrections).

    If *reference* and *alternative* are identical, the function returns the
    reference unchanged with an all-ones mask.

    Args:
        reference:      Token IDs of the gold reference sequence.
        alternative:    Token IDs of the alternative (hard-error source) sequence.
        undo_token_id:  Integer ID of the ``<UNDO>`` special token.

    Returns:
        A tuple ``(augmented_ids, mask)`` where:
        - ``augmented_ids``: token IDs of the augmented sequence.
        - ``mask``: binary list; 0 at error positions, 1 everywhere else.
    """
    matcher = difflib.SequenceMatcher(None, reference, alternative, autojunk=False)
    # opcodes: list of (tag, i1, i2, j1, j2)
    # tags: 'equal', 'replace', 'insert', 'delete'
    opcodes = matcher.get_opcodes()

    augmented: list[int] = []
    mask: list[int] = []

    for tag, ref_i1, ref_i2, alt_j1, alt_j2 in opcodes:
        if tag == "equal":
            # Matched block — emit reference (== alternative) tokens, mask=1
            chunk = reference[ref_i1:ref_i2]
            augmented.extend(chunk)
            mask.extend([1] * len(chunk))

        elif tag in ("replace", "insert", "delete"):
            # Divergence: alternative tokens are the error, reference tokens
            # are the correction.
            alt_error = alternative[alt_j1:alt_j2]   # what the alt (wrong) model generated
            ref_correction = reference[ref_i1:ref_i2] # what should be there

            k_error = len(alt_error)
            k_correction = len(ref_correction)

            if k_error == 0 and k_correction == 0:
                continue

            if k_error > 0:
                # Emit error span with mask=0
                augmented.extend(alt_error)
                mask.extend([0] * k_error)

                # Emit UNDO×k_error with mask=1
                augmented.extend([undo_token_id] * k_error)
                mask.extend([1] * k_error)

            if k_correction > 0:
                # Emit the reference correction with mask=1
                augmented.extend(ref_correction)
                mask.extend([1] * k_correction)

    assert len(augmented) == len(mask), (
        f"Length mismatch: augmented={len(augmented)}, mask={len(mask)}"
    )
    return augmented, mask
