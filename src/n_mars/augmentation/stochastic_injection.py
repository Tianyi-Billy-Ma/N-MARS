"""Stochastic error injection augmentation for N-MARS (paper Appendix).

Randomly inserts error tokens (sampled uniformly from the vocabulary) at
random positions in the reference sequence, each error span followed by the
same number of ``<UNDO>`` tokens.  This creates synthetic "mistake + erase"
patterns that train the model to learn the UNDO mechanism without requiring
a separate generative model to produce errors.

Usage::

    import random
    from n_mars.augmentation.stochastic_injection import stochastic_augment

    rng = random.Random(42)
    aug_ids, mask = stochastic_augment(
        reference=[1, 2, 3, 4, 5],
        undo_token_id=32000,
        vocab_size=32001,
        num_insertions=2,
        max_error_len=3,
        rng=rng,
    )
"""

import random as random_module


def stochastic_augment(
    reference: list[int],
    undo_token_id: int,
    vocab_size: int,
    num_insertions: int = 1,
    max_error_len: int = 5,
    rng: random_module.Random | None = None,
) -> tuple[list[int], list[int]]:
    """Return a stochastically augmented sequence and its binary mask.

    For each of *num_insertions* insertion points, a contiguous span of
    ``k`` random tokens (1 ≤ k ≤ max_error_len) is inserted into the
    reference sequence, followed by ``k`` ``<UNDO>`` tokens.  The mask
    marks inserted error tokens with 0 and all other positions (reference
    tokens and UNDO tokens) with 1.

    Insertion positions are sampled without replacement from the set of
    valid indices ``[0, len(reference)]``.  Insertions are applied in
    increasing order of position so that earlier insertions do not shift
    later ones.

    Args:
        reference:       Token IDs of the gold reference sequence.
        undo_token_id:   Integer ID of the ``<UNDO>`` special token.
        vocab_size:      Total vocabulary size (used to sample random tokens).
                         ``undo_token_id`` is excluded from the sample pool.
        num_insertions:  Number of error insertion points (default: 1).
        max_error_len:   Maximum length of each injected error span (default: 5).
        rng:             :class:`random.Random` instance for reproducibility.
                         If *None*, a fresh instance with a random seed is used.

    Returns:
        A tuple ``(augmented_ids, mask)`` where:
        - ``augmented_ids``: token IDs with injected errors and UNDO tokens.
        - ``mask``: binary list; 0 at error positions, 1 everywhere else.
    """
    if rng is None:
        rng = random_module.Random()

    # Clamp num_insertions to the number of valid positions
    max_positions = len(reference) + 1
    num_insertions = min(num_insertions, max_positions)

    # Sample insertion positions (without replacement), then sort ascending
    insertion_positions = sorted(rng.sample(range(max_positions), num_insertions))

    # Valid token IDs: entire vocab minus the UNDO token itself
    valid_ids = [i for i in range(vocab_size) if i != undo_token_id]

    augmented: list[int] = []
    mask: list[int] = []

    ref_cursor = 0

    for pos in insertion_positions:
        # Append reference tokens up to this insertion point
        chunk = reference[ref_cursor:pos]
        augmented.extend(chunk)
        mask.extend([1] * len(chunk))
        ref_cursor = pos

        # Sample error length k in [1, max_error_len]
        k = rng.randint(1, max_error_len)

        # Sample k random error tokens (uniform over valid vocab)
        error_tokens = [rng.choice(valid_ids) for _ in range(k)]

        # Append error span (mask=0) then UNDO×k (mask=1)
        augmented.extend(error_tokens)
        mask.extend([0] * k)

        augmented.extend([undo_token_id] * k)
        mask.extend([1] * k)

    # Append remaining reference tokens after last insertion
    tail = reference[ref_cursor:]
    augmented.extend(tail)
    mask.extend([1] * len(tail))

    assert len(augmented) == len(mask), (
        f"Length mismatch: augmented={len(augmented)}, mask={len(mask)}"
    )
    return augmented, mask
