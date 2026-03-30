"""Sequence augmentation pipeline for N-MARS (D_aug construction).

Provides three augmentation strategies:
- sequence_alignment: align model-generated tokens against reference via Myers' diff
- stochastic_injection: randomly inject error tokens followed by <UNDO> tokens
- hard_sample: use tokens from an alternative (GSM-Symbolic style) as hard errors
"""

from n_mars.augmentation.hard_sample import hard_sample_augment
from n_mars.augmentation.sequence_alignment import build_augmented_sequence, compute_matching_blocks
from n_mars.augmentation.stochastic_injection import stochastic_augment

__all__ = [
    "build_augmented_sequence",
    "compute_matching_blocks",
    "hard_sample_augment",
    "stochastic_augment",
]
