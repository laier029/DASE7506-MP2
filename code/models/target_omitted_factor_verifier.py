"""Model S: Model R plus terminal target recall for all-distinct answers."""

from pathlib import Path

import numpy as np

from coverage_cascade_verifier import coverage_adjusted_scores
from numeric_plausibility import fractional_answer_groups
from occurrence_plausibility_verifier import occurrence_adjusted_scores
from omitted_factor import select_with_omitted_factor_switch
from structure_text_fusion_verifier import answer_structures, structure_text_scores
from target_alignment import terminal_target_recall


_TARGET_WEIGHT = 0.125
_ACTIVE_STRUCTURE = "1-1-1-1"


def target_adjusted_scores(bank, scores):
    """Apply the small target-retention reward on all-distinct answers."""
    feature = terminal_target_recall(bank)
    feature = (feature - feature.mean(axis=1, keepdims=True)) / (
        feature.std(axis=1, keepdims=True) + 1e-6
    )
    rows = answer_structures(bank) == _ACTIVE_STRUCTURE
    adjusted = scores.copy()
    adjusted[rows] += _TARGET_WEIGHT * feature[rows]
    return adjusted


def predict(bank, arrays, checkpoint):
    """Run the full cascade with a final target-retention score."""
    checkpoint = Path(checkpoint)
    scores = structure_text_scores(bank, arrays, checkpoint)
    scores = coverage_adjusted_scores(bank, scores)
    scores = occurrence_adjusted_scores(bank, scores)
    scores = target_adjusted_scores(bank, scores)
    with np.load(checkpoint / "model.npz", allow_pickle=False) as model:
        group_mean_weight = float(model["group_mean_weight"].item())
    return select_with_omitted_factor_switch(
        bank,
        scores,
        group_mean_weight,
        fractional_answer_groups(bank),
    )
