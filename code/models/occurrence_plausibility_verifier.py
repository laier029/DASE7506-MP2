"""Model Q: Model P plus occurrence coverage for all-distinct answers."""

from pathlib import Path

import numpy as np

from calculation_coverage import calculation_occurrence_coverage
from cascade_verifier import select_after_negative_filter
from coverage_cascade_verifier import coverage_adjusted_scores
from numeric_plausibility import fractional_answer_groups
from structure_text_fusion_verifier import answer_structures, structure_text_scores


_OCCURRENCE_WEIGHT = 0.225
_ACTIVE_STRUCTURE = "1-1-1-1"


def occurrence_adjusted_scores(bank, scores):
    """Reward multiplicity-aware quantity use for all-distinct answers."""
    coverage = calculation_occurrence_coverage(bank)
    normalized = (coverage - coverage.mean(axis=1, keepdims=True)) / (
        coverage.std(axis=1, keepdims=True) + 1e-6
    )
    rows = answer_structures(bank) == _ACTIVE_STRUCTURE
    adjusted = scores.copy()
    adjusted[rows] += _OCCURRENCE_WEIGHT * normalized[rows]
    return adjusted


def predict(bank, arrays, checkpoint):
    """Run both plausibility gates and both structure-specific scorers."""
    checkpoint = Path(checkpoint)
    scores = structure_text_scores(bank, arrays, checkpoint)
    scores = coverage_adjusted_scores(bank, scores)
    scores = occurrence_adjusted_scores(bank, scores)
    with np.load(checkpoint / "model.npz", allow_pickle=False) as model:
        group_mean_weight = float(model["group_mean_weight"].item())
    return select_after_negative_filter(
        bank,
        scores,
        group_mean_weight,
        additional_rejected=fractional_answer_groups(bank),
    )
