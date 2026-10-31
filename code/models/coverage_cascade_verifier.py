"""Model O: Model N plus calculation-coverage scoring for 2-1-1 groups."""

from pathlib import Path

import numpy as np

from calculation_coverage import calculation_unique_coverage
from cascade_verifier import select_after_negative_filter
from structure_text_fusion_verifier import answer_structures, structure_text_scores


_COVERAGE_WEIGHT = 0.40
_ACTIVE_STRUCTURE = "2-1-1"


def coverage_adjusted_scores(bank, scores):
    """Reward using distinct question quantities in actual calculations."""
    coverage = calculation_unique_coverage(bank)
    centered = coverage - coverage.mean(axis=1, keepdims=True)
    normalized = centered / (coverage.std(axis=1, keepdims=True) + 1e-6)
    rows = answer_structures(bank) == _ACTIVE_STRUCTURE
    adjusted = scores.copy()
    adjusted[rows] += _COVERAGE_WEIGHT * normalized[rows]
    return adjusted


def predict(bank, arrays, checkpoint):
    """Apply Model N's filter, then structure-specific coverage scoring."""
    checkpoint = Path(checkpoint)
    scores = structure_text_scores(bank, arrays, checkpoint)
    scores = coverage_adjusted_scores(bank, scores)
    with np.load(checkpoint / "model.npz", allow_pickle=False) as model:
        group_mean_weight = float(model["group_mean_weight"].item())
    return select_after_negative_filter(bank, scores, group_mean_weight)
