"""Model P: integer plausibility gate plus Model O's cascade and scoring."""

from pathlib import Path

import numpy as np

from cascade_verifier import select_after_negative_filter
from coverage_cascade_verifier import coverage_adjusted_scores
from numeric_plausibility import fractional_answer_groups
from structure_text_fusion_verifier import structure_text_scores


def predict(bank, arrays, checkpoint):
    """Reject non-integral answer groups before Model O survivor scoring."""
    checkpoint = Path(checkpoint)
    scores = structure_text_scores(bank, arrays, checkpoint)
    scores = coverage_adjusted_scores(bank, scores)
    with np.load(checkpoint / "model.npz", allow_pickle=False) as model:
        group_mean_weight = float(model["group_mean_weight"].item())
    return select_after_negative_filter(
        bank,
        scores,
        group_mean_weight,
        additional_rejected=fractional_answer_groups(bank),
    )
