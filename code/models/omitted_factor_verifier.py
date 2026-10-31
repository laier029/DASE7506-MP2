"""Model R: Model Q plus a margin-gated omitted-factor group switch."""

from pathlib import Path

import numpy as np

from coverage_cascade_verifier import coverage_adjusted_scores
from numeric_plausibility import fractional_answer_groups
from occurrence_plausibility_verifier import occurrence_adjusted_scores
from omitted_factor import select_with_omitted_factor_switch
from structure_text_fusion_verifier import structure_text_scores


def predict(bank, arrays, checkpoint):
    """Run Model Q, then switch only on explicit omitted-factor evidence."""
    checkpoint = Path(checkpoint)
    scores = structure_text_scores(bank, arrays, checkpoint)
    scores = coverage_adjusted_scores(bank, scores)
    scores = occurrence_adjusted_scores(bank, scores)
    with np.load(checkpoint / "model.npz", allow_pickle=False) as model:
        group_mean_weight = float(model["group_mean_weight"].item())
    return select_with_omitted_factor_switch(
        bank,
        scores,
        group_mean_weight,
        fractional_answer_groups(bank),
    )
