"""Model U: Model T plus five-fold-stable weak survivor signals."""

import numpy as np

from coverage_cascade_verifier import coverage_adjusted_scores
from numeric_plausibility import implausible_answer_groups
from occurrence_plausibility_verifier import occurrence_adjusted_scores
from omitted_factor import select_with_omitted_factor_switch
from stable_signal_features import stable_signal_adjusted_scores
from structure_text_fusion_verifier import structure_text_scores
from target_omitted_factor_verifier import target_adjusted_scores


def stable_signal_scores(bank, arrays, checkpoint):
    scores = structure_text_scores(bank, arrays, checkpoint)
    scores = coverage_adjusted_scores(bank, scores)
    scores = occurrence_adjusted_scores(bank, scores)
    scores = target_adjusted_scores(bank, scores)
    return stable_signal_adjusted_scores(bank, scores)


def predict(bank, arrays, checkpoint):
    scores = stable_signal_scores(bank, arrays, checkpoint)
    return np.asarray(
        select_with_omitted_factor_switch(
            bank,
            scores,
            0.3,
            implausible_answer_groups(bank),
        ),
        dtype=np.int64,
    ).tolist()
