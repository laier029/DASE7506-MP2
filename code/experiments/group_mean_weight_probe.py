"""Recheck answer-group mean weights after all Model T cascade stages."""

from pathlib import Path
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from common import load_inputs
from coverage_cascade_verifier import coverage_adjusted_scores
from group_ranker_probe import candidate_labels
from model_t_probe_utils import FastFinalSelector, load_or_build_model_t_oof
from occurrence_plausibility_verifier import occurrence_adjusted_scores
from structure_text_fusion_verifier import structure_text_scores
from target_omitted_factor_verifier import target_adjusted_scores


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    score_sets = load_or_build_model_t_oof(train_bank, train_arrays, train_labels)
    train_selector = FastFinalSelector(train_bank)
    validation_selector = FastFinalSelector(validation_bank)
    validation_scores = structure_text_scores(
        validation_bank, validation_arrays, CODE / "runs/model_t_final_cascade"
    )
    validation_scores = coverage_adjusted_scores(validation_bank, validation_scores)
    validation_scores = occurrence_adjusted_scores(validation_bank, validation_scores)
    validation_scores = target_adjusted_scores(validation_bank, validation_scores)
    for weight in np.round(np.arange(0.0, 0.601, 0.025), 3):
        train_counts = [
            train_selector.correct_count(train_labels, scores, weight)
            for scores in score_sets
        ]
        validation_count = validation_selector.correct_count(
            validation_labels, validation_scores, weight
        )
        print(
            f"weight={weight:.3f} train={train_counts} validation={validation_count}",
            flush=True,
        )


if __name__ == "__main__":
    main()
