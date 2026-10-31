"""Choose stable-signal weights by worst-fold OOF improvement, not validation."""

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
from implicit_factor_probe import implicit_features
from model_t_consistency_soft_probe import consistency_features
from model_t_probe_utils import FastFinalSelector, load_or_build_model_t_oof, standardize_rows
from occurrence_plausibility_verifier import occurrence_adjusted_scores
from structure_text_fusion_verifier import answer_structures, structure_text_scores
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
    train_structures = answer_structures(train_bank)
    validation_structures = answer_structures(validation_bank)
    train_implicit = standardize_rows(implicit_features(train_bank)[:, :, -1])
    validation_implicit = standardize_rows(implicit_features(validation_bank)[:, :, -1])
    train_broad = standardize_rows(consistency_features(train_bank)[:, :, 2])
    validation_broad = standardize_rows(consistency_features(validation_bank)[:, :, 2])
    train_broad[train_structures != "2-1-1"] = 0
    validation_broad[validation_structures != "2-1-1"] = 0
    base_counts = [train_selector.correct_count(train_labels, scores) for scores in score_sets]
    validation_base = validation_selector.correct_count(validation_labels, validation_scores)

    trials = []
    for implicit_weight in np.round(np.arange(0.0, 1.001, 0.05), 3):
        for broad_weight in np.round(np.arange(-0.30, 0.001, 0.05), 3):
            adjustment = implicit_weight * train_implicit + broad_weight * train_broad
            deltas = [
                train_selector.correct_count(train_labels, scores + adjustment) - base
                for scores, base in zip(score_sets, base_counts)
            ]
            trials.append(
                (
                    min(deltas),
                    sum(deltas),
                    -abs(float(implicit_weight)) - abs(float(broad_weight)),
                    implicit_weight,
                    broad_weight,
                    deltas,
                )
            )
    trials.sort(reverse=True)
    for _, _, _, implicit_weight, broad_weight, deltas in trials[:12]:
        validation_adjustment = (
            implicit_weight * validation_implicit + broad_weight * validation_broad
        )
        validation_delta = validation_selector.correct_count(
            validation_labels, validation_scores + validation_adjustment
        ) - validation_base
        print(
            f"implicit={implicit_weight:.2f} broad={broad_weight:.2f} "
            f"train_deltas={deltas} validation_delta={validation_delta:+d}",
            flush=True,
        )


if __name__ == "__main__":
    main()
