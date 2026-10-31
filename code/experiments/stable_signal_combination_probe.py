"""Combine only weak signals that were nonnegative across five OOF schemes."""

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
from target_operation_probe import operation_features


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
    train_total = standardize_rows(operation_features(train_bank)[:, :, 8])
    validation_total = standardize_rows(operation_features(validation_bank)[:, :, 8])

    combinations = (
        ("implicit", 0.35 * train_implicit, 0.35 * validation_implicit),
        (
            "implicit+broad",
            0.35 * train_implicit - 0.15 * train_broad,
            0.35 * validation_implicit - 0.15 * validation_broad,
        ),
        (
            "implicit+broad+total_number",
            0.35 * train_implicit - 0.15 * train_broad + 0.10 * train_total,
            0.35 * validation_implicit - 0.15 * validation_broad + 0.10 * validation_total,
        ),
    )
    base_counts = [train_selector.correct_count(train_labels, scores) for scores in score_sets]
    validation_base = validation_selector.correct_count(validation_labels, validation_scores)
    for name, train_adjustment, validation_adjustment in combinations:
        train_deltas = [
            train_selector.correct_count(train_labels, scores + train_adjustment) - base
            for scores, base in zip(score_sets, base_counts)
        ]
        validation_delta = validation_selector.correct_count(
            validation_labels, validation_scores + validation_adjustment
        ) - validation_base
        print(
            f"combination={name} train_deltas={train_deltas} "
            f"validation_delta={validation_delta:+d}",
            flush=True,
        )


if __name__ == "__main__":
    main()
