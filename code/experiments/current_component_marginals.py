"""Report current five-fold and validation marginals for late cascade stages."""

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
from model_t_probe_utils import FastFinalSelector, load_or_build_model_t_oof, standardize_rows
from occurrence_plausibility_verifier import occurrence_adjusted_scores
from structure_text_fusion_verifier import answer_structures, structure_text_scores
from target_alignment import terminal_target_recall
from target_omitted_factor_verifier import target_adjusted_scores


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    score_sets = load_or_build_model_t_oof(train_bank, train_arrays, train_labels)
    train_selector = FastFinalSelector(train_bank)
    validation_selector = FastFinalSelector(validation_bank)
    train_structures = answer_structures(train_bank)
    validation_structures = answer_structures(validation_bank)
    train_target = standardize_rows(terminal_target_recall(train_bank))
    validation_target = standardize_rows(terminal_target_recall(validation_bank))
    train_target[train_structures != "1-1-1-1"] = 0
    validation_target[validation_structures != "1-1-1-1"] = 0

    validation_with = structure_text_scores(
        validation_bank, validation_arrays, CODE / "runs/model_t_final_cascade"
    )
    validation_with = coverage_adjusted_scores(validation_bank, validation_with)
    validation_with = occurrence_adjusted_scores(validation_bank, validation_with)
    validation_with = target_adjusted_scores(validation_bank, validation_with)
    validation_without = validation_with - 0.125 * validation_target
    without_sets = [scores - 0.125 * train_target for scores in score_sets]
    train_marginals = [
        train_selector.correct_count(train_labels, with_scores)
        - train_selector.correct_count(train_labels, without_scores)
        for with_scores, without_scores in zip(score_sets, without_sets)
    ]
    validation_marginal = (
        validation_selector.correct_count(validation_labels, validation_with)
        - validation_selector.correct_count(validation_labels, validation_without)
    )
    print(
        f"target_weight=0.125 train_marginals={train_marginals} "
        f"validation_marginal={validation_marginal:+d}"
    )
    base_counts = [
        train_selector.correct_count(train_labels, scores)
        for scores in without_sets
    ]
    validation_base = validation_selector.correct_count(
        validation_labels, validation_without
    )
    for weight in np.round(np.arange(0.0, 0.301, 0.025), 3):
        deltas = [
            train_selector.correct_count(
                train_labels, scores + weight * train_target
            )
            - base
            for scores, base in zip(without_sets, base_counts)
        ]
        validation_delta = validation_selector.correct_count(
            validation_labels,
            validation_without + weight * validation_target,
        ) - validation_base
        print(
            f"target_weight={weight:.3f} train_deltas={deltas} "
            f"validation_delta={validation_delta:+d}"
        )


if __name__ == "__main__":
    main()
