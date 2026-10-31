"""Probe explicit final-unit mismatch as a soft penalty on Model Z."""

from pathlib import Path
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from answer_unit_consistency_probe import candidate_final_unit, expected_unit
from common import load_inputs
from dimension_consistency_probe import choices, correct
from group_ranker_probe import candidate_labels
from model_t_probe_utils import load_or_build_model_t_oof, standardize_rows
from semantic_rank_features import semantic_rank_adjusted_scores
from stable_signal_features import stable_signal_adjusted_scores
from stable_signal_verifier import stable_signal_scores
from terminal_dependency_features import terminal_dependency_adjusted_scores


def mismatch_feature(bank):
    output = np.zeros((len(bank), 4), dtype=np.float32)
    for row_index, row in enumerate(bank):
        expected = expected_unit(row["question"])
        if expected is None:
            continue
        for candidate_index, candidate in enumerate(row["candidates"]):
            actual = candidate_final_unit(candidate["text"])
            output[row_index, candidate_index] = float(
                actual is not None and actual != expected
            )
    return standardize_rows(output)


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    train_scores = [
        stable_signal_adjusted_scores(train_bank, scores)
        for scores in load_or_build_model_t_oof(train_bank, train_arrays, train_labels)
    ]
    train_z = [
        terminal_dependency_adjusted_scores(
            train_bank, semantic_rank_adjusted_scores(train_bank, scores)
        )
        for scores in train_scores
    ]
    validation_scores = stable_signal_scores(
        validation_bank, validation_arrays, CODE / "runs/model_z_robust_semantic"
    )
    validation_z = terminal_dependency_adjusted_scores(
        validation_bank,
        semantic_rank_adjusted_scores(validation_bank, validation_scores),
    )
    train_feature = mismatch_feature(train_bank)
    validation_feature = mismatch_feature(validation_bank)
    empty_train = [set() for _ in train_bank]
    empty_validation = [set() for _ in validation_bank]
    base_train = [correct(train_labels, choices(train_bank, scores, empty_train)) for scores in train_z]
    base_validation = correct(
        validation_labels, choices(validation_bank, validation_z, empty_validation)
    )
    trials = []
    for weight in np.round(np.arange(-0.30, 0.001, 0.025), 3):
        selected = choices(
            train_bank, train_z[0] + weight * train_feature, empty_train
        )
        trials.append((correct(train_labels, selected), -abs(weight), float(weight)))
    _, _, weight = max(trials)
    deltas = []
    for scores, base in zip(train_z, base_train):
        selected = choices(
            train_bank, scores + weight * train_feature, empty_train
        )
        deltas.append(correct(train_labels, selected) - base)
    proposed = choices(
        validation_bank, validation_z + weight * validation_feature, empty_validation
    )
    print(
        f"weight={weight:+.3f} support={int((np.ptp(train_feature, axis=1) > 0).sum())}/"
        f"{int((np.ptp(validation_feature, axis=1) > 0).sum())} "
        f"oof_deltas={deltas} "
        f"validation_delta={correct(validation_labels, proposed) - base_validation:+d}"
    )


if __name__ == "__main__":
    main()
