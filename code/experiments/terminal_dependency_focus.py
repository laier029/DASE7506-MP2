"""Focused weight and change audit for terminal dependency signals."""

from pathlib import Path
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from common import load_inputs, parse_answer
from dimension_consistency_probe import choices, correct
from group_ranker_probe import candidate_labels
from model_t_probe_utils import load_or_build_model_t_oof, standardize_rows
from stable_signal_features import stable_signal_adjusted_scores
from stable_signal_verifier import stable_signal_scores
from terminal_dependency_probe import TARGETS, dependency_features


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    train_scores = [
        stable_signal_adjusted_scores(train_bank, scores)
        for scores in load_or_build_model_t_oof(train_bank, train_arrays, train_labels)
    ]
    validation_scores = stable_signal_scores(
        validation_bank, validation_arrays, CODE / "runs/model_x_full_cascade"
    )
    empty_train = [set() for _ in train_bank]
    empty_validation = [set() for _ in validation_bank]
    base_train_choices = [choices(train_bank, scores, empty_train) for scores in train_scores]
    base_train = [correct(train_labels, selected) for selected in base_train_choices]
    base_validation_choices = choices(
        validation_bank, validation_scores, empty_validation
    )
    base_validation = correct(validation_labels, base_validation_choices)
    train_raw = dependency_features(train_bank, TARGETS["now_everything"])
    validation_raw = dependency_features(
        validation_bank, TARGETS["now_everything"]
    )
    for feature_index, feature_name in enumerate(("leaf_count", "depth")):
        train_feature = standardize_rows(train_raw[:, :, feature_index])
        validation_feature = standardize_rows(validation_raw[:, :, feature_index])
        for weight in (0.05, 0.10, 0.15, 0.20):
            deltas = []
            for scores, base in zip(train_scores, base_train):
                selected = choices(
                    train_bank, scores + weight * train_feature, empty_train
                )
                deltas.append(correct(train_labels, selected) - base)
            proposed = choices(
                validation_bank,
                validation_scores + weight * validation_feature,
                empty_validation,
            )
            changed = np.flatnonzero(proposed != base_validation_choices).tolist()
            changes = []
            for row_index in changed:
                before = bool(
                    validation_labels[row_index, base_validation_choices[row_index]]
                )
                after = bool(validation_labels[row_index, proposed[row_index]])
                changes.append(
                    (
                        row_index,
                        int(after) - int(before),
                        parse_answer(
                            validation_bank[row_index]["candidates"][
                                base_validation_choices[row_index]
                            ]["text"]
                        ),
                        parse_answer(
                            validation_bank[row_index]["candidates"][
                                proposed[row_index]
                            ]["text"]
                        ),
                    )
                )
            print(
                f"feature={feature_name} weight={weight:+.2f} "
                f"oof_deltas={deltas} "
                f"validation_delta={correct(validation_labels, proposed) - base_validation:+d} "
                f"changes={changes}",
                flush=True,
            )


if __name__ == "__main__":
    main()
