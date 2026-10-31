"""Check the upper weight range for Model Z's cumulative dependency signal."""

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
from semantic_answer_rank_probe import PATTERNS, conditioned_rank
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
        validation_bank, validation_arrays, CODE / "runs/model_z_robust_semantic"
    )
    train_rank = conditioned_rank(train_bank, PATTERNS["remainder_difference"])
    validation_rank = conditioned_rank(validation_bank, PATTERNS["remainder_difference"])
    train_dependency = standardize_rows(
        dependency_features(train_bank, TARGETS["now_everything"])[:, :, 0]
    )
    validation_dependency = standardize_rows(
        dependency_features(validation_bank, TARGETS["now_everything"])[:, :, 0]
    )
    train_y = [scores - 0.025 * train_rank for scores in train_scores]
    validation_y = validation_scores - 0.025 * validation_rank
    empty_train = [set() for _ in train_bank]
    empty_validation = [set() for _ in validation_bank]
    base_train = [
        correct(train_labels, choices(train_bank, scores, empty_train))
        for scores in train_y
    ]
    base_validation_choices = choices(validation_bank, validation_y, empty_validation)
    base_validation = correct(validation_labels, base_validation_choices)
    for weight in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60):
        deltas = []
        for scores, base in zip(train_y, base_train):
            selected = choices(
                train_bank, scores + weight * train_dependency, empty_train
            )
            deltas.append(correct(train_labels, selected) - base)
        proposed = choices(
            validation_bank,
            validation_y + weight * validation_dependency,
            empty_validation,
        )
        answer_changes = []
        for row_index in np.flatnonzero(proposed != base_validation_choices):
            before = parse_answer(
                validation_bank[row_index]["candidates"][base_validation_choices[row_index]]["text"]
            )
            after = parse_answer(
                validation_bank[row_index]["candidates"][proposed[row_index]]["text"]
            )
            if before != after:
                answer_changes.append((int(row_index), before, after))
        print(
            f"weight={weight:.2f} oof_deltas={deltas} "
            f"validation_delta={correct(validation_labels, proposed) - base_validation:+d} "
            f"answer_changes={answer_changes}",
            flush=True,
        )


if __name__ == "__main__":
    main()
