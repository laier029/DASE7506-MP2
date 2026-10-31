"""Probe extra dependency weight for narrowly explicit cumulative targets."""

from pathlib import Path
import re
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


PATTERN_SETS = {
    "total_number_now": re.compile(r"\btotal\s+number\b.*\b(?:now|currently)\b", re.I),
    "total_before": re.compile(r"\btotal\b.*\bbefore\b", re.I),
    "total_number": re.compile(r"\btotal\s+number\b", re.I),
}


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
    train_general = standardize_rows(
        dependency_features(train_bank, TARGETS["now_everything"])[:, :, 0]
    )
    validation_general = standardize_rows(
        dependency_features(validation_bank, TARGETS["now_everything"])[:, :, 0]
    )
    train_z = [scores - 0.025 * train_rank + 0.40 * train_general for scores in train_scores]
    validation_z = validation_scores - 0.025 * validation_rank + 0.40 * validation_general
    empty_train = [set() for _ in train_bank]
    empty_validation = [set() for _ in validation_bank]
    base_train_choices = [choices(train_bank, scores, empty_train) for scores in train_z]
    base_train = [correct(train_labels, selected) for selected in base_train_choices]
    base_validation_choices = choices(validation_bank, validation_z, empty_validation)
    base_validation = correct(validation_labels, base_validation_choices)
    weights = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
    for name, pattern in PATTERN_SETS.items():
        train_feature = standardize_rows(
            dependency_features(train_bank, pattern)[:, :, 0]
        )
        validation_feature = standardize_rows(
            dependency_features(validation_bank, pattern)[:, :, 0]
        )
        trials = []
        for weight in weights:
            selected = choices(
                train_bank, train_z[0] + weight * train_feature, empty_train
            )
            trials.append((correct(train_labels, selected), -weight, weight))
        _, _, weight = max(trials)
        deltas = []
        for scores, base in zip(train_z, base_train):
            selected = choices(
                train_bank, scores + weight * train_feature, empty_train
            )
            deltas.append(correct(train_labels, selected) - base)
        proposed = choices(
            validation_bank,
            validation_z + weight * validation_feature,
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
            f"pattern={name} weight={weight:+.2f} "
            f"support={int((np.ptp(train_feature, axis=1) > 0).sum())}/"
            f"{int((np.ptp(validation_feature, axis=1) > 0).sum())} "
            f"oof_deltas={deltas} "
            f"validation_delta={correct(validation_labels, proposed) - base_validation:+d} "
            f"answer_changes={answer_changes}",
            flush=True,
        )


if __name__ == "__main__":
    main()
