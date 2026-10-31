"""Focused low-weight stability check for two promising rank priors."""

from pathlib import Path
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from common import load_inputs
from dimension_consistency_probe import choices, correct
from group_ranker_probe import candidate_labels
from model_t_probe_utils import load_or_build_model_t_oof
from semantic_answer_rank_probe import PATTERNS, conditioned_rank
from stable_signal_features import stable_signal_adjusted_scores
from stable_signal_verifier import stable_signal_scores


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
    train_base_choices = [choices(train_bank, scores, empty_train) for scores in train_scores]
    train_base = [correct(train_labels, selected) for selected in train_base_choices]
    validation_base_choices = choices(
        validation_bank, validation_scores, empty_validation
    )
    validation_base = correct(validation_labels, validation_base_choices)

    settings = {
        "current_now": (0.025, 0.05, 0.075, 0.10, 0.125, 0.15),
        "remainder_difference": (-0.025, -0.05, -0.075, -0.10),
    }
    for name, weights in settings.items():
        train_feature = conditioned_rank(train_bank, PATTERNS[name])
        validation_feature = conditioned_rank(validation_bank, PATTERNS[name])
        for weight in weights:
            deltas = []
            changes = []
            for scores, base, base_choices in zip(
                train_scores, train_base, train_base_choices
            ):
                selected = choices(
                    train_bank, scores + weight * train_feature, empty_train
                )
                deltas.append(correct(train_labels, selected) - base)
                changes.append(int(np.sum(selected != base_choices)))
            proposed = choices(
                validation_bank,
                validation_scores + weight * validation_feature,
                empty_validation,
            )
            print(
                f"feature={name} weight={weight:+.3f} "
                f"oof_deltas={deltas} oof_changes={changes} "
                f"validation_delta={correct(validation_labels, proposed) - validation_base:+d} "
                f"validation_changes={np.flatnonzero(proposed != validation_base_choices).tolist()}",
                flush=True,
            )


if __name__ == "__main__":
    main()
