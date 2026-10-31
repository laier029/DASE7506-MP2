"""Probe keyword-conditioned numeric answer-rank priors."""

from fractions import Fraction
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
from stable_signal_features import stable_signal_adjusted_scores
from stable_signal_verifier import stable_signal_scores


PATTERNS = {
    "explicit_total": re.compile(
        r"\b(?:total|altogether|in all|all together|overall)\b", re.I
    ),
    "current_now": re.compile(r"\b(?:currently|now|everything)\b", re.I),
    "entire_period": re.compile(r"\bentire\s+(?:week|month|year)\b", re.I),
    "remainder_difference": re.compile(
        r"\b(?:left|remain(?:s|ing|ed)?|difference|saved|savings|behind)\b",
        re.I,
    ),
}


def conditioned_rank(bank, pattern):
    output = np.zeros((len(bank), 4), dtype=np.float32)
    for row_index, row in enumerate(bank):
        if pattern.search(row["question"]) is None:
            continue
        values = []
        for candidate in row["candidates"]:
            try:
                values.append(float(Fraction(parse_answer(candidate["text"]))))
            except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                values.append(np.nan)
        finite = np.asarray([value for value in values if np.isfinite(value)])
        if len(set(finite.tolist())) <= 1:
            continue
        for candidate_index, value in enumerate(values):
            if np.isfinite(value):
                output[row_index, candidate_index] = float((finite < value).sum()) / max(
                    len(finite) - 1, 1
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
    validation_scores = stable_signal_scores(
        validation_bank, validation_arrays, CODE / "runs/model_x_full_cascade"
    )
    empty_train = [set() for _ in train_bank]
    empty_validation = [set() for _ in validation_bank]
    base_train_choices = [choices(train_bank, scores, empty_train) for scores in train_scores]
    base_train = [
        correct(train_labels, selected) for selected in base_train_choices
    ]
    base_validation_choices = choices(
        validation_bank, validation_scores, empty_validation
    )
    base_validation = correct(validation_labels, base_validation_choices)
    weights = np.round(np.arange(-0.50, 0.501, 0.025), 3)
    for name, pattern in PATTERNS.items():
        train_feature = conditioned_rank(train_bank, pattern)
        validation_feature = conditioned_rank(validation_bank, pattern)
        trials = []
        for weight in weights:
            selected = choices(
                train_bank, train_scores[0] + weight * train_feature, empty_train
            )
            trials.append(
                (correct(train_labels, selected), -abs(float(weight)), float(weight))
            )
        _, _, weight = max(trials)
        deltas = []
        changes = []
        for scores, base, base_selected in zip(
            train_scores, base_train, base_train_choices
        ):
            selected = choices(
                train_bank, scores + weight * train_feature, empty_train
            )
            deltas.append(correct(train_labels, selected) - base)
            changes.append(int(np.sum(selected != base_selected)))
        proposed = choices(
            validation_bank,
            validation_scores + weight * validation_feature,
            empty_validation,
        )
        support = int((np.ptp(train_feature, axis=1) > 0).sum())
        validation_support = int((np.ptp(validation_feature, axis=1) > 0).sum())
        print(
            f"feature={name} support={support}/{validation_support} "
            f"weight={weight:+.3f} oof_deltas={deltas} "
            f"oof_changes={changes} "
            f"validation_delta={correct(validation_labels, proposed) - base_validation:+d} "
            f"validation_changes={np.flatnonzero(proposed != base_validation_choices).tolist()}",
            flush=True,
        )


if __name__ == "__main__":
    main()
