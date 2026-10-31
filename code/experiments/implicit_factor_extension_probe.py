"""Screen additional language-implied factors across five OOF schemes."""

from collections import Counter
from pathlib import Path
import re
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from calculation_coverage import calculation_context, calculation_quantities
from common import load_inputs
from group_ranker_probe import candidate_labels
from model_t_probe_utils import load_or_build_model_t_oof, standardize_rows
from stable_signal_features import stable_signal_adjusted_scores
from stable_signal_verifier import stable_signal_scores
from dimension_consistency_probe import choices, correct


PATTERN_SETS = {
    "weeknight_weekend": (
        (re.compile(r"\bweeknights?\b", re.I), "5"),
        (re.compile(r"\bweekends?\b", re.I), "2"),
    ),
    "weekday_weekend": (
        (re.compile(r"\bweekdays?\b", re.I), "5"),
        (re.compile(r"\bweekends?\b", re.I), "2"),
    ),
    "dozen": ((re.compile(r"\bdozen\b", re.I), "12"),),
    "fortnight": ((re.compile(r"\bfortnights?\b", re.I), "14"),),
}


def factor_coverage(bank, patterns):
    output = np.zeros((len(bank), 4), dtype=np.float32)
    for row_index, row in enumerate(bank):
        required = Counter(
            value for pattern, value in patterns if pattern.search(row["question"])
        )
        if not required:
            continue
        for candidate_index, candidate in enumerate(row["candidates"]):
            used = Counter(
                calculation_quantities(
                    calculation_context(candidate["text"]), include_words=True
                )
            )
            output[row_index, candidate_index] = sum(
                min(count, used[value]) for value, count in required.items()
            ) / sum(required.values())
    return output


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    train_scores = [
        stable_signal_adjusted_scores(train_bank, scores)
        for scores in load_or_build_model_t_oof(
            train_bank, train_arrays, train_labels
        )
    ]
    validation_scores = stable_signal_scores(
        validation_bank,
        validation_arrays,
        CODE / "runs/model_x_full_cascade",
    )
    empty_train = [set() for _ in train_bank]
    empty_validation = [set() for _ in validation_bank]
    base_train = [
        correct(train_labels, choices(train_bank, scores, empty_train))
        for scores in train_scores
    ]
    base_validation = correct(
        validation_labels,
        choices(validation_bank, validation_scores, empty_validation),
    )
    weights = np.round(np.arange(-0.75, 0.751, 0.05), 3)
    for name, patterns in PATTERN_SETS.items():
        train_raw = factor_coverage(train_bank, patterns)
        validation_raw = factor_coverage(validation_bank, patterns)
        train_feature = standardize_rows(train_raw)
        validation_feature = standardize_rows(validation_raw)
        trials = []
        for weight in weights:
            selected = choices(
                train_bank,
                train_scores[0] + weight * train_feature,
                empty_train,
            )
            trials.append(
                (correct(train_labels, selected), -abs(float(weight)), float(weight))
            )
        _, _, weight = max(trials)
        deltas = [
            correct(
                train_labels,
                choices(
                    train_bank,
                    scores + weight * train_feature,
                    empty_train,
                ),
            )
            - base
            for scores, base in zip(train_scores, base_train)
        ]
        proposed = choices(
            validation_bank,
            validation_scores + weight * validation_feature,
            empty_validation,
        )
        support = int((np.ptp(train_raw, axis=1) > 0).sum())
        validation_support = int((np.ptp(validation_raw, axis=1) > 0).sum())
        print(
            f"feature={name} support={support}/{validation_support} "
            f"weight={weight:+.2f} oof_deltas={deltas} "
            f"validation_delta={correct(validation_labels, proposed) - base_validation:+d}",
            flush=True,
        )


if __name__ == "__main__":
    main()
