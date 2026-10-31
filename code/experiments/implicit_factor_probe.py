"""Probe coverage of implicit conversion and multiplicity factors."""

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
from coverage_cascade_verifier import coverage_adjusted_scores
from group_ranker_probe import candidate_labels
from model_t_probe_utils import (
    FastFinalSelector,
    load_or_build_model_t_oof,
    standardize_rows,
)
from occurrence_plausibility_verifier import occurrence_adjusted_scores
from stable_signal_features import implicit_factor_coverage
from structure_text_fusion_verifier import structure_text_scores
from target_omitted_factor_verifier import target_adjusted_scores


PATTERNS = (
    ("both", re.compile(r"\bboth\b", re.I), "2"),
    ("parents", re.compile(r"\bparents\b", re.I), "2"),
    ("round_trip", re.compile(r"\bround trips?\b", re.I), "2"),
    ("second_coat", re.compile(r"\bsecond coat\b", re.I), "2"),
    ("daily_week", re.compile(r"\b(?:daily|per day|each day)\b.*\bweek\b", re.I), "7"),
    ("monthly_year", re.compile(r"\b(?:monthly|per month|each month)\b.*\byear\b", re.I), "12"),
    ("hours_minutes", re.compile(r"\bhours?\b.*\bminutes?\b", re.I), "60"),
    ("minutes_seconds", re.compile(r"\bminutes?\b.*\bseconds?\b", re.I), "60"),
)


def implicit_features(bank):
    output = np.zeros((len(bank), 4, len(PATTERNS) + 1), dtype=np.float32)
    for row_index, row in enumerate(bank):
        active = [
            (feature_index, value)
            for feature_index, (_, pattern, value) in enumerate(PATTERNS)
            if pattern.search(row["question"])
        ]
        if not active:
            continue
        required = Counter(value for _, value in active)
        for candidate_index, candidate in enumerate(row["candidates"]):
            used = Counter(
                calculation_quantities(
                    calculation_context(candidate["text"]), include_words=True
                )
            )
            for feature_index, value in active:
                output[row_index, candidate_index, feature_index] = float(
                    used[value] > 0
                )
            output[row_index, candidate_index, -1] = sum(
                min(count, used[value]) for value, count in required.items()
            ) / sum(required.values())
    output[:, :, -1] = implicit_factor_coverage(bank)
    return output


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    score_sets = load_or_build_model_t_oof(
        train_bank, train_arrays, train_labels
    )
    train_selector = FastFinalSelector(train_bank)
    validation_selector = FastFinalSelector(validation_bank)
    validation_scores = structure_text_scores(
        validation_bank,
        validation_arrays,
        CODE / "runs/model_t_final_cascade",
    )
    validation_scores = coverage_adjusted_scores(
        validation_bank, validation_scores
    )
    validation_scores = occurrence_adjusted_scores(
        validation_bank, validation_scores
    )
    validation_scores = target_adjusted_scores(
        validation_bank, validation_scores
    )
    train_base = [
        train_selector.correct_count(train_labels, scores) for scores in score_sets
    ]
    validation_base = validation_selector.correct_count(
        validation_labels, validation_scores
    )
    train_features = implicit_features(train_bank)
    validation_features = implicit_features(validation_bank)
    names = tuple(name for name, _, _ in PATTERNS) + ("combined",)
    weights = np.round(np.arange(-0.5, 0.501, 0.025), 3)
    for feature_index, name in enumerate(names):
        train_feature = standardize_rows(train_features[:, :, feature_index])
        validation_feature = standardize_rows(
            validation_features[:, :, feature_index]
        )
        trials = []
        for weight in weights:
            count = train_selector.correct_count(
                train_labels, score_sets[0] + weight * train_feature
            )
            trials.append((count, -abs(float(weight)), float(weight)))
        _, _, weight = max(trials)
        train_deltas = [
            train_selector.correct_count(
                train_labels, scores + weight * train_feature
            )
            - base
            for scores, base in zip(score_sets, train_base)
        ]
        validation_delta = validation_selector.correct_count(
            validation_labels, validation_scores + weight * validation_feature
        ) - validation_base
        support = int(
            (np.ptp(train_features[:, :, feature_index], axis=1) > 0).sum()
        )
        print(
            f"feature={name} support={support} weight={weight:+.3f} "
            f"train_deltas={train_deltas} validation_delta={validation_delta:+d}",
            flush=True,
        )


if __name__ == "__main__":
    main()
