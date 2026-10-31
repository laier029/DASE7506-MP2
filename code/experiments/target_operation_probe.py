"""Probe whether the final arithmetic operation matches the question target."""

from pathlib import Path
import re
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from answer_step_consistency import _numeric_rhs_value, _reasoning_prefix
from arithmetic_consistency import _normalize
from common import load_inputs
from coverage_cascade_verifier import coverage_adjusted_scores
from group_ranker_probe import candidate_labels
from model_t_probe_utils import (
    FastFinalSelector,
    load_or_build_model_t_oof,
)
from occurrence_plausibility_verifier import occurrence_adjusted_scores
from structure_text_fusion_verifier import structure_text_scores
from target_omitted_factor_verifier import target_adjusted_scores


TOTAL_TARGET = re.compile(r"\b(?:total|altogether|in all|combined|overall)\b", re.I)
REMAINDER_TARGET = re.compile(
    r"\b(?:left|remaining|remain|difference|unable|unpaired|saved|saving)\b",
    re.I,
)
RATE_TARGET = re.compile(r"\b(?:each|per|every|times|twice|double)\b", re.I)
TOTAL_NUMBER_TARGET = re.compile(r"\btotal number\b", re.I)
TOTAL_NOW_TARGET = re.compile(r"\btotal\b.*\b(?:now|currently)\b", re.I)
TOTAL_BEFORE_TARGET = re.compile(r"\btotal\b.*\bbefore\b", re.I)
COMBINED_TARGET = re.compile(r"\b(?:combined|in all|altogether)\b", re.I)
EVERYTHING_TARGET = re.compile(r"\beverything\b", re.I)
ENTIRE_YEAR_TARGET = re.compile(r"\bentire\b.*\byear\b", re.I)
ROUND_TRIP_TARGET = re.compile(r"\bround trips?\b", re.I)


def last_result_operation(text):
    """Return operator flags for the last equation with a numeric RHS."""
    normalized = _normalize(_reasoning_prefix(text))
    normalized = normalized.replace("\\Rightarrow", "\n")
    normalized = normalized.replace("\\implies", "\n")
    normalized = normalized.replace("\\\\", "\n")
    normalized = normalized.replace("\\$", "").replace("$", " ")
    output = None
    for line in normalized.splitlines():
        if "=" not in line:
            continue
        parts = line.split("=")
        for index in range(1, len(parts)):
            if _numeric_rhs_value(parts[index]) is None:
                continue
            lhs = parts[index - 1]
            output = (
                float("+" in lhs),
                float(bool(re.search(r"(?<=\d)\s*-\s*(?=\d|\()", lhs))),
                float("*" in lhs),
                float("/" in lhs),
                float(lhs.count("+") >= 2),
            )
    return output or (0.0, 0.0, 0.0, 0.0, 0.0)


def operation_features(bank):
    output = np.zeros((len(bank), 4, 15), dtype=np.float32)
    for row_index, row in enumerate(bank):
        total = bool(TOTAL_TARGET.search(row["question"]))
        remainder = bool(REMAINDER_TARGET.search(row["question"]))
        rate = bool(RATE_TARGET.search(row["question"]))
        total_number = bool(TOTAL_NUMBER_TARGET.search(row["question"]))
        total_now = bool(TOTAL_NOW_TARGET.search(row["question"]))
        total_before = bool(TOTAL_BEFORE_TARGET.search(row["question"]))
        combined = bool(COMBINED_TARGET.search(row["question"]))
        everything = bool(EVERYTHING_TARGET.search(row["question"]))
        entire_year = bool(ENTIRE_YEAR_TARGET.search(row["question"]))
        round_trip = bool(ROUND_TRIP_TARGET.search(row["question"]))
        for candidate_index, candidate in enumerate(row["candidates"]):
            add, subtract, multiply, divide, multi_add = last_result_operation(
                candidate["text"]
            )
            output[row_index, candidate_index] = (
                total * add,
                total * multi_add,
                remainder * subtract,
                rate * multiply,
                add,
                subtract,
                multiply,
                divide,
                total_number * add,
                total_now * add,
                total_before * add,
                combined * add,
                everything * add,
                entire_year * multiply,
                round_trip * add,
            )
    return output


def standardize_rows(values):
    return (values - values.mean(axis=1, keepdims=True)) / (
        values.std(axis=1, keepdims=True) + 1e-6
    )


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
    train_features = operation_features(train_bank)
    validation_features = operation_features(validation_bank)
    names = (
        "total_add",
        "total_multi_add",
        "remainder_subtract",
        "rate_multiply",
        "any_add",
        "any_subtract",
        "any_multiply",
        "any_divide",
        "total_number_add",
        "total_now_add",
        "total_before_add",
        "combined_add",
        "everything_add",
        "entire_year_multiply",
        "round_trip_add",
    )
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
        print(
            f"feature={name} weight={weight:+.3f} "
            f"train_deltas={train_deltas} validation_delta={validation_delta:+d}",
            flush=True,
        )


if __name__ == "__main__":
    main()
