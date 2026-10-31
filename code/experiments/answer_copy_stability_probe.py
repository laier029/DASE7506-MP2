"""Probe a soft copied-question-answer penalty on Model T all-distinct rows."""

from fractions import Fraction
from pathlib import Path
import sys

import joblib
import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from answer_copy_probe import copy_groups
from calculation_coverage import calculation_occurrence_coverage, quantities
from cascade_stability_probe import (
    answer_structures,
    balanced_random_folds,
    model_m_oof,
)
from cascade_verifier import unanimous_broad_error_groups
from common import load_inputs, parse_answer
from coverage_cascade_verifier import coverage_adjusted_scores
from group_ranker_probe import candidate_labels
from numeric_filter_stability_probe import apply_model_o_coverage
from numeric_plausibility import implausible_answer_groups
from occurrence_plausibility_verifier import occurrence_adjusted_scores
from structure_text_fusion_verifier import structure_text_scores
from target_alignment import terminal_target_recall
from target_omitted_factor_verifier import target_adjusted_scores


def standardize_rows(values):
    return (values - values.mean(axis=1, keepdims=True)) / (
        values.std(axis=1, keepdims=True) + 1e-6
    )


def copy_feature(bank):
    copied = copy_groups(bank)
    output = np.zeros((len(bank), 4), dtype=np.float32)
    for row_index, row in enumerate(bank):
        for candidate_index, candidate in enumerate(row["candidates"]):
            output[row_index, candidate_index] = float(
                parse_answer(candidate["text"]) in copied[row_index]
            )
    return output


def allowed_candidates(bank):
    arithmetic = unanimous_broad_error_groups(bank)
    implausible = implausible_answer_groups(bank)
    allowed = np.zeros((len(bank), 4), dtype=bool)
    for row_index, row in enumerate(bank):
        rejected = arithmetic[row_index] | implausible[row_index]
        for candidate_index, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate["text"])
            allowed[row_index, candidate_index] = (
                answer is not None and answer not in rejected
            )
        if not allowed[row_index].any():
            allowed[row_index] = [
                parse_answer(candidate["text"]) is not None
                for candidate in row["candidates"]
            ]
    return allowed


def correct_count(labels, scores, rows, allowed):
    values = scores[rows].copy()
    values[~allowed[rows]] = -np.inf
    choices = values.argmax(axis=1)
    return int(labels[rows][np.arange(rows.sum()), choices].sum())


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    train_structures = answer_structures(train_bank)
    validation_structures = answer_structures(validation_bank)
    train_rows = train_structures == "1-1-1-1"
    validation_rows = validation_structures == "1-1-1-1"
    train_allowed = allowed_candidates(train_bank)
    validation_allowed = allowed_candidates(validation_bank)
    artifact = joblib.load(
        CODE / "runs/model_t_final_cascade/structure_text_model.joblib"
    )
    texts = [
        candidate["text"]
        for row in train_bank
        for candidate in row["candidates"]
    ]
    text_features = artifact["vectorizer"].transform(texts)
    occurrence = standardize_rows(calculation_occurrence_coverage(train_bank))
    target = standardize_rows(terminal_target_recall(train_bank))
    score_sets = []
    for seed in (7506, 11, 29, 47, 83):
        folds = (
            np.arange(len(train_bank)) % 5
            if seed == 7506
            else balanced_random_folds(len(train_bank), seed)
        )
        scores = model_m_oof(
            train_bank,
            train_arrays,
            train_labels,
            folds,
            text_features,
            train_structures,
        )
        scores = apply_model_o_coverage(train_bank, scores, train_structures)
        scores[train_rows] += 0.225 * occurrence[train_rows]
        scores[train_rows] += 0.125 * target[train_rows]
        score_sets.append(scores)

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
        correct_count(train_labels, scores, train_rows, train_allowed)
        for scores in score_sets
    ]
    validation_base = correct_count(
        validation_labels,
        validation_scores,
        validation_rows,
        validation_allowed,
    )
    train_feature = standardize_rows(copy_feature(train_bank))
    validation_feature = standardize_rows(copy_feature(validation_bank))
    weights = np.round(np.arange(-0.5, 0.001, 0.025), 3)
    trials = []
    for weight in weights:
        count = correct_count(
            train_labels,
            score_sets[0] + weight * train_feature,
            train_rows,
            train_allowed,
        )
        trials.append((count, -abs(float(weight)), float(weight)))
    _, _, weight = max(trials)
    train_deltas = [
        correct_count(
            train_labels,
            scores + weight * train_feature,
            train_rows,
            train_allowed,
        )
        - base
        for scores, base in zip(score_sets, train_base)
    ]
    validation_delta = correct_count(
        validation_labels,
        validation_scores + weight * validation_feature,
        validation_rows,
        validation_allowed,
    ) - validation_base
    print(
        f"weight={weight:+.3f} train_deltas={train_deltas} "
        f"validation_delta={validation_delta:+d}",
        flush=True,
    )


if __name__ == "__main__":
    main()
