"""Probe target-aware switches from a remaining amount to its complement."""

from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import re
import sys

import joblib
import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from calculation_coverage import quantities
from cascade_stability_probe import (
    answer_structures,
    balanced_random_folds,
    model_m_oof,
)
from common import load_inputs, parse_answer
from coverage_cascade_verifier import coverage_adjusted_scores
from group_ranker_probe import candidate_labels
from numeric_filter_stability_probe import apply_model_o_coverage
from numeric_plausibility import fractional_answer_groups
from occurrence_plausibility_verifier import occurrence_adjusted_scores
from omitted_factor import select_with_omitted_factor_switch
from structure_text_fusion_verifier import structure_text_scores


TARGET = re.compile(r"(?i)\b(?:difference|reduction|change|saving|saved)\b")


def answer_groups(row):
    groups = defaultdict(list)
    for index, candidate in enumerate(row["candidates"]):
        answer = parse_answer(candidate["text"])
        if answer is not None:
            groups[answer].append(index)
    return groups


def apply_switch(bank, labels, base_choices):
    delta = positive = negative = neutral = switches = 0
    rows = []
    output = list(base_choices)
    for row_index, (row, choice) in enumerate(zip(bank, base_choices)):
        if TARGET.search(row["question"]) is None:
            continue
        selected_answer = parse_answer(row["candidates"][choice]["text"])
        if selected_answer is None:
            continue
        selected_value = Fraction(selected_answer)
        if selected_value <= 0:
            continue
        question_values = set()
        for value in quantities(row["question"], include_words=True):
            try:
                question_values.add(Fraction(value))
            except (ValueError, ZeroDivisionError):
                continue
        alternatives = []
        for answer, indices in answer_groups(row).items():
            value = Fraction(answer)
            if value < 0 or value >= selected_value:
                continue
            if selected_value + value not in question_values:
                continue
            alternatives.append(indices[0])
        if not alternatives:
            continue
        alternative = alternatives[0]
        change = int(labels[row_index, alternative]) - int(
            labels[row_index, choice]
        )
        output[row_index] = alternative
        switches += 1
        delta += change
        positive += int(change > 0)
        negative += int(change < 0)
        neutral += int(change == 0)
        rows.append((row_index, change))
    return output, (delta, switches, positive, negative, neutral), rows


def train_occurrence_adjustment(bank, structures, scores):
    from calculation_coverage import calculation_occurrence_coverage

    feature = calculation_occurrence_coverage(bank)
    feature = (feature - feature.mean(axis=1, keepdims=True)) / (
        feature.std(axis=1, keepdims=True) + 1e-6
    )
    rows = structures == "1-1-1-1"
    adjusted = scores.copy()
    adjusted[rows] += 0.225 * feature[rows]
    return adjusted


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    train_structures = answer_structures(train_bank)
    artifact = joblib.load(
        CODE / "runs/model_r_omitted_factor/structure_text_model.joblib"
    )
    texts = [
        candidate["text"]
        for row in train_bank
        for candidate in row["candidates"]
    ]
    text_features = artifact["vectorizer"].transform(texts)

    train_results = []
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
        scores = train_occurrence_adjustment(
            train_bank, train_structures, scores
        )
        choices = select_with_omitted_factor_switch(
            train_bank,
            scores,
            0.3,
            fractional_answer_groups(train_bank),
        )
        _, result, _ = apply_switch(train_bank, train_labels, choices)
        train_results.append(result)

    validation_scores = structure_text_scores(
        validation_bank,
        validation_arrays,
        CODE / "runs/model_r_omitted_factor",
    )
    validation_scores = coverage_adjusted_scores(
        validation_bank, validation_scores
    )
    validation_scores = occurrence_adjusted_scores(
        validation_bank, validation_scores
    )
    validation_choices = select_with_omitted_factor_switch(
        validation_bank,
        validation_scores,
        0.3,
        fractional_answer_groups(validation_bank),
    )
    _, validation_result, rows = apply_switch(
        validation_bank, validation_labels, validation_choices
    )
    print(
        f"train={train_results} validation={validation_result} rows={rows}",
        flush=True,
    )


if __name__ == "__main__":
    main()
