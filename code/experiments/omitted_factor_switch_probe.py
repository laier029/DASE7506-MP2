"""Probe switches when answer ratios expose an omitted question multiplier."""

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

from calculation_coverage import (
    calculation_context,
    calculation_occurrence_coverage,
    calculation_quantities,
    quantities,
)
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
from numeric_plausibility import fractional_answer_groups
from occurrence_plausibility_verifier import occurrence_adjusted_scores
from structure_text_fusion_verifier import structure_text_scores


STRUCTURAL_NUMBER = re.compile(
    r"(?im)^\s*(?:(?:#{1,6}\s*)?(?:step|stage|part)\s+\d+\s*:?|"
    r"\d+[.)])\s*"
)


def without_structural_numbers(text):
    return STRUCTURAL_NUMBER.sub("", text)


def answer_groups(row):
    groups = defaultdict(list)
    for index, candidate in enumerate(row["candidates"]):
        answer = parse_answer(candidate["text"])
        if answer is not None:
            groups[answer].append(index)
    return groups


def fraction_set(values):
    output = set()
    for value in values:
        try:
            output.add(Fraction(value))
        except (ValueError, ZeroDivisionError):
            continue
    return output


def quantity_sets(bank):
    question_sets = []
    candidate_sets = []
    for row in bank:
        question_sets.append(
            fraction_set(quantities(row["question"], include_words=True))
        )
        candidate_sets.append(
            [
                fraction_set(
                    calculation_quantities(
                        calculation_context(
                            without_structural_numbers(candidate["text"])
                        ),
                        include_words=True,
                    )
                )
                for candidate in row["candidates"]
            ]
        )
    return question_sets, candidate_sets


def evaluate(
    bank,
    labels,
    scores,
    question_sets,
    candidate_sets,
    minimum_group,
    margin_threshold=float("inf"),
    operation="ratio",
):
    arithmetic = unanimous_broad_error_groups(bank)
    fractional = fractional_answer_groups(bank)
    delta = switches = positive = negative = neutral = 0
    changed_rows = []
    for row_index, (row, row_scores) in enumerate(zip(bank, scores)):
        groups = answer_groups(row)
        rejected = arithmetic[row_index] | fractional[row_index]
        survivors = {
            answer: indices
            for answer, indices in groups.items()
            if answer not in rejected
        }
        if not survivors:
            survivors = groups
        scored = []
        for answer, indices in survivors.items():
            values = row_scores[indices]
            candidate = indices[int(values.argmax())]
            scored.append(
                (
                    float(values.max() + 0.3 * values.mean()),
                    answer,
                    indices,
                    candidate,
                )
            )
        scored.sort(reverse=True)
        selected = scored[0]
        if len(selected[2]) < minimum_group:
            continue
        selected_value = Fraction(selected[1])
        if selected_value == 0:
            continue
        selected_used = set().union(
            *(candidate_sets[row_index][index] for index in selected[2])
        )
        alternatives = []
        for item in scored[1:]:
            alternative_value = Fraction(item[1])
            if alternative_value == 0:
                continue
            if operation == "ratio":
                evidence = abs(alternative_value / selected_value)
                if evidence < 1:
                    evidence = 1 / evidence
            elif operation == "difference":
                evidence = abs(alternative_value - selected_value)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            if evidence <= 0 or evidence not in question_sets[row_index]:
                continue
            if evidence in selected_used:
                continue
            alternative_used = set().union(
                *(candidate_sets[row_index][index] for index in item[2])
            )
            if evidence not in alternative_used:
                continue
            alternatives.append(item)
        if not alternatives:
            continue
        alternative = max(alternatives)
        if selected[0] - alternative[0] > margin_threshold:
            continue
        switches += 1
        change = int(labels[row_index, alternative[3]]) - int(
            labels[row_index, selected[3]]
        )
        delta += change
        positive += int(change > 0)
        negative += int(change < 0)
        neutral += int(change == 0)
        changed_rows.append((row_index, change))
    return delta, switches, positive, negative, neutral, changed_rows


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    train_structures = answer_structures(train_bank)
    artifact = joblib.load(
        CODE / "runs/model_q_occurrence_plausibility/structure_text_model.joblib"
    )
    texts = [
        candidate["text"]
        for row in train_bank
        for candidate in row["candidates"]
    ]
    text_features = artifact["vectorizer"].transform(texts)
    train_question_sets, train_candidate_sets = quantity_sets(train_bank)
    val_question_sets, val_candidate_sets = quantity_sets(validation_bank)
    occurrence = calculation_occurrence_coverage(train_bank)
    occurrence = (occurrence - occurrence.mean(axis=1, keepdims=True)) / (
        occurrence.std(axis=1, keepdims=True) + 1e-6
    )

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
        scores = apply_model_o_coverage(
            train_bank, scores, train_structures
        )
        # Apply the occurrence feature to mirror Model Q for 1-1-1-1 rows.
        rows = train_structures == "1-1-1-1"
        scores[rows] += 0.225 * occurrence[rows]
        score_sets.append(scores)

    validation_scores = structure_text_scores(
        validation_bank,
        validation_arrays,
        CODE / "runs/model_q_occurrence_plausibility",
    )
    validation_scores = coverage_adjusted_scores(
        validation_bank, validation_scores
    )
    validation_scores = occurrence_adjusted_scores(
        validation_bank, validation_scores
    )
    for minimum_group in (1, 2, 3):
        train_results = [
            evaluate(
                train_bank,
                train_labels,
                scores,
                train_question_sets,
                train_candidate_sets,
                minimum_group,
            )[:5]
            for scores in score_sets
        ]
        validation_result = evaluate(
            validation_bank,
            validation_labels,
            validation_scores,
            val_question_sets,
            val_candidate_sets,
            minimum_group,
        )
        print(
            f"minimum_group={minimum_group} train={train_results} "
            f"validation={validation_result[:5]} "
            f"validation_rows={validation_result[5]}",
            flush=True,
        )

    for minimum_group in (1, 2):
        print(f"minimum_group={minimum_group} margin sweep")
        fixed_trials = []
        for margin_threshold in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
            result = evaluate(
                train_bank,
                train_labels,
                score_sets[0],
                train_question_sets,
                train_candidate_sets,
                minimum_group,
                margin_threshold,
            )
            fixed_trials.append(
                (result[0], -result[1], -margin_threshold, margin_threshold)
            )
        _, _, _, chosen_margin = max(fixed_trials)
        train_results = [
            evaluate(
                train_bank,
                train_labels,
                scores,
                train_question_sets,
                train_candidate_sets,
                minimum_group,
                chosen_margin,
            )[:5]
            for scores in score_sets
        ]
        validation_result = evaluate(
            validation_bank,
            validation_labels,
            validation_scores,
            val_question_sets,
            val_candidate_sets,
            minimum_group,
            chosen_margin,
        )
        print(
            f"minimum_group={minimum_group} chosen_margin={chosen_margin:.2f} "
            f"train={train_results} validation={validation_result[:5]} "
            f"validation_rows={validation_result[5]}",
            flush=True,
        )

    print("difference operation margin sweep")
    fixed_trials = []
    for margin_threshold in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
        result = evaluate(
            train_bank,
            train_labels,
            score_sets[0],
            train_question_sets,
            train_candidate_sets,
            2,
            margin_threshold,
            operation="difference",
        )
        fixed_trials.append(
            (result[0], -result[1], -margin_threshold, margin_threshold)
        )
    _, _, _, difference_margin = max(fixed_trials)
    train_results = [
        evaluate(
            train_bank,
            train_labels,
            scores,
            train_question_sets,
            train_candidate_sets,
            2,
            difference_margin,
            operation="difference",
        )[:5]
        for scores in score_sets
    ]
    validation_result = evaluate(
        validation_bank,
        validation_labels,
        validation_scores,
        val_question_sets,
        val_candidate_sets,
        2,
        difference_margin,
        operation="difference",
    )
    print(
        f"difference_margin={difference_margin:.2f} train={train_results} "
        f"validation={validation_result[:5]} "
        f"validation_rows={validation_result[5]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
