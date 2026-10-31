"""Probe switches supported by extra occurrences of an omitted quantity."""

from collections import Counter
from fractions import Fraction
from pathlib import Path
import re
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from calculation_coverage import (
    calculation_context,
    calculation_quantities,
    quantities,
)
from cascade_verifier import unanimous_broad_error_groups
from common import load_inputs
from coverage_cascade_verifier import coverage_adjusted_scores
from group_ranker_probe import candidate_labels
from model_t_probe_utils import (
    FastFinalSelector,
    load_or_build_model_t_oof,
)
from numeric_plausibility import implausible_answer_groups
from occurrence_plausibility_verifier import occurrence_adjusted_scores
from omitted_factor import _answer_groups, _fraction_set
from structure_text_fusion_verifier import structure_text_scores
from target_omitted_factor_verifier import target_adjusted_scores


ADDITIVE_TARGET = re.compile(
    r"\b(?:total|everything|now|combined|altogether|in all)\b", re.I
)


def row_state(bank):
    arithmetic = unanimous_broad_error_groups(bank)
    implausible = implausible_answer_groups(bank)
    output = []
    for row_index, row in enumerate(bank):
        groups = _answer_groups(row)
        rejected = arithmetic[row_index] | implausible[row_index]
        survivors = {
            answer: indices
            for answer, indices in groups.items()
            if answer not in rejected
        }
        if not survivors:
            survivors = groups
        question_values = _fraction_set(
            quantities(row["question"], include_words=True)
        )
        question_values = {value for value in question_values if abs(value) > 1}
        counters = []
        for candidate in row["candidates"]:
            counter = Counter()
            for value in calculation_quantities(
                calculation_context(candidate["text"]), include_words=True
            ):
                try:
                    counter[Fraction(value)] += 1
                except (ValueError, ZeroDivisionError):
                    continue
            counters.append(counter)
        output.append(
            (
                survivors,
                question_values,
                counters,
                bool(ADDITIVE_TARGET.search(row["question"])),
            )
        )
    return output


def group_count(counters, indices, value):
    return max(counters[index][value] for index in indices)


def select_with_occurrence_switch(
    scores,
    base_choices,
    states,
    minimum_group,
    margin,
    operation,
):
    output = base_choices.copy()
    changes = []
    for row_index, (row_scores, choice, state) in enumerate(
        zip(scores, base_choices, states)
    ):
        survivors, question_values, counters, additive_target = state
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
        selected = next(item for item in scored if choice in item[2])
        if len(selected[2]) < minimum_group:
            continue
        selected_value = Fraction(selected[1])
        alternatives = []
        for item in scored:
            if item[1] == selected[1]:
                continue
            alternative_value = Fraction(item[1])
            evidences = []
            if operation in {"ratio", "both"} and selected_value and alternative_value:
                ratio = abs(alternative_value / selected_value)
                if ratio < 1:
                    ratio = 1 / ratio
                evidences.append(ratio)
            if operation in {"difference", "both", "target_difference"}:
                if operation == "target_difference" and (
                    not additive_target or alternative_value <= selected_value
                ):
                    continue
                evidences.append(abs(alternative_value - selected_value))
            supported = False
            for evidence in evidences:
                if evidence not in question_values:
                    continue
                if group_count(counters, item[2], evidence) <= group_count(
                    counters, selected[2], evidence
                ):
                    continue
                supported = True
                break
            if not supported or selected[0] - item[0] > margin:
                continue
            alternatives.append(item)
        if alternatives:
            replacement = max(alternatives)
            output[row_index] = replacement[3]
            changes.append(row_index)
    return output, changes


def correct(labels, choices):
    return int(labels[np.arange(len(labels)), choices].sum())


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
    train_states = row_state(train_bank)
    validation_states = row_state(validation_bank)
    validation_scores = structure_text_scores(
        validation_bank,
        validation_arrays,
        CODE / "runs/model_t_final_cascade",
    )
    validation_scores = coverage_adjusted_scores(validation_bank, validation_scores)
    validation_scores = occurrence_adjusted_scores(validation_bank, validation_scores)
    validation_scores = target_adjusted_scores(validation_bank, validation_scores)
    train_base = [train_selector.select(scores) for scores in score_sets]
    validation_base = validation_selector.select(validation_scores)

    margins = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
    for operation in ("ratio", "difference", "target_difference", "both"):
        for minimum_group in (1, 2):
            trials = []
            for margin in margins:
                choices, changes = select_with_occurrence_switch(
                    score_sets[0],
                    train_base[0],
                    train_states,
                    minimum_group,
                    margin,
                    operation,
                )
                trials.append(
                    (
                        correct(train_labels, choices),
                        -len(changes),
                        -margin,
                        margin,
                    )
                )
            _, _, _, margin = max(trials)
            train_results = []
            for scores, base in zip(score_sets, train_base):
                choices, changes = select_with_occurrence_switch(
                    scores,
                    base,
                    train_states,
                    minimum_group,
                    margin,
                    operation,
                )
                train_results.append(
                    (correct(train_labels, choices) - correct(train_labels, base), len(changes))
                )
            validation_choices, validation_changes = select_with_occurrence_switch(
                validation_scores,
                validation_base,
                validation_states,
                minimum_group,
                margin,
                operation,
            )
            validation_delta = correct(
                validation_labels, validation_choices
            ) - correct(validation_labels, validation_base)
            print(
                f"operation={operation} minimum_group={minimum_group} "
                f"margin={margin:.2f} train={train_results} "
                f"validation_delta={validation_delta:+d} "
                f"validation_rows={validation_changes}",
                flush=True,
            )


if __name__ == "__main__":
    main()
