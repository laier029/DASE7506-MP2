"""Probe hard rejection when protocol parsing disagrees with final boxed value."""

from collections import defaultdict
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
from coverage_cascade_verifier import coverage_adjusted_scores
from group_ranker_probe import candidate_labels
from model_t_probe_utils import load_or_build_model_t_oof
from numeric_plausibility import implausible_answer_groups
from occurrence_plausibility_verifier import occurrence_adjusted_scores
from omitted_factor import select_with_omitted_factor_switch
from stable_signal_features import stable_signal_adjusted_scores
from structure_text_fusion_verifier import structure_text_scores
from target_omitted_factor_verifier import target_adjusted_scores


BOXED = re.compile(r"\\boxed\s*\{\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)\s*\}")


def boxed_answer(text):
    matches = BOXED.findall(text)
    if not matches:
        return None
    try:
        return str(Fraction(matches[-1].replace(",", "")))
    except (ValueError, ZeroDivisionError):
        return None


def mismatch_flags(bank):
    return np.asarray(
        [
            [
                boxed_answer(candidate["text"]) is not None
                and parse_answer(candidate["text"]) != boxed_answer(candidate["text"])
                for candidate in row["candidates"]
            ]
            for row in bank
        ],
        dtype=bool,
    )


def rejected_groups(bank, flags):
    output = []
    for row, row_flags in zip(bank, flags):
        groups = defaultdict(list)
        for candidate_index, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate["text"])
            if answer is not None:
                groups[answer].append(candidate_index)
        output.append(
            {
                answer
                for answer, indices in groups.items()
                if all(row_flags[index] for index in indices)
            }
        )
    return output


def combined_rejections(bank, mismatch):
    implausible = implausible_answer_groups(bank)
    return [left | right for left, right in zip(implausible, mismatch)]


def select(bank, scores, rejected):
    return np.asarray(
        select_with_omitted_factor_switch(bank, scores, 0.3, rejected)
    )


def correct(labels, choices):
    return int(labels[np.arange(len(labels)), choices].sum())


def redirect_selected_mismatch(bank, scores, choices, flags):
    output = choices.copy()
    changes = []
    for row_index, (row, choice) in enumerate(zip(bank, choices)):
        if not flags[row_index, choice]:
            continue
        target = boxed_answer(row["candidates"][choice]["text"])
        indices = [
            index
            for index, candidate in enumerate(row["candidates"])
            if parse_answer(candidate["text"]) == target
        ]
        if not indices:
            continue
        values = scores[row_index, indices]
        output[row_index] = indices[int(values.argmax())]
        changes.append(row_index)
    return output, changes


def redirect_any_unanimous_mismatch(bank, scores, choices, flags):
    output = choices.copy()
    changes = []
    for row_index, row in enumerate(bank):
        targets = {
            boxed_answer(candidate["text"])
            for candidate_index, candidate in enumerate(row["candidates"])
            if flags[row_index, candidate_index]
            and boxed_answer(candidate["text"]) is not None
        }
        if len(targets) != 1:
            continue
        target = next(iter(targets))
        indices = [
            index
            for index, candidate in enumerate(row["candidates"])
            if parse_answer(candidate["text"]) == target
        ]
        if not indices:
            continue
        values = scores[row_index, indices]
        replacement = indices[int(values.argmax())]
        if parse_answer(row["candidates"][replacement]["text"]) != parse_answer(
            row["candidates"][choices[row_index]]["text"]
        ):
            changes.append(row_index)
        output[row_index] = replacement
    return output, changes


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    train_flags = mismatch_flags(train_bank)
    validation_flags = mismatch_flags(validation_bank)
    for name, flags, labels in (
        ("train", train_flags, train_labels),
        ("validation", validation_flags, validation_labels),
    ):
        total = int(flags.sum())
        wrong = int((flags & ~labels).sum())
        print(
            f"{name}_candidates={total} {name}_wrong={wrong} "
            f"precision={wrong / max(total, 1):.4f}"
        )
    train_mismatch = rejected_groups(train_bank, train_flags)
    validation_mismatch = rejected_groups(validation_bank, validation_flags)
    for name, bank, labels, groups in (
        ("train", train_bank, train_labels, train_mismatch),
        ("validation", validation_bank, validation_labels, validation_mismatch),
    ):
        total = wrong = 0
        for row_index, (row, rejected) in enumerate(zip(bank, groups)):
            for answer in rejected:
                indices = [
                    index
                    for index, candidate in enumerate(row["candidates"])
                    if parse_answer(candidate["text"]) == answer
                ]
                total += 1
                wrong += int(not labels[row_index, indices[0]])
        print(
            f"{name}_groups={total} {name}_wrong_groups={wrong} "
            f"precision={wrong / max(total, 1):.4f}"
        )

    score_sets = load_or_build_model_t_oof(train_bank, train_arrays, train_labels)
    train_base_rejected = implausible_answer_groups(train_bank)
    train_new_rejected = combined_rejections(train_bank, train_mismatch)
    validation_base_rejected = implausible_answer_groups(validation_bank)
    validation_new_rejected = combined_rejections(validation_bank, validation_mismatch)
    validation_scores = structure_text_scores(
        validation_bank, validation_arrays, CODE / "runs/model_t_final_cascade"
    )
    validation_scores = coverage_adjusted_scores(validation_bank, validation_scores)
    validation_scores = occurrence_adjusted_scores(validation_bank, validation_scores)
    validation_scores = target_adjusted_scores(validation_bank, validation_scores)
    for model_name, train_score_values, validation_score_values in (
        ("t", score_sets, validation_scores),
        (
            "u",
            [stable_signal_adjusted_scores(train_bank, scores) for scores in score_sets],
            stable_signal_adjusted_scores(validation_bank, validation_scores),
        ),
    ):
        train_deltas = []
        for scores in train_score_values:
            base = select(train_bank, scores, train_base_rejected)
            filtered = select(train_bank, scores, train_new_rejected)
            train_deltas.append(correct(train_labels, filtered) - correct(train_labels, base))
        validation_base = select(validation_bank, validation_score_values, validation_base_rejected)
        validation_filtered = select(validation_bank, validation_score_values, validation_new_rejected)
        changed = np.flatnonzero(validation_base != validation_filtered).tolist()
        print(
            f"model={model_name} train_deltas={train_deltas} "
            f"validation_delta={correct(validation_labels, validation_filtered) - correct(validation_labels, validation_base):+d} "
            f"validation_rows={changed}"
        )
        redirect_train_deltas = []
        for scores in train_score_values:
            base = select(train_bank, scores, train_base_rejected)
            redirected, _ = redirect_selected_mismatch(
                train_bank, scores, base, train_flags
            )
            redirect_train_deltas.append(
                correct(train_labels, redirected) - correct(train_labels, base)
            )
        redirected_validation, redirected_rows = redirect_selected_mismatch(
            validation_bank,
            validation_score_values,
            validation_base,
            validation_flags,
        )
        print(
            f"model={model_name}_redirect train_deltas={redirect_train_deltas} "
            f"validation_delta={correct(validation_labels, redirected_validation) - correct(validation_labels, validation_base):+d} "
            f"validation_rows={redirected_rows}"
        )
        any_train_deltas = []
        for scores in train_score_values:
            base = select(train_bank, scores, train_base_rejected)
            redirected, _ = redirect_any_unanimous_mismatch(
                train_bank, scores, base, train_flags
            )
            any_train_deltas.append(
                correct(train_labels, redirected) - correct(train_labels, base)
            )
        any_validation, any_rows = redirect_any_unanimous_mismatch(
            validation_bank,
            validation_score_values,
            validation_base,
            validation_flags,
        )
        print(
            f"model={model_name}_any_redirect train_deltas={any_train_deltas} "
            f"validation_delta={correct(validation_labels, any_validation) - correct(validation_labels, validation_base):+d} "
            f"validation_rows={any_rows}"
        )


if __name__ == "__main__":
    main()
