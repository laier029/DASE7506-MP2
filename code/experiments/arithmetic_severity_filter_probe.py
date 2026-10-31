"""Probe severe candidate arithmetic failures as answer-group hard gates."""

from pathlib import Path
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from arithmetic_consistency import (
    arithmetic_counts,
    broad_arithmetic_counts,
    plain_text_arithmetic_counts,
)
from common import load_inputs, parse_answer
from dimension_consistency_probe import choices, correct
from group_ranker_probe import candidate_labels
from model_t_probe_utils import load_or_build_model_t_oof
from stable_signal_features import stable_signal_adjusted_scores
from stable_signal_verifier import stable_signal_scores


def candidate_counts(bank):
    output = np.zeros((len(bank), 4, 3), dtype=np.int16)
    for row_index, row in enumerate(bank):
        for candidate_index, candidate in enumerate(row["candidates"]):
            text = candidate["text"]
            output[row_index, candidate_index] = (
                arithmetic_counts(text)[1],
                broad_arithmetic_counts(text)[1],
                plain_text_arithmetic_counts(text)[1],
            )
    return output


def group_rejections(bank, flags):
    output = []
    for row_index, row in enumerate(bank):
        groups = {}
        for index, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate["text"])
            if answer is not None:
                groups.setdefault(answer, []).append(index)
        output.append(
            {
                answer
                for answer, indices in groups.items()
                if flags[row_index, indices].all()
            }
        )
    return output


def group_statistics(bank, labels, rejected):
    total = wrong = 0
    for row_index, answers in enumerate(rejected):
        for answer in answers:
            indices = [
                index
                for index, candidate in enumerate(bank[row_index]["candidates"])
                if parse_answer(candidate["text"]) == answer
            ]
            total += 1
            wrong += int(not any(labels[row_index, indices]))
    return total, wrong


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    train_counts = candidate_counts(train_bank)
    validation_counts = candidate_counts(validation_bank)
    rules = {
        "narrow_1": lambda values: values[:, :, 0] >= 1,
        "narrow_2": lambda values: values[:, :, 0] >= 2,
        "broad_2": lambda values: values[:, :, 1] >= 2,
        "broad_3": lambda values: values[:, :, 1] >= 3,
        "plain_2": lambda values: values[:, :, 2] >= 2,
        "plain_3": lambda values: values[:, :, 2] >= 3,
        "corroborated": lambda values: (
            (values[:, :, 0] > 0).astype(int)
            + (values[:, :, 1] > 0).astype(int)
            + (values[:, :, 2] > 0).astype(int)
        ) >= 2,
        "total_4": lambda values: values.sum(axis=2) >= 4,
    }
    oof_scores = [
        stable_signal_adjusted_scores(train_bank, scores)
        for scores in load_or_build_model_t_oof(train_bank, train_arrays, train_labels)
    ]
    validation_scores = stable_signal_scores(
        validation_bank, validation_arrays, CODE / "runs/model_x_full_cascade"
    )
    empty_train = [set() for _ in train_bank]
    empty_validation = [set() for _ in validation_bank]
    base_train_choices = [choices(train_bank, scores, empty_train) for scores in oof_scores]
    base_train = [correct(train_labels, selected) for selected in base_train_choices]
    base_validation_choices = choices(
        validation_bank, validation_scores, empty_validation
    )
    base_validation = correct(validation_labels, base_validation_choices)

    for name, rule in rules.items():
        train_rejected = group_rejections(train_bank, rule(train_counts))
        validation_rejected = group_rejections(
            validation_bank, rule(validation_counts)
        )
        train_total, train_wrong = group_statistics(
            train_bank, train_labels, train_rejected
        )
        validation_total, validation_wrong = group_statistics(
            validation_bank, validation_labels, validation_rejected
        )
        deltas = []
        changes = []
        for scores, base, base_selected in zip(
            oof_scores, base_train, base_train_choices
        ):
            selected = choices(train_bank, scores, train_rejected)
            deltas.append(correct(train_labels, selected) - base)
            changes.append(int(np.sum(selected != base_selected)))
        proposed = choices(
            validation_bank, validation_scores, validation_rejected
        )
        print(
            f"rule={name} train={train_wrong}/{train_total} "
            f"validation={validation_wrong}/{validation_total} "
            f"oof_deltas={deltas} oof_changes={changes} "
            f"validation_delta={correct(validation_labels, proposed) - base_validation:+d} "
            f"validation_changes={np.flatnonzero(proposed != base_validation_choices).tolist()}",
            flush=True,
        )


if __name__ == "__main__":
    main()
