"""Compare soft terminal-dependency scoring with progressively harder gates."""

from pathlib import Path
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
from semantic_answer_rank_probe import PATTERNS, conditioned_rank
from stable_signal_features import stable_signal_adjusted_scores
from stable_signal_verifier import stable_signal_scores
from terminal_dependency_probe import TARGETS, dependency_features


def rejected_groups(bank, raw, rule):
    output = []
    for row_index, row in enumerate(bank):
        groups = {}
        for index, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate["text"])
            if answer is not None:
                groups.setdefault(answer, []).append(index)
        row_max = float(raw[row_index].max())
        output.append(
            {
                answer
                for answer, indices in groups.items()
                if rule(raw[row_index, indices], row_max)
            }
        )
    return output


def precision(bank, labels, rejected):
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
    return wrong, total


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
        validation_bank, validation_arrays, CODE / "runs/model_z_robust_semantic"
    )
    train_rank = conditioned_rank(train_bank, PATTERNS["remainder_difference"])
    validation_rank = conditioned_rank(validation_bank, PATTERNS["remainder_difference"])
    train_raw = dependency_features(train_bank, TARGETS["now_everything"])[:, :, 0]
    validation_raw = dependency_features(
        validation_bank, TARGETS["now_everything"]
    )[:, :, 0]
    train_dependency = standardize_rows(train_raw)
    validation_dependency = standardize_rows(validation_raw)
    train_z = [
        scores - 0.025 * train_rank + 0.20 * train_dependency
        for scores in train_scores
    ]
    validation_z = validation_scores - 0.025 * validation_rank + 0.20 * validation_dependency
    empty_train = [set() for _ in train_bank]
    empty_validation = [set() for _ in validation_bank]
    base_train_choices = [choices(train_bank, scores, empty_train) for scores in train_z]
    base_train = [correct(train_labels, selected) for selected in base_train_choices]
    base_validation_choices = choices(validation_bank, validation_z, empty_validation)
    base_validation = correct(validation_labels, base_validation_choices)
    rules = {
        "zero_if_positive": lambda values, row_max: row_max > 0 and np.all(values == 0),
        "at_most_one_if_deeper": lambda values, row_max: row_max > 1 and np.all(values <= 1),
        "below_row_max": lambda values, row_max: row_max > 0 and np.max(values) < row_max,
    }
    for name, rule in rules.items():
        train_rejected = rejected_groups(train_bank, train_raw, rule)
        validation_rejected = rejected_groups(validation_bank, validation_raw, rule)
        deltas = []
        changes = []
        for scores, base, base_selected in zip(
            train_z, base_train, base_train_choices
        ):
            selected = choices(train_bank, scores, train_rejected)
            deltas.append(correct(train_labels, selected) - base)
            changes.append(int(np.sum(selected != base_selected)))
        proposed = choices(
            validation_bank, validation_z, validation_rejected
        )
        print(
            f"rule={name} train_precision={precision(train_bank, train_labels, train_rejected)} "
            f"validation_precision={precision(validation_bank, validation_labels, validation_rejected)} "
            f"oof_deltas={deltas} oof_changes={changes} "
            f"validation_delta={correct(validation_labels, proposed) - base_validation:+d} "
            f"validation_changes={np.flatnonzero(proposed != base_validation_choices).tolist()}",
            flush=True,
        )


if __name__ == "__main__":
    main()
