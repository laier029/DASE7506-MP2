"""Probe unanimous final-answer/last-calculation mismatch group filters."""

from collections import defaultdict
from pathlib import Path
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from answer_step_consistency import answer_step_consistency, last_step_value
from common import load_inputs, parse_answer
from coverage_cascade_verifier import coverage_adjusted_scores
from coverage_group_filter_probe import (
    accuracy,
    answer_groups,
    group_statistics,
    select,
)
from group_ranker_probe import candidate_labels
from numeric_plausibility import fractional_answer_groups
from structure_text_fusion_verifier import structure_text_scores


def unanimous_step_groups(bank, minimum_size, require_same_last=False):
    rejected = []
    for row in bank:
        groups = answer_groups(row)
        row_rejected = set()
        for answer, indices in groups.items():
            if len(indices) < minimum_size:
                continue
            states = [
                answer_step_consistency(row["candidates"][index]["text"])
                for index in indices
            ]
            if all(state is False for state in states):
                if require_same_last:
                    last_values = [
                        last_step_value(row["candidates"][index]["text"])
                        for index in indices
                    ]
                    if any(value is None for value in last_values):
                        continue
                    canonical = {round(float(value), 6) for value in last_values}
                    if len(canonical) != 1:
                        continue
                row_rejected.add(answer)
        rejected.append(row_rejected)
    return rejected


def union_rejections(*collections):
    return [set().union(*values) for values in zip(*collections)]


def main():
    train_bank, _ = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    validation_scores = structure_text_scores(
        validation_bank,
        validation_arrays,
        CODE / "runs/model_p_plausibility_coverage",
    )
    validation_scores = coverage_adjusted_scores(
        validation_bank, validation_scores
    )
    fractional = fractional_answer_groups(validation_bank)
    base = select(validation_bank, validation_scores, fractional)
    base_correct = accuracy(validation_labels, base)

    for minimum_size, require_same_last in ((2, False), (3, False), (2, True)):
        train_rejected = unanimous_step_groups(
            train_bank, minimum_size, require_same_last
        )
        validation_rejected = unanimous_step_groups(
            validation_bank, minimum_size, require_same_last
        )
        train_total, train_wrong, train_kills = group_statistics(
            train_bank, train_labels, train_rejected
        )
        val_total, val_wrong, val_kills = group_statistics(
            validation_bank, validation_labels, validation_rejected
        )
        combined = union_rejections(fractional, validation_rejected)
        chosen = select(validation_bank, validation_scores, combined)
        delta = accuracy(validation_labels, chosen) - base_correct
        print(
            f"minimum_size={minimum_size} same_last={require_same_last} "
            f"train_groups={train_total} "
            f"train_precision={100 * train_wrong / max(train_total, 1):.2f}% "
            f"train_correct_kills={train_kills} validation_groups={val_total} "
            f"validation_precision={100 * val_wrong / max(val_total, 1):.2f}% "
            f"validation_correct_kills={val_kills} delta={delta:+d}",
            flush=True,
        )


if __name__ == "__main__":
    main()
