"""Probe plain-text arithmetic errors as candidate and unanimous group gates."""

from pathlib import Path
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from arithmetic_consistency import plain_text_arithmetic_counts
from common import load_inputs, parse_answer
from coverage_cascade_verifier import coverage_adjusted_scores
from coverage_group_filter_probe import answer_groups, group_statistics
from group_ranker_probe import candidate_labels
from numeric_plausibility import implausible_answer_groups
from occurrence_plausibility_verifier import occurrence_adjusted_scores
from omitted_factor import select_with_omitted_factor_switch
from structure_text_fusion_verifier import structure_text_scores
from target_omitted_factor_verifier import target_adjusted_scores


def candidate_flags(bank):
    output = np.zeros((len(bank), 4), dtype=bool)
    for row_index, row in enumerate(bank):
        for candidate_index, candidate in enumerate(row["candidates"]):
            output[row_index, candidate_index] = (
                plain_text_arithmetic_counts(candidate["text"])[1] > 0
            )
    return output


def unanimous_groups(bank, flags, minimum_size=2):
    output = []
    for row_index, row in enumerate(bank):
        rejected = set()
        for answer, indices in answer_groups(row).items():
            if len(indices) >= minimum_size and flags[row_index, indices].all():
                rejected.add(answer)
        output.append(rejected)
    return output


def candidate_statistics(flags, labels):
    total = int(flags.sum())
    wrong = int((flags & ~labels).sum())
    kills = int(((labels & ~flags).sum(axis=1) == 0).sum())
    return total, wrong, kills


def main():
    train_bank, _ = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    train_flags = candidate_flags(train_bank)
    validation_flags = candidate_flags(validation_bank)
    print("train candidates", candidate_statistics(train_flags, train_labels))
    print(
        "validation candidates",
        candidate_statistics(validation_flags, validation_labels),
    )

    scores = structure_text_scores(
        validation_bank,
        validation_arrays,
        CODE / "runs/model_t_final_cascade",
    )
    scores = coverage_adjusted_scores(validation_bank, scores)
    scores = occurrence_adjusted_scores(validation_bank, scores)
    scores = target_adjusted_scores(validation_bank, scores)
    implausible = implausible_answer_groups(validation_bank)
    base = np.asarray(
        select_with_omitted_factor_switch(
            validation_bank, scores, 0.3, implausible
        )
    )
    base_correct = int(
        validation_labels[np.arange(len(validation_labels)), base].sum()
    )
    for minimum_size in (2, 3):
        train_rejected = unanimous_groups(
            train_bank, train_flags, minimum_size
        )
        validation_rejected = unanimous_groups(
            validation_bank, validation_flags, minimum_size
        )
        print(
            "train groups",
            minimum_size,
            group_statistics(train_bank, train_labels, train_rejected),
        )
        print(
            "validation groups",
            minimum_size,
            group_statistics(
                validation_bank, validation_labels, validation_rejected
            ),
        )
        combined = [
            left | right
            for left, right in zip(implausible, validation_rejected)
        ]
        choices = np.asarray(
            select_with_omitted_factor_switch(
                validation_bank, scores, 0.3, combined
            )
        )
        correct = int(
            validation_labels[
                np.arange(len(validation_labels)), choices
            ].sum()
        )
        print(
            f"minimum_size={minimum_size} validation_delta={correct - base_correct:+d}"
        )


if __name__ == "__main__":
    main()
