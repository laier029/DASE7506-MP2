"""Probe high-precision final-answer plausibility filters."""

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
from coverage_group_filter_probe import (
    accuracy,
    answer_groups,
    group_statistics,
    select,
)
from group_ranker_probe import candidate_labels
from structure_text_fusion_verifier import structure_text_scores


COUNT_QUESTION = re.compile(r"(?i)\b(?:how many|number of)\b")
NEGATIVE_INPUT = re.compile(r"(?<!\w)-\s*\d")


def rejected_groups(bank, rule):
    output = []
    for row in bank:
        rejected = set()
        groups = answer_groups(row)
        for answer in groups:
            value = Fraction(answer)
            if rule(row["question"], value, groups):
                rejected.add(answer)
        output.append(rejected)
    return output


def main():
    train_bank, _ = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    validation_scores = structure_text_scores(
        validation_bank,
        validation_arrays,
        CODE / "runs/model_o_coverage_cascade",
    )
    validation_scores = coverage_adjusted_scores(
        validation_bank, validation_scores
    )
    base = select(
        validation_bank,
        validation_scores,
        [set() for _ in validation_bank],
    )
    base_correct = accuracy(validation_labels, base)

    rules = {
        "negative": lambda question, value, groups: value < 0,
        "nonpositive": lambda question, value, groups: value <= 0,
        "nonpositive_without_negative_input": lambda question, value, groups: (
            value <= 0 and NEGATIVE_INPUT.search(question) is None
        ),
        "fractional_any": lambda question, value, groups: value.denominator != 1,
        "fractional_count_question": lambda question, value, groups: (
            value.denominator != 1 and COUNT_QUESTION.search(question) is not None
        ),
        "fractional_count_with_integer_alternative": lambda question, value, groups: (
            value.denominator != 1
            and COUNT_QUESTION.search(question) is not None
            and any(Fraction(answer).denominator == 1 for answer in groups)
        ),
        "fractional_or_safe_nonpositive": lambda question, value, groups: (
            value.denominator != 1
            or (value <= 0 and NEGATIVE_INPUT.search(question) is None)
        ),
    }
    for name, rule in rules.items():
        train_rejected = rejected_groups(train_bank, rule)
        validation_rejected = rejected_groups(validation_bank, rule)
        train_total, train_wrong, train_kills = group_statistics(
            train_bank, train_labels, train_rejected
        )
        val_total, val_wrong, val_kills = group_statistics(
            validation_bank, validation_labels, validation_rejected
        )
        chosen = select(validation_bank, validation_scores, validation_rejected)
        delta = accuracy(validation_labels, chosen) - base_correct
        print(
            f"rule={name} train_groups={train_total} "
            f"train_precision={100 * train_wrong / max(train_total, 1):.2f}% "
            f"train_correct_kills={train_kills} validation_groups={val_total} "
            f"validation_precision={100 * val_wrong / max(val_total, 1):.2f}% "
            f"validation_correct_kills={val_kills} delta={delta:+d}",
            flush=True,
        )


if __name__ == "__main__":
    main()
