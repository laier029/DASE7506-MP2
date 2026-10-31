"""Probe repeated use of an explicitly original/initial base quantity."""

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

from calculation_coverage import calculation_context, calculation_quantities, quantities
from common import load_inputs, parse_answer
from dimension_consistency_probe import choices, correct
from group_ranker_probe import candidate_labels
from model_t_probe_utils import load_or_build_model_t_oof, standardize_rows
from semantic_answer_rank_probe import PATTERNS, conditioned_rank
from stable_signal_features import stable_signal_adjusted_scores
from stable_signal_verifier import stable_signal_scores
from terminal_dependency_probe import TARGETS, dependency_features


ORIGINAL_BASE = re.compile(r"\b(?:original|initial)\s+amount\b", re.I)


def original_base_reuse(bank):
    output = np.zeros((len(bank), 4), dtype=np.float32)
    for row_index, row in enumerate(bank):
        if ORIGINAL_BASE.search(row["question"]) is None:
            continue
        numeric = []
        for token in quantities(row["question"], include_words=True):
            try:
                numeric.append((abs(Fraction(token)), token))
            except (ValueError, ZeroDivisionError):
                continue
        if not numeric:
            continue
        base = max(numeric)[1]
        for candidate_index, candidate in enumerate(row["candidates"]):
            used = Counter(
                calculation_quantities(
                    calculation_context(candidate["text"]), include_words=True
                )
            )
            output[row_index, candidate_index] = np.log1p(used[base])
    return standardize_rows(output)


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
    train_dependency = standardize_rows(
        dependency_features(train_bank, TARGETS["now_everything"])[:, :, 0]
    )
    validation_dependency = standardize_rows(
        dependency_features(validation_bank, TARGETS["now_everything"])[:, :, 0]
    )
    train_z = [scores - 0.025 * train_rank + 0.40 * train_dependency for scores in train_scores]
    validation_z = validation_scores - 0.025 * validation_rank + 0.40 * validation_dependency
    train_feature = original_base_reuse(train_bank)
    validation_feature = original_base_reuse(validation_bank)
    empty_train = [set() for _ in train_bank]
    empty_validation = [set() for _ in validation_bank]
    base_train = [correct(train_labels, choices(train_bank, scores, empty_train)) for scores in train_z]
    base_validation_choices = choices(validation_bank, validation_z, empty_validation)
    base_validation = correct(validation_labels, base_validation_choices)
    for weight in (0.0, 0.025, 0.05, 0.10, 0.20, 0.30, 0.40):
        deltas = []
        for scores, base in zip(train_z, base_train):
            selected = choices(train_bank, scores + weight * train_feature, empty_train)
            deltas.append(correct(train_labels, selected) - base)
        proposed = choices(
            validation_bank, validation_z + weight * validation_feature, empty_validation
        )
        answer_changes = []
        for row_index in np.flatnonzero(proposed != base_validation_choices):
            before = parse_answer(validation_bank[row_index]["candidates"][base_validation_choices[row_index]]["text"])
            after = parse_answer(validation_bank[row_index]["candidates"][proposed[row_index]]["text"])
            if before != after:
                answer_changes.append((int(row_index), before, after))
        print(
            f"weight={weight:+.3f} support={int((np.ptp(train_feature, axis=1) > 0).sum())}/"
            f"{int((np.ptp(validation_feature, axis=1) > 0).sum())} "
            f"oof_deltas={deltas} validation_delta={correct(validation_labels, proposed) - base_validation:+d} "
            f"answer_changes={answer_changes}",
            flush=True,
        )


if __name__ == "__main__":
    main()
