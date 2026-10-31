"""Probe answer-structure-specific remainder answer-rank weights."""

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
from structure_text_fusion_verifier import answer_structures
from terminal_dependency_probe import TARGETS, dependency_features


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
    train_structures = answer_structures(train_bank)
    validation_structures = answer_structures(validation_bank)
    train_dep_weight = np.full(len(train_bank), 0.40, dtype=np.float32)
    validation_dep_weight = np.full(len(validation_bank), 0.40, dtype=np.float32)
    train_dep_weight[train_structures == "3-1"] = 0.50
    validation_dep_weight[validation_structures == "3-1"] = 0.50
    train_z = [
        scores - 0.025 * train_rank + train_dep_weight[:, None] * train_dependency
        for scores in train_scores
    ]
    validation_z = (
        validation_scores
        - 0.025 * validation_rank
        + validation_dep_weight[:, None] * validation_dependency
    )
    empty_train = [set() for _ in train_bank]
    empty_validation = [set() for _ in validation_bank]
    base_train = [correct(train_labels, choices(train_bank, scores, empty_train)) for scores in train_z]
    base_validation_choices = choices(validation_bank, validation_z, empty_validation)
    base_validation = correct(validation_labels, base_validation_choices)
    weights = (0.0, -0.025, -0.05, -0.075, -0.10, -0.15)
    for structure in ("3-1", "2-2", "2-1-1", "1-1-1-1"):
        train_rows = train_structures == structure
        validation_rows = validation_structures == structure
        trials = []
        for weight in weights:
            scores = train_z[0].copy()
            scores[train_rows] += (weight + 0.025) * train_rank[train_rows]
            selected = choices(train_bank, scores, empty_train)
            trials.append((correct(train_labels, selected), -abs(weight + 0.025), weight))
        _, _, weight = max(trials)
        deltas = []
        for base_scores, base in zip(train_z, base_train):
            scores = base_scores.copy()
            scores[train_rows] += (weight + 0.025) * train_rank[train_rows]
            selected = choices(train_bank, scores, empty_train)
            deltas.append(correct(train_labels, selected) - base)
        scores = validation_z.copy()
        scores[validation_rows] += (weight + 0.025) * validation_rank[validation_rows]
        proposed = choices(validation_bank, scores, empty_validation)
        answer_changes = []
        for row_index in np.flatnonzero(proposed != base_validation_choices):
            before = parse_answer(validation_bank[row_index]["candidates"][base_validation_choices[row_index]]["text"])
            after = parse_answer(validation_bank[row_index]["candidates"][proposed[row_index]]["text"])
            if before != after:
                answer_changes.append((int(row_index), before, after))
        print(
            f"structure={structure} weight={weight:+.3f} oof_deltas={deltas} "
            f"validation_delta={correct(validation_labels, proposed) - base_validation:+d} "
            f"answer_changes={answer_changes}",
            flush=True,
        )


if __name__ == "__main__":
    main()
