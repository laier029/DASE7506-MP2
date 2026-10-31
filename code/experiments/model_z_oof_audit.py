"""Audit production Model Z features against Model X on five OOF schemes."""

import json
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
from model_t_probe_utils import load_or_build_model_t_oof
from semantic_rank_features import semantic_rank_adjusted_scores
from stable_signal_features import stable_signal_adjusted_scores
from stable_signal_verifier import stable_signal_scores
from terminal_dependency_features import terminal_dependency_adjusted_scores


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    x_score_sets = [
        stable_signal_adjusted_scores(train_bank, scores)
        for scores in load_or_build_model_t_oof(train_bank, train_arrays, train_labels)
    ]
    empty_train = [set() for _ in train_bank]
    deltas = []
    fixes = []
    harms = []
    candidate_changes = []
    answer_changes = []
    for scores in x_score_sets:
        x_choices = choices(train_bank, scores, empty_train)
        z_scores = semantic_rank_adjusted_scores(train_bank, scores)
        z_scores = terminal_dependency_adjusted_scores(train_bank, z_scores)
        z_choices = choices(train_bank, z_scores, empty_train)
        x_correct = train_labels[np.arange(len(train_labels)), x_choices]
        z_correct = train_labels[np.arange(len(train_labels)), z_choices]
        deltas.append(int(z_correct.sum() - x_correct.sum()))
        fixes.append(int((~x_correct & z_correct).sum()))
        harms.append(int((x_correct & ~z_correct).sum()))
        candidate_changes.append(int(np.sum(z_choices != x_choices)))
        answer_changes.append(
            sum(
                parse_answer(row["candidates"][left]["text"])
                != parse_answer(row["candidates"][right]["text"])
                for row, left, right in zip(train_bank, x_choices, z_choices)
            )
        )

    x_validation_scores = stable_signal_scores(
        validation_bank, validation_arrays, CODE / "runs/model_z_robust_semantic"
    )
    z_validation_scores = semantic_rank_adjusted_scores(
        validation_bank, x_validation_scores
    )
    z_validation_scores = terminal_dependency_adjusted_scores(
        validation_bank, z_validation_scores
    )
    empty_validation = [set() for _ in validation_bank]
    x_validation_choices = choices(
        validation_bank, x_validation_scores, empty_validation
    )
    z_validation_choices = choices(
        validation_bank, z_validation_scores, empty_validation
    )
    changed = np.flatnonzero(z_validation_choices != x_validation_choices).tolist()
    changed_answers = [
        row_index
        for row_index in changed
        if parse_answer(
            validation_bank[row_index]["candidates"][z_validation_choices[row_index]]["text"]
        )
        != parse_answer(
            validation_bank[row_index]["candidates"][x_validation_choices[row_index]]["text"]
        )
    ]
    report = {
        "protocol_note": "train OOF and validation only; test references sealed",
        "oof_correct_deltas_model_z_over_x": deltas,
        "oof_fixes": fixes,
        "oof_harms": harms,
        "oof_candidate_index_changes": candidate_changes,
        "oof_answer_group_changes": answer_changes,
        "validation_correct_delta_model_z_over_x": correct(validation_labels, z_validation_choices)
        - correct(validation_labels, x_validation_choices),
        "validation_fixes": int(
            (
                ~validation_labels[np.arange(len(validation_labels)), x_validation_choices]
                & validation_labels[np.arange(len(validation_labels)), z_validation_choices]
            ).sum()
        ),
        "validation_harms": int(
            (
                validation_labels[np.arange(len(validation_labels)), x_validation_choices]
                & ~validation_labels[np.arange(len(validation_labels)), z_validation_choices]
            ).sum()
        ),
        "validation_candidate_index_changes": changed,
        "validation_answer_group_changes": changed_answers,
    }
    output = CODE / "runs/model_z_robust_semantic/oof-audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
