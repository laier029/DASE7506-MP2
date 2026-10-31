"""Inspect the weak 3-1 total-dependency candidate before promotion."""

import json
from pathlib import Path
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from common import load_inputs, parse_answer
from dimension_consistency_probe import choices
from group_ranker_probe import candidate_labels
from hard_gate_marginal_audit import _score_adjustments
from model_t_probe_utils import load_or_build_model_t_oof
from stable_signal_features import stable_signal_adjusted_scores
from stable_signal_verifier import stable_signal_scores
from structure_text_fusion_verifier import answer_structures
from structure_total_dependency_probe import _exclusive_total_feature


WEIGHT = -0.10


def _audit(bank, labels, scores, feature, rows):
    empty = [set() for _ in bank]
    base = choices(bank, scores, empty)
    proposed_scores = scores.copy()
    proposed_scores[rows] += WEIGHT * feature[rows]
    proposed = choices(bank, proposed_scores, empty)
    base_correct = labels[np.arange(len(labels)), base]
    proposed_correct = labels[np.arange(len(labels)), proposed]
    candidate_changes = np.flatnonzero(base != proposed)
    answer_changes = [
        int(index)
        for index in candidate_changes
        if parse_answer(bank[index]["candidates"][base[index]]["text"])
        != parse_answer(bank[index]["candidates"][proposed[index]]["text"])
    ]
    return {
        "correct_delta": int(proposed_correct.sum() - base_correct.sum()),
        "fixes": np.flatnonzero(~base_correct & proposed_correct).astype(int).tolist(),
        "harms": np.flatnonzero(base_correct & ~proposed_correct).astype(int).tolist(),
        "candidate_changes": candidate_changes.astype(int).tolist(),
        "answer_group_changes": answer_changes,
    }


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    train_scores = [
        _score_adjustments(train_bank, stable_signal_adjusted_scores(train_bank, scores))
        for scores in load_or_build_model_t_oof(
            train_bank, train_arrays, train_labels
        )
    ]
    validation_scores = _score_adjustments(
        validation_bank,
        stable_signal_scores(
            validation_bank,
            validation_arrays,
            CODE / "runs/model_z_robust_semantic",
        ),
    )
    _, train_feature = _exclusive_total_feature(train_bank)
    _, validation_feature = _exclusive_total_feature(validation_bank)
    train_rows = answer_structures(train_bank) == "3-1"
    validation_rows = answer_structures(validation_bank) == "3-1"
    report = {
        "protocol_note": "train OOF and validation only; test sealed",
        "weight": WEIGHT,
        "oof": [
            _audit(train_bank, train_labels, scores, train_feature, train_rows)
            for scores in train_scores
        ],
        "validation": _audit(
            validation_bank,
            validation_labels,
            validation_scores,
            validation_feature,
            validation_rows,
        ),
    }
    output = CODE / "runs/model_z_robust_semantic/structure-total-dependency-focus.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
