"""Break down the joint hard-gate contribution by answer structure."""

import json
from pathlib import Path
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from common import load_inputs
from group_ranker_probe import candidate_labels
from hard_gate_marginal_audit import GATES, _gate_rows, _score_adjustments, _select
from model_t_probe_utils import load_or_build_model_t_oof
from stable_signal_features import stable_signal_adjusted_scores
from stable_signal_verifier import stable_signal_scores
from structure_text_fusion_verifier import answer_structures


STRUCTURES = ("4", "3-1", "2-2", "2-1-1", "1-1-1-1")


def _delta(labels, full, empty, rows):
    row_indices = np.flatnonzero(rows)
    full_correct = labels[row_indices, full[row_indices]]
    empty_correct = labels[row_indices, empty[row_indices]]
    return {
        "correct_delta": int(full_correct.sum() - empty_correct.sum()),
        "fixes": int((~empty_correct & full_correct).sum()),
        "harms": int((empty_correct & ~full_correct).sum()),
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
    train_rows = _gate_rows(train_bank)
    validation_rows = _gate_rows(validation_bank)
    active = set(GATES)
    train_full = [
        _select(train_bank, scores, train_rows, active) for scores in train_scores
    ]
    train_empty = [
        _select(train_bank, scores, train_rows, set()) for scores in train_scores
    ]
    validation_full = _select(
        validation_bank, validation_scores, validation_rows, active
    )
    validation_empty = _select(
        validation_bank, validation_scores, validation_rows, set()
    )
    train_structures = answer_structures(train_bank)
    validation_structures = answer_structures(validation_bank)
    report = {
        "protocol_note": "train OOF and validation only; test sealed",
        "comparison": "all four hard gates versus no hard gates",
        "structures": {},
    }
    for structure in STRUCTURES:
        train_mask = train_structures == structure
        validation_mask = validation_structures == structure
        report["structures"][structure] = {
            "train_rows": int(train_mask.sum()),
            "oof": [
                _delta(train_labels, full, empty, train_mask)
                for full, empty in zip(train_full, train_empty)
            ],
            "validation_rows": int(validation_mask.sum()),
            "validation": _delta(
                validation_labels,
                validation_full,
                validation_empty,
                validation_mask,
            ),
        }
    output = CODE / "runs/model_z_robust_semantic/hard-gate-structure-audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
