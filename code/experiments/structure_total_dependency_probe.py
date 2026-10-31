"""Probe total-target dependency coverage within answer structures."""

import argparse
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
from hard_gate_marginal_audit import _score_adjustments
from model_t_probe_utils import load_or_build_model_t_oof, standardize_rows
from stable_signal_features import stable_signal_adjusted_scores
from stable_signal_verifier import stable_signal_scores
from structure_text_fusion_verifier import answer_structures
from terminal_dependency_probe import TARGETS, dependency_features


WEIGHTS = (-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30, 0.40)
STRUCTURES = ("3-1", "2-2", "2-1-1", "1-1-1-1")


def _exclusive_total_feature(bank):
    raw = dependency_features(bank, TARGETS["total"])[:, :, 0]
    cumulative_rows = np.asarray(
        [TARGETS["now_everything"].search(row["question"]) is not None for row in bank]
    )
    raw[cumulative_rows] = 0.0
    return raw, standardize_rows(raw)


def _answer_changes(bank, left, right):
    return [
        int(row_index)
        for row_index in np.flatnonzero(left != right)
        if parse_answer(bank[row_index]["candidates"][left[row_index]]["text"])
        != parse_answer(bank[row_index]["candidates"][right[row_index]]["text"])
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="runs/model_z_robust_semantic/structure-total-dependency-probe.json",
    )
    args = parser.parse_args()
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
    train_raw, train_feature = _exclusive_total_feature(train_bank)
    validation_raw, validation_feature = _exclusive_total_feature(validation_bank)
    train_structures = answer_structures(train_bank)
    validation_structures = answer_structures(validation_bank)
    empty_train = [set() for _ in train_bank]
    empty_validation = [set() for _ in validation_bank]
    base_train_choices = [
        choices(train_bank, scores, empty_train) for scores in train_scores
    ]
    base_train = [
        correct(train_labels, selected) for selected in base_train_choices
    ]
    base_validation_choices = choices(
        validation_bank, validation_scores, empty_validation
    )
    base_validation = correct(validation_labels, base_validation_choices)
    report = {
        "protocol_note": "train OOF and validation only; test sealed",
        "feature": "total/altogether dependency leaf count excluding now/currently/everything",
        "structures": {},
    }
    for structure in STRUCTURES:
        train_rows = train_structures == structure
        validation_rows = validation_structures == structure
        trials = []
        for weight in WEIGHTS:
            deltas = []
            for scores, base in zip(train_scores, base_train):
                proposed_scores = scores.copy()
                proposed_scores[train_rows] += weight * train_feature[train_rows]
                selected = choices(train_bank, proposed_scores, empty_train)
                deltas.append(correct(train_labels, selected) - base)
            validation_proposed_scores = validation_scores.copy()
            validation_proposed_scores[validation_rows] += (
                weight * validation_feature[validation_rows]
            )
            validation_selected = choices(
                validation_bank, validation_proposed_scores, empty_validation
            )
            trials.append(
                {
                    "weight": weight,
                    "oof_correct_deltas": deltas,
                    "validation_correct_delta": correct(
                        validation_labels, validation_selected
                    )
                    - base_validation,
                    "validation_answer_group_changes": _answer_changes(
                        validation_bank,
                        validation_selected,
                        base_validation_choices,
                    ),
                }
            )
        selected_trial = max(
            trials,
            key=lambda trial: (
                trial["oof_correct_deltas"][0],
                min(trial["oof_correct_deltas"]),
                sum(trial["oof_correct_deltas"]),
                -abs(trial["weight"]),
            ),
        )
        report["structures"][structure] = {
            "train_support_rows": int(
                ((np.ptp(train_raw, axis=1) > 0) & train_rows).sum()
            ),
            "validation_support_rows": int(
                ((np.ptp(validation_raw, axis=1) > 0) & validation_rows).sum()
            ),
            "selected_by_first_oof_weight": selected_trial["weight"],
            "trials": trials,
        }
    output = CODE / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
