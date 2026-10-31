"""Compare irreversible hard gates with bounded soft penalties."""

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
from group_ranker_probe import candidate_labels
from hard_gate_marginal_audit import GATES, _gate_rows, _score_adjustments, _select
from model_t_probe_utils import load_or_build_model_t_oof
from stable_signal_features import stable_signal_adjusted_scores
from stable_signal_verifier import stable_signal_scores


WEIGHTS = (0.0, 0.10, 0.25, 0.50, 1.0, 2.0, 4.0)


def _indicator(bank, rejected):
    return np.asarray(
        [
            [
                float(parse_answer(candidate["text"]) in row_rejected)
                for candidate in row["candidates"]
            ]
            for row, row_rejected in zip(bank, rejected)
        ],
        dtype=np.float32,
    )


def _correct(labels, choices):
    return int(labels[np.arange(len(labels)), choices].sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="runs/model_z_robust_semantic/gate-soft-vs-hard.json",
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
    train_rows = _gate_rows(train_bank)
    validation_rows = _gate_rows(validation_bank)
    all_gates = set(GATES)
    report = {
        "protocol_note": (
            "fixed grid selected on primary OOF; four extra OOF schemes for robustness; validation descriptive; test sealed"
        ),
        "weights_are_candidate_score_penalties": list(WEIGHTS),
        "gates": {},
    }
    for name in GATES:
        active = all_gates - {name}
        train_indicator = _indicator(train_bank, train_rows[name])
        validation_indicator = _indicator(validation_bank, validation_rows[name])
        train_without = [
            _select(train_bank, scores, train_rows, active)
            for scores in train_scores
        ]
        validation_without = _select(
            validation_bank, validation_scores, validation_rows, active
        )
        hard_train = [
            _select(train_bank, scores, train_rows, all_gates)
            for scores in train_scores
        ]
        hard_validation = _select(
            validation_bank, validation_scores, validation_rows, all_gates
        )
        trials = []
        for weight in WEIGHTS:
            soft_train = [
                _select(
                    train_bank,
                    scores - weight * train_indicator,
                    train_rows,
                    active,
                )
                for scores in train_scores
            ]
            soft_validation = _select(
                validation_bank,
                validation_scores - weight * validation_indicator,
                validation_rows,
                active,
            )
            trials.append(
                {
                    "weight": weight,
                    "oof_correct_deltas_over_without_gate": [
                        _correct(train_labels, soft) - _correct(train_labels, without)
                        for soft, without in zip(soft_train, train_without)
                    ],
                    "validation_correct_delta_over_without_gate": _correct(
                        validation_labels, soft_validation
                    )
                    - _correct(validation_labels, validation_without),
                }
            )
        best = max(
            trials,
            key=lambda trial: (
                trial["oof_correct_deltas_over_without_gate"][0],
                min(trial["oof_correct_deltas_over_without_gate"]),
                sum(trial["oof_correct_deltas_over_without_gate"]),
                -trial["weight"],
            ),
        )
        report["gates"][name] = {
            "hard_oof_correct_deltas_over_without_gate": [
                _correct(train_labels, hard) - _correct(train_labels, without)
                for hard, without in zip(hard_train, train_without)
            ],
            "hard_validation_correct_delta_over_without_gate": _correct(
                validation_labels, hard_validation
            )
            - _correct(validation_labels, validation_without),
            "selected_by_primary_oof_weight": best["weight"],
            "soft_trials": trials,
        }
    output = CODE / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
