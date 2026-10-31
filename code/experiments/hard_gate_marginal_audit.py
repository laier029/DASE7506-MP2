"""Measure each hard gate's marginal effect inside the final cascade."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from cascade_verifier import unanimous_broad_error_groups
from common import load_inputs, parse_answer
from declared_answer_consistency import redirect_consistent_boxed_endorsement
from final_declaration_consistency import final_declaration_mismatch_groups
from group_ranker_probe import candidate_labels
from model_t_probe_utils import load_or_build_model_t_oof
from numeric_plausibility import (
    fractional_answer_groups,
    safe_nonpositive_answer_groups,
)
import omitted_factor
from semantic_rank_features import semantic_rank_adjusted_scores
from stable_signal_features import stable_signal_adjusted_scores
from stable_signal_verifier import stable_signal_scores
from terminal_dependency_features import terminal_dependency_adjusted_scores


GATES = (
    "unanimous_broad_arithmetic",
    "fractional_answer",
    "safe_nonpositive_answer",
    "explicit_final_declaration_conflict",
)


def _gate_rows(bank):
    return {
        "unanimous_broad_arithmetic": unanimous_broad_error_groups(bank),
        "fractional_answer": fractional_answer_groups(bank),
        "safe_nonpositive_answer": safe_nonpositive_answer_groups(bank),
        "explicit_final_declaration_conflict": final_declaration_mismatch_groups(bank),
    }


def _select(bank, scores, rows, active):
    arithmetic = (
        rows["unanimous_broad_arithmetic"]
        if "unanimous_broad_arithmetic" in active
        else [set() for _ in bank]
    )
    additional = [set() for _ in bank]
    for name in active - {"unanimous_broad_arithmetic"}:
        additional = [
            combined | rejected
            for combined, rejected in zip(additional, rows[name])
        ]
    original = omitted_factor.unanimous_broad_error_groups
    omitted_factor.unanimous_broad_error_groups = lambda unused: arithmetic
    try:
        choices = omitted_factor.select_with_omitted_factor_switch(
            bank, scores, 0.3, additional
        )
    finally:
        omitted_factor.unanimous_broad_error_groups = original
    return np.asarray(
        redirect_consistent_boxed_endorsement(bank, scores, choices)
    )


def _correct(labels, choices):
    return int(labels[np.arange(len(labels)), choices].sum())


def _answer_changes(bank, left, right):
    return sum(
        parse_answer(row["candidates"][left_choice]["text"])
        != parse_answer(row["candidates"][right_choice]["text"])
        for row, left_choice, right_choice in zip(bank, left, right)
    )


def _score_adjustments(bank, scores):
    scores = semantic_rank_adjusted_scores(bank, scores)
    return terminal_dependency_adjusted_scores(bank, scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="runs/model_z_robust_semantic/hard-gate-marginal-audit.json",
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
    no_gates = set()
    full_train = [
        _select(train_bank, scores, train_rows, all_gates)
        for scores in train_scores
    ]
    empty_train = [
        _select(train_bank, scores, train_rows, no_gates)
        for scores in train_scores
    ]
    full_validation = _select(
        validation_bank, validation_scores, validation_rows, all_gates
    )
    empty_validation = _select(
        validation_bank, validation_scores, validation_rows, no_gates
    )
    report = {
        "protocol_note": "train OOF and validation only; test references sealed",
        "all_gates_minus_no_gates": {
            "oof_correct_deltas": [
                _correct(train_labels, full) - _correct(train_labels, empty)
                for full, empty in zip(full_train, empty_train)
            ],
            "validation_correct_delta": _correct(
                validation_labels, full_validation
            )
            - _correct(validation_labels, empty_validation),
            "validation_answer_group_changes": _answer_changes(
                validation_bank, full_validation, empty_validation
            ),
        },
        "gates": {},
    }
    for name in GATES:
        without_train = [
            _select(train_bank, scores, train_rows, all_gates - {name})
            for scores in train_scores
        ]
        only_train = [
            _select(train_bank, scores, train_rows, {name})
            for scores in train_scores
        ]
        without_validation = _select(
            validation_bank,
            validation_scores,
            validation_rows,
            all_gates - {name},
        )
        only_validation = _select(
            validation_bank, validation_scores, validation_rows, {name}
        )
        report["gates"][name] = {
            "leave_one_out_oof_correct_contribution": [
                _correct(train_labels, full) - _correct(train_labels, without)
                for full, without in zip(full_train, without_train)
            ],
            "add_alone_oof_correct_contribution": [
                _correct(train_labels, only) - _correct(train_labels, empty)
                for only, empty in zip(only_train, empty_train)
            ],
            "leave_one_out_validation_correct_contribution": _correct(
                validation_labels, full_validation
            )
            - _correct(validation_labels, without_validation),
            "add_alone_validation_correct_contribution": _correct(
                validation_labels, only_validation
            )
            - _correct(validation_labels, empty_validation),
            "leave_one_out_validation_answer_group_changes": _answer_changes(
                validation_bank, full_validation, without_validation
            ),
        }
    output = CODE / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
