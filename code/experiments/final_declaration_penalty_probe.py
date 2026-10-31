"""Probe candidate-level penalties for explicit final-declaration conflicts."""

from pathlib import Path
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from common import load_inputs
from coverage_cascade_verifier import coverage_adjusted_scores
from declared_answer_consistency import redirect_consistent_boxed_endorsement
from final_declaration_probe import mismatch_targets
from group_ranker_probe import candidate_labels
from model_t_probe_utils import FastFinalSelector, load_or_build_model_t_oof, standardize_rows
from occurrence_plausibility_verifier import occurrence_adjusted_scores
from stable_signal_features import stable_signal_adjusted_scores
from structure_text_fusion_verifier import structure_text_scores
from target_omitted_factor_verifier import target_adjusted_scores


def correct(labels, choices):
    return int(labels[np.arange(len(labels)), choices].sum())


def endorsed_select(selector, bank, scores):
    base = selector.select(scores)
    return np.asarray(
        redirect_consistent_boxed_endorsement(bank, scores, base)
    )


def flags(targets):
    return np.asarray(
        [[target is not None for target in row] for row in targets],
        dtype=np.float32,
    )


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    train_scores = [
        stable_signal_adjusted_scores(train_bank, scores)
        for scores in load_or_build_model_t_oof(
            train_bank, train_arrays, train_labels
        )
    ]
    validation_scores = structure_text_scores(
        validation_bank, validation_arrays, CODE / "runs/model_t_final_cascade"
    )
    validation_scores = coverage_adjusted_scores(validation_bank, validation_scores)
    validation_scores = occurrence_adjusted_scores(validation_bank, validation_scores)
    validation_scores = target_adjusted_scores(validation_bank, validation_scores)
    validation_scores = stable_signal_adjusted_scores(validation_bank, validation_scores)
    train_selector = FastFinalSelector(train_bank)
    validation_selector = FastFinalSelector(validation_bank)
    train_feature = standardize_rows(flags(mismatch_targets(train_bank)))
    validation_feature = standardize_rows(flags(mismatch_targets(validation_bank)))
    train_base = [
        correct(train_labels, endorsed_select(train_selector, train_bank, scores))
        for scores in train_scores
    ]
    validation_base = correct(
        validation_labels,
        endorsed_select(validation_selector, validation_bank, validation_scores),
    )
    trials = []
    for weight in np.round(np.arange(0.0, 1.001, 0.05), 3):
        choices = endorsed_select(
            train_selector,
            train_bank,
            train_scores[0] - weight * train_feature,
        )
        trials.append((correct(train_labels, choices), -weight, weight))
    _, _, weight = max(trials)
    train_deltas = [
        correct(
            train_labels,
            endorsed_select(
                train_selector, train_bank, scores - weight * train_feature
            ),
        )
        - base
        for scores, base in zip(train_scores, train_base)
    ]
    validation_delta = correct(
        validation_labels,
        endorsed_select(
            validation_selector,
            validation_bank,
            validation_scores - weight * validation_feature,
        ),
    ) - validation_base
    print(
        f"chosen_weight={weight:.2f} train_deltas={train_deltas} "
        f"validation_delta={validation_delta:+d}"
    )
    for weight in np.round(np.arange(0.1, 1.001, 0.1), 3):
        deltas = [
            correct(
                train_labels,
                endorsed_select(
                    train_selector, train_bank, scores - weight * train_feature
                ),
            )
            - base
            for scores, base in zip(train_scores, train_base)
        ]
        val_delta = correct(
            validation_labels,
            endorsed_select(
                validation_selector,
                validation_bank,
                validation_scores - weight * validation_feature,
            ),
        ) - validation_base
        print(
            f"weight={weight:.2f} train_deltas={deltas} "
            f"validation_delta={val_delta:+d}"
        )


if __name__ == "__main__":
    main()
