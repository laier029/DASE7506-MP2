"""Test residual multiplicity-aware calculation coverage on Model P."""

from pathlib import Path
import argparse
import sys

import joblib
import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from cascade_stability_probe import (
    answer_structures,
    balanced_random_folds,
    model_m_oof,
)
from coverage_group_filter_probe import accuracy, select
from group_ranker_probe import candidate_labels
from numeric_plausibility import fractional_answer_groups
from numeric_filter_stability_probe import apply_model_o_coverage
from question_number_coverage_probe import coverage_features
from common import load_inputs
from coverage_cascade_verifier import coverage_adjusted_scores
from structure_text_fusion_verifier import structure_text_scores


def standardize_rows(values):
    return (values - values.mean(axis=1, keepdims=True)) / (
        values.std(axis=1, keepdims=True) + 1e-6
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--focus", choices=("3-1", "2-2", "2-1-1", "1-1-1-1")
    )
    args = parser.parse_args()
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    train_structures = answer_structures(train_bank)
    validation_structures = answer_structures(validation_bank)
    artifact = joblib.load(
        CODE / "runs/model_p_plausibility_coverage/structure_text_model.joblib"
    )
    texts = [
        candidate["text"]
        for row in train_bank
        for candidate in row["candidates"]
    ]
    text_features = artifact["vectorizer"].transform(texts)

    seeds = (7506, 11, 29, 47, 83)
    score_sets = []
    for seed in seeds:
        folds = (
            np.arange(len(train_bank)) % 5
            if seed == 7506
            else balanced_random_folds(len(train_bank), seed)
        )
        scores = model_m_oof(
            train_bank,
            train_arrays,
            train_labels,
            folds,
            text_features,
            train_structures,
        )
        score_sets.append(
            apply_model_o_coverage(train_bank, scores, train_structures)
        )
    validation_scores = structure_text_scores(
        validation_bank,
        validation_arrays,
        CODE / "runs/model_p_plausibility_coverage",
    )
    validation_scores = coverage_adjusted_scores(
        validation_bank, validation_scores
    )
    train_rejected = fractional_answer_groups(train_bank)
    validation_rejected = fractional_answer_groups(validation_bank)
    train_base = [
        accuracy(train_labels, select(train_bank, scores, train_rejected))
        for scores in score_sets
    ]
    validation_base = accuracy(
        validation_labels,
        select(validation_bank, validation_scores, validation_rejected),
    )

    train_feature = standardize_rows(coverage_features(train_bank)[:, :, 1])
    validation_feature = standardize_rows(
        coverage_features(validation_bank)[:, :, 1]
    )
    weights = np.round(np.arange(-0.5, 0.5001, 0.025), 3)
    if args.focus:
        train_rows = train_structures == args.focus
        validation_rows = validation_structures == args.focus
        print(f"focused weight plateau structure={args.focus}")
        for weight in np.arange(0.0, 0.501, 0.025):
            train_adjustment = np.zeros_like(train_feature)
            train_adjustment[train_rows] = weight * train_feature[train_rows]
            train_deltas = [
                accuracy(
                    train_labels,
                    select(
                        train_bank, scores + train_adjustment, train_rejected
                    ),
                )
                - base
                for scores, base in zip(score_sets, train_base)
            ]
            validation_adjustment = np.zeros_like(validation_feature)
            validation_adjustment[validation_rows] = (
                weight * validation_feature[validation_rows]
            )
            validation_delta = accuracy(
                validation_labels,
                select(
                    validation_bank,
                    validation_scores + validation_adjustment,
                    validation_rejected,
                ),
            ) - validation_base
            print(
                f"  weight={weight:.3f} train_deltas={train_deltas} "
                f"validation_delta={validation_delta:+d}",
                flush=True,
            )
        return

    for structure in ("3-1", "2-2", "2-1-1", "1-1-1-1"):
        train_rows = train_structures == structure
        validation_rows = validation_structures == structure
        trials = []
        for weight in weights:
            adjustment = np.zeros_like(train_feature)
            adjustment[train_rows] = weight * train_feature[train_rows]
            count = accuracy(
                train_labels,
                select(
                    train_bank,
                    score_sets[0] + adjustment,
                    train_rejected,
                ),
            )
            trials.append((count, -abs(float(weight)), float(weight)))
        _, _, weight = max(trials)
        train_deltas = []
        for scores, base in zip(score_sets, train_base):
            adjustment = np.zeros_like(train_feature)
            adjustment[train_rows] = weight * train_feature[train_rows]
            train_deltas.append(
                accuracy(
                    train_labels,
                    select(
                        train_bank, scores + adjustment, train_rejected
                    ),
                )
                - base
            )
        validation_adjustment = np.zeros_like(validation_feature)
        validation_adjustment[validation_rows] = (
            weight * validation_feature[validation_rows]
        )
        validation_delta = accuracy(
            validation_labels,
            select(
                validation_bank,
                validation_scores + validation_adjustment,
                validation_rejected,
            ),
        ) - validation_base
        print(
            f"structure={structure} weight={weight:+.3f} "
            f"train_deltas={train_deltas} validation_delta={validation_delta:+d}",
            flush=True,
        )


if __name__ == "__main__":
    main()
