"""Test question-number calculation coverage by answer structure."""

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
from cascade_verifier import select_after_negative_filter
from common import load_inputs
from group_ranker_probe import candidate_labels
from question_number_coverage_probe import coverage_features
from structure_text_fusion_verifier import structure_text_scores


def standardize_rows(values):
    return (values - values.mean(axis=1, keepdims=True)) / (
        values.std(axis=1, keepdims=True) + 1e-6
    )


def correct_count(bank, labels, scores):
    selected = np.asarray(select_after_negative_filter(bank, scores, 0.3))
    return int(labels[np.arange(len(labels)), selected].sum())


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
        CODE / "runs/model_n_cascade/structure_text_model.joblib"
    )
    train_text = [
        candidate["text"]
        for row in train_bank
        for candidate in row["candidates"]
    ]
    text_features = artifact["vectorizer"].transform(train_text)
    seeds = (7506, 11, 29, 47, 83)
    score_sets = []
    for seed in seeds:
        folds = (
            np.arange(len(train_bank)) % 5
            if seed == 7506
            else balanced_random_folds(len(train_bank), seed)
        )
        score_sets.append(
            model_m_oof(
                train_bank,
                train_arrays,
                train_labels,
                folds,
                text_features,
                train_structures,
            )
        )
    validation_scores = structure_text_scores(
        validation_bank,
        validation_arrays,
        CODE / "runs/model_n_cascade",
    )
    base_counts = [
        correct_count(train_bank, train_labels, scores) for scores in score_sets
    ]
    validation_base = correct_count(
        validation_bank, validation_labels, validation_scores
    )

    train_feature = standardize_rows(coverage_features(train_bank)[:, :, 3])
    validation_feature = standardize_rows(
        coverage_features(validation_bank)[:, :, 3]
    )
    weights = np.round(np.arange(-0.5, 0.5001, 0.025), 3)
    if args.focus:
        train_mask = train_structures == args.focus
        validation_mask = validation_structures == args.focus
        print(f"focused weight plateau structure={args.focus}")
        for weight in np.arange(0.0, 0.501, 0.025):
            train_adjustment = np.zeros_like(train_feature)
            train_adjustment[train_mask] = weight * train_feature[train_mask]
            train_deltas = [
                correct_count(train_bank, train_labels, scores + train_adjustment)
                - base
                for scores, base in zip(score_sets, base_counts)
            ]
            validation_adjustment = np.zeros_like(validation_feature)
            validation_adjustment[validation_mask] = (
                weight * validation_feature[validation_mask]
            )
            validation_delta = correct_count(
                validation_bank,
                validation_labels,
                validation_scores + validation_adjustment,
            ) - validation_base
            print(
                f"  weight={weight:.3f} train_deltas={train_deltas} "
                f"validation_delta={validation_delta:+d}",
                flush=True,
            )
        return

    chosen = {}
    for structure in ("3-1", "2-2", "2-1-1", "1-1-1-1"):
        train_mask = train_structures == structure
        validation_mask = validation_structures == structure
        trials = []
        for weight in weights:
            adjustment = np.zeros_like(train_feature)
            adjustment[train_mask] = weight * train_feature[train_mask]
            count = correct_count(
                train_bank, train_labels, score_sets[0] + adjustment
            )
            trials.append((count, -abs(float(weight)), float(weight)))
        _, _, best_weight = max(trials)
        chosen[structure] = best_weight
        train_deltas = []
        for scores, base in zip(score_sets, base_counts):
            adjustment = np.zeros_like(train_feature)
            adjustment[train_mask] = best_weight * train_feature[train_mask]
            train_deltas.append(
                correct_count(train_bank, train_labels, scores + adjustment) - base
            )
        validation_adjustment = np.zeros_like(validation_feature)
        validation_adjustment[validation_mask] = (
            best_weight * validation_feature[validation_mask]
        )
        validation_delta = correct_count(
            validation_bank,
            validation_labels,
            validation_scores + validation_adjustment,
        ) - validation_base
        print(
            f"structure={structure} weight={best_weight:+.3f} "
            f"train_deltas={train_deltas} validation_delta={validation_delta:+d}",
            flush=True,
        )

    combined_train = np.zeros_like(train_feature)
    combined_validation = np.zeros_like(validation_feature)
    for structure, weight in chosen.items():
        combined_train[train_structures == structure] = (
            weight * train_feature[train_structures == structure]
        )
        combined_validation[validation_structures == structure] = (
            weight * validation_feature[validation_structures == structure]
        )
    combined_deltas = [
        correct_count(train_bank, train_labels, scores + combined_train) - base
        for scores, base in zip(score_sets, base_counts)
    ]
    combined_validation_delta = correct_count(
        validation_bank,
        validation_labels,
        validation_scores + combined_validation,
    ) - validation_base
    print(
        f"combined_weights={chosen} train_deltas={combined_deltas} "
        f"validation_delta={combined_validation_delta:+d}",
        flush=True,
    )

    print("2-1-1 weight plateau")
    train_mask = train_structures == "2-1-1"
    validation_mask = validation_structures == "2-1-1"
    for weight in np.arange(0.10, 0.651, 0.05):
        train_adjustment = np.zeros_like(train_feature)
        train_adjustment[train_mask] = weight * train_feature[train_mask]
        train_deltas = [
            correct_count(train_bank, train_labels, scores + train_adjustment) - base
            for scores, base in zip(score_sets, base_counts)
        ]
        validation_adjustment = np.zeros_like(validation_feature)
        validation_adjustment[validation_mask] = (
            weight * validation_feature[validation_mask]
        )
        validation_delta = correct_count(
            validation_bank,
            validation_labels,
            validation_scores + validation_adjustment,
        ) - validation_base
        print(
            f"  weight={weight:.2f} train_deltas={train_deltas} "
            f"validation_delta={validation_delta:+d}",
            flush=True,
        )

    print("1-1-1-1 weight plateau")
    train_mask = train_structures == "1-1-1-1"
    validation_mask = validation_structures == "1-1-1-1"
    for weight in np.arange(0.025, 0.301, 0.025):
        train_adjustment = np.zeros_like(train_feature)
        train_adjustment[train_mask] = weight * train_feature[train_mask]
        train_deltas = [
            correct_count(train_bank, train_labels, scores + train_adjustment) - base
            for scores, base in zip(score_sets, base_counts)
        ]
        validation_adjustment = np.zeros_like(validation_feature)
        validation_adjustment[validation_mask] = (
            weight * validation_feature[validation_mask]
        )
        validation_delta = correct_count(
            validation_bank,
            validation_labels,
            validation_scores + validation_adjustment,
        ) - validation_base
        print(
            f"  weight={weight:.3f} train_deltas={train_deltas} "
            f"validation_delta={validation_delta:+d}",
            flush=True,
        )


if __name__ == "__main__":
    main()
