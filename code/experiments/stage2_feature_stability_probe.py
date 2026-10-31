"""Screen survivor-scoring features on fixed and random Model M OOF folds."""

from pathlib import Path
import sys

import joblib
import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from answer_support_probe import answer_support_features
from cascade_stability_probe import (
    answer_structures,
    balanced_random_folds,
    model_m_oof,
)
from cascade_verifier import select_after_negative_filter
from common import load_inputs
from group_ranker_probe import candidate_labels, candidate_static_features
from intermediate_agreement_probe import agreement_features
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
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)

    artifact = joblib.load(
        CODE / "runs/model_n_cascade/structure_text_model.joblib"
    )
    train_text = [
        candidate["text"]
        for row in train_bank
        for candidate in row["candidates"]
    ]
    text_features = artifact["vectorizer"].transform(train_text)
    structures = answer_structures(train_bank)

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
                structures,
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
    print(
        f"base_train={base_counts} validation={validation_base}", flush=True
    )

    feature_families = (
        (
            "coverage",
            coverage_features(train_bank),
            coverage_features(validation_bank),
            (
                "body_occurrence",
                "calculation_occurrence",
                "body_unique",
                "calculation_unique",
                "calculation_number_count",
            ),
        ),
        (
            "answer_support",
            answer_support_features(train_bank),
            answer_support_features(validation_bank),
            (
                "any_equation_support",
                "equation_support_fraction",
                "last_equation_support",
                "any_body_number_support",
                "body_number_support_fraction",
                "last_body_number_support",
                "body_support_count",
            ),
        ),
        (
            "agreement",
            agreement_features(train_bank),
            agreement_features(validation_bank),
            (
                "supported_result_fraction",
                "supported_result_count",
                "mean_peer_support",
                "maximum_peer_jaccard",
                "mean_peer_jaccard",
                "final_answer_peer_support_count",
                "final_answer_any_peer_support",
                "nonfinal_supported_fraction",
            ),
        ),
        (
            "static",
            candidate_static_features(train_bank),
            candidate_static_features(validation_bank),
            (
                "text_length",
                "number_count",
                "operation_count",
                "narrow_checked",
                "narrow_error",
                "broad_checked",
                "broad_error",
                "step_match",
                "step_mismatch",
                "step_missing",
            ),
        ),
    )

    weights = np.round(np.arange(-0.5, 0.5001, 0.025), 3)
    for family, train_raw, validation_raw, names in feature_families:
        for feature_index, name in enumerate(names):
            train_feature = standardize_rows(train_raw[:, :, feature_index])
            validation_feature = standardize_rows(
                validation_raw[:, :, feature_index]
            )
            trials = []
            for weight in weights:
                count = correct_count(
                    train_bank,
                    train_labels,
                    score_sets[0] + weight * train_feature,
                )
                trials.append((count, -abs(float(weight)), float(weight)))
            _, _, best_weight = max(trials)
            deltas = [
                correct_count(
                    train_bank,
                    train_labels,
                    scores + best_weight * train_feature,
                )
                - base
                for scores, base in zip(score_sets, base_counts)
            ]
            validation_delta = correct_count(
                validation_bank,
                validation_labels,
                validation_scores + best_weight * validation_feature,
            ) - validation_base
            print(
                f"feature={family}.{name} weight={best_weight:+.3f} "
                f"train_deltas={deltas} validation_delta={validation_delta:+d}",
                flush=True,
            )


if __name__ == "__main__":
    main()
