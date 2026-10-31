"""Probe pairwise margin ranking on within-question embedding differences."""

from pathlib import Path
import sys

import numpy as np
from sklearn.svm import LinearSVC


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from common import load_inputs
from embedding_classifier_probe import candidate_labels, cross_fitted_scores, normalize
from linear_verifier import linear_scores
from parseable_group_verifier import select_parseable_groups
from reasoning_density_verifier import reasoning_density_scores


def pairwise_data(features, labels, row_mask):
    differences = []
    for row_index in np.flatnonzero(row_mask):
        correct = np.flatnonzero(labels[row_index])
        wrong = np.flatnonzero(~labels[row_index])
        for correct_index in correct:
            for wrong_index in wrong:
                differences.append(
                    features[row_index, correct_index]
                    - features[row_index, wrong_index]
                )
    positive = np.asarray(differences, dtype=np.float32)
    x = np.concatenate([positive, -positive], axis=0)
    y = np.concatenate(
        [np.ones(len(positive), dtype=np.int8), np.zeros(len(positive), dtype=np.int8)]
    )
    return x, y


def pairwise_scores(train_embeddings, validation_embeddings, labels, folds, c_value):
    oof = np.empty(labels.shape, dtype=np.float32)
    for fold in range(5):
        fit_rows = folds != fold
        held_rows = ~fit_rows
        fit_flat = train_embeddings[fit_rows].reshape(-1, 1024)
        mean = fit_flat.mean(axis=0)
        scale = fit_flat.std(axis=0)
        scale[scale < 1e-6] = 1.0
        normalized = (train_embeddings - mean) / scale
        x, y = pairwise_data(normalized, labels, fit_rows)
        classifier = LinearSVC(
            C=c_value,
            fit_intercept=False,
            dual="auto",
            max_iter=10000,
            random_state=7506,
        ).fit(x, y)
        oof[held_rows] = classifier.decision_function(
            normalized[held_rows].reshape(-1, 1024)
        ).reshape(-1, 4)

    fit_flat = train_embeddings.reshape(-1, 1024)
    mean = fit_flat.mean(axis=0)
    scale = fit_flat.std(axis=0)
    scale[scale < 1e-6] = 1.0
    train_normalized = (train_embeddings - mean) / scale
    validation_normalized = (validation_embeddings - mean) / scale
    x, y = pairwise_data(
        train_normalized, labels, np.ones(len(labels), dtype=bool)
    )
    classifier = LinearSVC(
        C=c_value,
        fit_intercept=False,
        dual="auto",
        max_iter=10000,
        random_state=7506,
    ).fit(x, y)
    validation = classifier.decision_function(
        validation_normalized.reshape(-1, 1024)
    ).reshape(-1, 4)
    return oof, validation


def accuracy(bank, labels, scores):
    selected = np.asarray(select_parseable_groups(bank, scores, 0.3))
    return labels[np.arange(len(labels)), selected].mean()


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank)
    validation_labels = candidate_labels("validation", validation_bank)
    folds = np.arange(len(train_bank)) % 5
    train_embeddings = train_arrays["embeddings"].astype(np.float32)
    validation_embeddings = validation_arrays["embeddings"].astype(np.float32)
    train_embeddings -= train_embeddings.mean(axis=1, keepdims=True)
    validation_embeddings -= validation_embeddings.mean(axis=1, keepdims=True)

    ridge_oof, ridge_validation = cross_fitted_scores(
        train_embeddings,
        validation_embeddings,
        train_labels,
        folds,
        "ridge",
    )
    checkpoint = CODE / "runs/model_k_reasoning_density"
    full_train_ridge = linear_scores(
        train_bank, train_arrays, checkpoint, relative=True
    )
    full_validation_ridge = linear_scores(
        validation_bank, validation_arrays, checkpoint, relative=True
    )
    train_adjustment = (
        reasoning_density_scores(train_bank, train_arrays, checkpoint)
        - full_train_ridge
    )
    validation_adjustment = (
        reasoning_density_scores(
            validation_bank, validation_arrays, checkpoint
        )
        - full_validation_ridge
    )
    ridge_oof += train_adjustment
    ridge_validation += validation_adjustment
    ridge_oof, ridge_scale = normalize(ridge_oof)
    ridge_validation, _ = normalize(ridge_validation, ridge_scale)

    for c_value in (0.00003, 0.0001, 0.0003, 0.001, 0.003):
        pair_oof, pair_validation = pairwise_scores(
            train_embeddings,
            validation_embeddings,
            train_labels,
            folds,
            c_value,
        )
        pair_oof += train_adjustment
        pair_validation += validation_adjustment
        pair_oof, pair_scale = normalize(pair_oof)
        pair_validation, _ = normalize(pair_validation, pair_scale)
        best = (-1.0, 0.0)
        for weight in np.arange(0.0, 1.001, 0.1):
            fused = (1 - weight) * ridge_oof + weight * pair_oof
            score = accuracy(train_bank, train_labels, fused)
            if score > best[0]:
                best = (score, float(weight))
        validation_fused = (
            (1 - best[1]) * ridge_validation + best[1] * pair_validation
        )
        print(
            f"c={c_value:g} pair_oof={100 * accuracy(train_bank, train_labels, pair_oof):.4f} "
            f"mix={best[1]:.1f} mixed_oof={100 * best[0]:.4f} "
            f"validation={100 * accuracy(validation_bank, validation_labels, validation_fused):.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
