"""Probe downweighting all-correct questions in relative ridge training."""

from pathlib import Path
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from group_ranker_probe import candidate_labels
from common import load_inputs
from linear_verifier import linear_scores
from parseable_group_verifier import select_parseable_groups
from reasoning_density_verifier import reasoning_density_scores


def sufficient_statistics(features, labels, question_weights):
    x = features.reshape(-1, features.shape[-1])
    y = labels.reshape(-1).astype(np.float32)
    weights = np.repeat(question_weights.astype(np.float32), 4)
    total_weight = weights.sum()
    gram = x.T @ (x * weights[:, None]) / total_weight
    target = x.T @ (y * weights) / total_weight
    return np.linalg.eigh(gram), target


def ridge_scores(eigendecomposition, target, score_x, regularization):
    eigenvalues, eigenvectors = eigendecomposition
    weights = eigenvectors @ (
        (eigenvectors.T @ target) / (eigenvalues + regularization)
    )
    return score_x @ weights


def accuracy(bank, labels, scores):
    selected = np.asarray(select_parseable_groups(bank, scores, 0.3))
    return labels[np.arange(len(labels)), selected].mean()


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    folds = np.arange(len(train_bank)) % 5
    train_x = train_arrays["embeddings"].astype(np.float32)
    validation_x = validation_arrays["embeddings"].astype(np.float32)
    train_x -= train_x.mean(axis=1, keepdims=True)
    validation_x -= validation_x.mean(axis=1, keepdims=True)

    checkpoint = CODE / "runs/model_k_reasoning_density"
    full_old_train = linear_scores(
        train_bank, train_arrays, checkpoint, relative=True
    )
    full_old_validation = linear_scores(
        validation_bank, validation_arrays, checkpoint, relative=True
    )
    train_adjustment = (
        reasoning_density_scores(train_bank, train_arrays, checkpoint)
        - full_old_train
    )
    validation_adjustment = (
        reasoning_density_scores(
            validation_bank, validation_arrays, checkpoint
        )
        - full_old_validation
    )
    informative = train_labels.sum(axis=1) < 4
    print(
        f"informative_questions={informative.sum()} "
        f"all_correct_questions={(~informative).sum()}",
        flush=True,
    )

    for easy_weight in (0.0, 0.1, 0.25, 0.5, 1.0):
        oof_by_regularization = {
            value: np.empty(train_labels.shape, dtype=np.float32)
            for value in (0.03, 0.1, 0.3, 1.0, 3.0)
        }
        for fold in range(5):
            fit = folds != fold
            held = ~fit
            fit_flat = train_x[fit].reshape(-1, 1024)
            mean = fit_flat.mean(axis=0)
            scale = fit_flat.std(axis=0)
            scale[scale < 1e-6] = 1.0
            fit_x = (train_x[fit] - mean) / scale
            held_x = (train_x[held] - mean) / scale
            question_weights = np.where(
                informative[fit], 1.0, easy_weight
            ).astype(np.float32)
            decomposition, target = sufficient_statistics(
                fit_x, train_labels[fit], question_weights
            )
            for regularization, output in oof_by_regularization.items():
                output[held] = ridge_scores(
                    decomposition,
                    target,
                    held_x.reshape(-1, 1024),
                    regularization,
                ).reshape(-1, 4)

        flat = train_x.reshape(-1, 1024)
        mean = flat.mean(axis=0)
        scale = flat.std(axis=0)
        scale[scale < 1e-6] = 1.0
        normalized_train = (train_x - mean) / scale
        normalized_validation = (validation_x - mean) / scale
        question_weights = np.where(
            informative, 1.0, easy_weight
        ).astype(np.float32)
        decomposition, target = sufficient_statistics(
            normalized_train, train_labels, question_weights
        )
        for regularization, oof in oof_by_regularization.items():
            oof += train_adjustment
            validation_scores = ridge_scores(
                decomposition,
                target,
                normalized_validation.reshape(-1, 1024),
                regularization,
            ).reshape(-1, 4)
            validation_scores += validation_adjustment
            print(
                f"easy_weight={easy_weight:g} lambda={regularization:g} "
                f"oof={100 * accuracy(train_bank, train_labels, oof):.4f} "
                f"validation={100 * accuracy(validation_bank, validation_labels, validation_scores):.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
