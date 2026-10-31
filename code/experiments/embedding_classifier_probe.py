"""Compare ridge and margin classifiers on relative reasoning embeddings."""

from collections import Counter
from pathlib import Path
import sys

import numpy as np
from sklearn.svm import LinearSVC


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))

from common import DATA, load_inputs, parse_answer, read_jsonl
from linear_verifier import linear_scores
from parseable_group_verifier import select_parseable_groups
from reasoning_density_verifier import reasoning_density_scores


def candidate_labels(split, bank):
    references = read_jsonl(DATA / f"gsm8k_{split}.jsonl")
    return np.asarray(
        [
            [
                parse_answer(candidate["text"])
                == parse_answer(reference["answer"])
                for candidate in row["candidates"]
            ]
            for row, reference in zip(bank, references)
        ]
    )


def fit_ridge(train_x, train_y, score_x):
    gram = train_x.T @ train_x / len(train_x)
    target = train_x.T @ train_y / len(train_x)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    weights = eigenvectors @ (
        (eigenvectors.T @ target) / (eigenvalues + 0.3)
    )
    return score_x @ weights


def standardize(train_x, score_x):
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return (train_x - mean) / scale, (score_x - mean) / scale


def cross_fitted_scores(train_embeddings, validation_embeddings, labels, folds, model):
    oof = np.empty(labels.size, dtype=np.float32)
    for fold in range(folds.max() + 1):
        fit_rows = folds != fold
        held_rows = ~fit_rows
        fit_x = train_embeddings[fit_rows].reshape(-1, 1024)
        held_x = train_embeddings[held_rows].reshape(-1, 1024)
        fit_x, held_x = standardize(fit_x, held_x)
        fit_y = labels[fit_rows].reshape(-1)
        if model == "ridge":
            oof[np.repeat(held_rows, 4)] = fit_ridge(
                fit_x, fit_y.astype(np.float32), held_x
            )
        else:
            c_value, class_weight = model
            classifier = LinearSVC(
                C=c_value,
                class_weight=class_weight,
                dual="auto",
                max_iter=10000,
                random_state=7506,
            ).fit(fit_x, fit_y)
            oof[np.repeat(held_rows, 4)] = classifier.decision_function(held_x)

    fit_x = train_embeddings.reshape(-1, 1024)
    score_x = validation_embeddings.reshape(-1, 1024)
    fit_x, score_x = standardize(fit_x, score_x)
    fit_y = labels.reshape(-1)
    if model == "ridge":
        validation = fit_ridge(fit_x, fit_y.astype(np.float32), score_x)
    else:
        c_value, class_weight = model
        classifier = LinearSVC(
            C=c_value,
            class_weight=class_weight,
            dual="auto",
            max_iter=10000,
            random_state=7506,
        ).fit(fit_x, fit_y)
        validation = classifier.decision_function(score_x)
    return oof.reshape(-1, 4), validation.reshape(-1, 4)


def normalize(scores, scale=None):
    scores = scores - scores.mean(axis=1, keepdims=True)
    if scale is None:
        scale = float(scores.std())
    return scores / scale, scale


def accuracy(bank, labels, scores, group_mean_weight=0.3):
    selected = np.asarray(
        select_parseable_groups(bank, scores, group_mean_weight)
    )
    return labels[np.arange(len(labels)), selected].mean()


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_y = candidate_labels("train", train_bank)
    validation_y = candidate_labels("validation", validation_bank)
    folds = np.arange(len(train_bank)) % 5
    train_embeddings = train_arrays["embeddings"].astype(np.float32)
    validation_embeddings = validation_arrays["embeddings"].astype(np.float32)
    train_embeddings -= train_embeddings.mean(axis=1, keepdims=True)
    validation_embeddings -= validation_embeddings.mean(axis=1, keepdims=True)

    ridge_oof, ridge_validation = cross_fitted_scores(
        train_embeddings, validation_embeddings, train_y, folds, "ridge"
    )
    checkpoint = CODE / "runs/model_k_reasoning_density"
    full_train_ridge = linear_scores(
        train_bank, train_arrays, checkpoint, relative=True
    )
    full_validation_ridge = linear_scores(
        validation_bank, validation_arrays, checkpoint, relative=True
    )
    train_k = reasoning_density_scores(train_bank, train_arrays, checkpoint)
    validation_k = reasoning_density_scores(
        validation_bank, validation_arrays, checkpoint
    )
    train_adjustment = train_k - full_train_ridge
    validation_adjustment = validation_k - full_validation_ridge
    ridge_oof += train_adjustment
    ridge_validation += validation_adjustment
    ridge_oof, ridge_scale = normalize(ridge_oof)
    ridge_validation, _ = normalize(ridge_validation, ridge_scale)
    print(
        f"ridge oof={100 * accuracy(train_bank, train_y, ridge_oof):.4f} "
        f"validation={100 * accuracy(validation_bank, validation_y, ridge_validation):.4f}",
        flush=True,
    )

    configurations = [
        (0.0003, None),
        (0.001, None),
        (0.003, None),
        (0.01, None),
        (0.03, None),
        (0.1, None),
        (0.001, "balanced"),
        (0.01, "balanced"),
        (0.1, "balanced"),
    ]
    for configuration in configurations:
        svc_oof, svc_validation = cross_fitted_scores(
            train_embeddings,
            validation_embeddings,
            train_y,
            folds,
            configuration,
        )
        svc_oof += train_adjustment
        svc_validation += validation_adjustment
        svc_oof, svc_scale = normalize(svc_oof)
        svc_validation, _ = normalize(svc_validation, svc_scale)

        best = (-1.0, 0.0)
        for svc_weight in np.arange(0.0, 1.001, 0.1):
            fused = (1 - svc_weight) * ridge_oof + svc_weight * svc_oof
            score = accuracy(train_bank, train_y, fused)
            if score > best[0]:
                best = (score, float(svc_weight))
        validation_fused = (
            (1 - best[1]) * ridge_validation + best[1] * svc_validation
        )
        validation_score = accuracy(
            validation_bank, validation_y, validation_fused
        )
        print(
            f"svc_c={configuration[0]:g} class_weight={configuration[1]} "
            f"svc_oof={100 * accuracy(train_bank, train_y, svc_oof):.4f} "
            f"mix={best[1]:.1f} mixed_oof={100 * best[0]:.4f} "
            f"validation={100 * validation_score:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
