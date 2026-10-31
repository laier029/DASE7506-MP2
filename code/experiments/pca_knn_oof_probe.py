"""Evaluate compact PCA k-nearest-neighbour signals with question-level OOF."""

from pathlib import Path
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from common import load_inputs
from group_ranker_probe import candidate_labels, oof_model_k
from parseable_group_verifier import select_parseable_groups
from structure_text_fusion_verifier import structure_text_scores


def neighbour_scores(query, reference, reference_y, neighbours, metric):
    outputs = {
        count: np.empty(len(query), dtype=np.float32) for count in neighbours
    }
    max_neighbours = max(neighbours)
    if metric == "cosine":
        query = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-6)
        reference = reference / (
            np.linalg.norm(reference, axis=1, keepdims=True) + 1e-6
        )
        reference_norm = None
    else:
        reference_norm = (reference * reference).sum(axis=1)
    for start in range(0, len(query), 256):
        stop = min(start + 256, len(query))
        similarity = query[start:stop] @ reference.T
        if metric == "euclidean":
            similarity = 2 * similarity - reference_norm[None, :]
        top = np.argpartition(
            similarity, -max_neighbours, axis=1
        )[:, -max_neighbours:]
        top_similarity = np.take_along_axis(similarity, top, axis=1)
        order = np.argsort(top_similarity, axis=1)[:, ::-1]
        top = np.take_along_axis(top, order, axis=1)
        top_y = reference_y[top]
        cumulative = np.cumsum(top_y, axis=1)
        for count in neighbours:
            outputs[count][start:stop] = cumulative[:, count - 1] / count
    return outputs


def normalize_rows(values, scale=None):
    values = values - values.mean(axis=1, keepdims=True)
    if scale is None:
        scale = float(values.std())
    return values / (scale + 1e-6), scale


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
    flat = train_x.reshape(-1, 1024)
    mean = flat.mean(axis=0)
    scale = flat.std(axis=0)
    scale[scale < 1e-6] = 1.0
    flat = (flat - mean) / scale
    validation_flat = (validation_x.reshape(-1, 1024) - mean) / scale
    covariance = flat.T @ flat / len(flat)
    _, eigenvectors = np.linalg.eigh(covariance)
    projection = eigenvectors[:, -64:].astype(np.float32)
    train_projection = flat @ projection
    validation_projection = validation_flat @ projection

    train_k = oof_model_k(train_bank, train_arrays, train_labels, folds)
    train_k, k_scale = normalize_rows(train_k)
    validation_m = structure_text_scores(
        validation_bank,
        validation_arrays,
        CODE / "runs/model_m_structure_text",
    )
    del k_scale
    neighbours = (5, 10, 20, 50, 100, 200)
    for metric in ("cosine", "euclidean"):
        oof = {
            count: np.empty(train_labels.size, dtype=np.float32)
            for count in neighbours
        }
        for fold in range(5):
            fit_rows = folds != fold
            held_rows = ~fit_rows
            fit = np.repeat(fit_rows, 4)
            held = np.repeat(held_rows, 4)
            fold_scores = neighbour_scores(
                train_projection[held],
                train_projection[fit],
                train_labels.reshape(-1)[fit].astype(np.float32),
                neighbours,
                metric,
            )
            for count in neighbours:
                oof[count][held] = fold_scores[count]
        validation = neighbour_scores(
            validation_projection,
            train_projection,
            train_labels.reshape(-1).astype(np.float32),
            neighbours,
            metric,
        )
        for count in neighbours:
            train_score = oof[count].reshape(-1, 4)
            validation_score = validation[count].reshape(-1, 4)
            train_score, score_scale = normalize_rows(train_score)
            validation_score, _ = normalize_rows(validation_score, score_scale)
            best = (-1.0, 0.0)
            for weight in np.arange(-0.30, 0.301, 0.025):
                score = accuracy(
                    train_bank,
                    train_labels,
                    train_k + weight * train_score,
                )
                if score > best[0]:
                    best = (score, float(weight))
            validation_accuracy = accuracy(
                validation_bank,
                validation_labels,
                validation_m + best[1] * validation_score,
            )
            print(
                f"metric={metric} k={count} "
                f"knn_oof={100 * accuracy(train_bank, train_labels, train_score):.4f} "
                f"weight={best[1]:.3f} fused_oof={100 * best[0]:.4f} "
                f"validation={100 * validation_accuracy:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
