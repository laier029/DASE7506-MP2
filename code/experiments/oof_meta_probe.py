"""Probe a low-dimensional meta-ranker using out-of-fold embedding scores."""

from collections import Counter
from fractions import Fraction
from pathlib import Path
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))

from answer_step_consistency import answer_step_consistency
from arithmetic_consistency import arithmetic_counts
from common import DATA, load_inputs, parse_answer, read_jsonl
from linear_verifier import linear_scores


def candidate_labels(split, bank):
    references = read_jsonl(DATA / f"gsm8k_{split}.jsonl")
    return np.asarray(
        [
            [
                parse_answer(candidate["text"]) == parse_answer(reference["answer"])
                for candidate in row["candidates"]
            ]
            for row, reference in zip(bank, references)
        ],
        dtype=np.float32,
    )


def fit_ridge_scores(train_x, train_y, score_x, regularization):
    shape = score_x.shape[:2]
    train_x = train_x.reshape(-1, train_x.shape[-1])
    score_x = score_x.reshape(-1, score_x.shape[-1])
    train_y = train_y.reshape(-1)
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    train_x = (train_x - mean) / scale
    score_x = (score_x - mean) / scale
    gram = train_x.T @ train_x / len(train_x)
    target = train_x.T @ train_y / len(train_x)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    weights = eigenvectors @ (
        (eigenvectors.T @ target) / (eigenvalues + regularization)
    )
    return (score_x @ weights).reshape(shape)


def out_of_fold_scores(embeddings, labels, *, relative, regularization, folds=5):
    features = embeddings.astype(np.float32, copy=True)
    if relative:
        features -= features.mean(axis=1, keepdims=True)
    output = np.empty(labels.shape, dtype=np.float32)
    fold_ids = np.arange(len(features)) % folds
    for fold in range(folds):
        held_out = fold_ids == fold
        output[held_out] = fit_ridge_scores(
            features[~held_out],
            labels[~held_out],
            features[held_out],
            regularization,
        )
    return output


def engineered_features(bank, raw_scores, relative_scores):
    rows = []
    for row, raw_row, relative_row in zip(bank, raw_scores, relative_scores):
        answers = [parse_answer(candidate["text"]) for candidate in row["candidates"]]
        counts = Counter(answer for answer in answers if answer is not None)
        group_indices = {
            answer: [index for index, value in enumerate(answers) if value == answer]
            for answer in counts
        }
        for index, candidate in enumerate(row["candidates"]):
            answer = answers[index]
            support = counts.get(answer, 0) / 4 if answer is not None else 0.0
            indices = group_indices.get(answer, [index])
            checked, incorrect = arithmetic_counts(candidate["text"])
            step_state = answer_step_consistency(candidate["text"])
            try:
                answer_magnitude = np.log1p(abs(float(Fraction(answer))))
            except (TypeError, ValueError, ZeroDivisionError):
                answer_magnitude = 0.0
            text = candidate["text"]
            relative_centered = relative_row[index] - relative_row.mean()
            raw_centered = raw_row[index] - raw_row.mean()
            group_mean = float(relative_row[indices].mean())
            group_max = float(relative_row[indices].max())
            rows.append(
                [
                    relative_centered,
                    raw_centered,
                    support,
                    group_mean,
                    group_max,
                    relative_centered * support,
                    raw_centered * support,
                    float(checked),
                    float(incorrect),
                    float(incorrect > 0),
                    float(step_state is True),
                    float(step_state is False),
                    float(step_state is None),
                    np.log1p(len(text)),
                    np.log1p(text.count("=")),
                    float(answer is None),
                    answer_magnitude,
                    float(index == 0),
                    float(index == 1),
                    float(index == 2),
                    float(index == 3),
                ]
            )
    return np.asarray(rows, dtype=np.float32)


def group_select(bank, scores, mean_weight):
    selected = []
    for row, row_scores in zip(bank, scores):
        groups = {}
        for index, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate["text"])
            key = ("missing", index) if answer is None else ("answer", answer)
            groups.setdefault(key, []).append(index)
        best = None
        for indices in groups.values():
            values = row_scores[indices]
            candidate = indices[int(values.argmax())]
            item = (float(values.max() + mean_weight * values.mean()), -candidate)
            if best is None or item > best[0]:
                best = (item, candidate)
        selected.append(best[1])
    return np.asarray(selected)


def main():
    train_bank, train_arrays = load_inputs("train")
    val_bank, val_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank)
    val_labels = candidate_labels("validation", val_bank)

    oof_raw = out_of_fold_scores(
        train_arrays["embeddings"], train_labels, relative=False, regularization=0.1
    )
    oof_relative = out_of_fold_scores(
        train_arrays["embeddings"], train_labels, relative=True, regularization=0.3
    )
    val_raw = linear_scores(
        val_bank, val_arrays, CODE / "runs/model_c_raw", relative=False
    )
    val_relative = linear_scores(
        val_bank, val_arrays, CODE / "runs/model_h_group_aggregate", relative=True
    )

    train_meta = engineered_features(train_bank, oof_raw, oof_relative)
    val_meta = engineered_features(val_bank, val_raw, val_relative)
    mean = train_meta.mean(axis=0)
    scale = train_meta.std(axis=0)
    scale[scale < 1e-6] = 1.0
    train_meta = (train_meta - mean) / scale
    val_meta = (val_meta - mean) / scale
    train_target = train_labels.reshape(-1)

    gram = train_meta.T @ train_meta / len(train_meta)
    target = train_meta.T @ train_target / len(train_meta)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    projected = eigenvectors.T @ target
    for regularization in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10):
        weights = eigenvectors @ (projected / (eigenvalues + regularization))
        train_scores = (train_meta @ weights).reshape(-1, 4)
        val_scores = (val_meta @ weights).reshape(-1, 4)
        train_selected = train_scores.argmax(axis=1)
        val_selected = val_scores.argmax(axis=1)
        train_accuracy = train_labels[
            np.arange(len(train_labels)), train_selected
        ].mean()
        val_accuracy = val_labels[np.arange(len(val_labels)), val_selected].mean()
        group_results = []
        for mean_weight in (0.1, 0.2, 0.3, 0.4):
            grouped = group_select(val_bank, val_scores, mean_weight)
            grouped_accuracy = val_labels[
                np.arange(len(val_labels)), grouped
            ].mean()
            group_results.append(f"g{mean_weight:.1f}={100 * grouped_accuracy:.4f}")
        print(
            f"lambda={regularization:g} "
            f"oof_train={100 * train_accuracy:.4f} "
            f"validation={100 * val_accuracy:.4f} " + " ".join(group_results)
        )


if __name__ == "__main__":
    main()
