"""Probe compact histogram-boosted meta-rankers with OOF base scores."""

from collections import Counter
from pathlib import Path
import re
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))

from answer_step_consistency import answer_step_consistency
from arithmetic_consistency import arithmetic_counts, broad_arithmetic_counts
from common import DATA, load_inputs, parse_answer, read_jsonl
from linear_verifier import linear_scores
from parseable_group_verifier import select_parseable_groups
from reasoning_density_verifier import reasoning_density_scores


NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
OPERATORS = (" + ", " - ", "\\times", "×", "\\div", "÷")


def labels(split, bank):
    references = read_jsonl(DATA / f"gsm8k_{split}.jsonl")
    return np.asarray(
        [
            [
                parse_answer(candidate["text"]) == parse_answer(reference["answer"])
                for candidate in row["candidates"]
            ]
            for row, reference in zip(bank, references)
        ]
    )


def meta_features(bank, scores):
    rows = []
    for row, row_scores in zip(bank, scores):
        answers = [parse_answer(candidate["text"]) for candidate in row["candidates"]]
        counts = Counter(answer for answer in answers if answer is not None)
        for index, candidate in enumerate(row["candidates"]):
            text = candidate["text"]
            peers = (
                [peer for peer, answer in enumerate(answers) if answer == answers[index]]
                if answers[index] is not None
                else [index]
            )
            narrow_checked, narrow_error = arithmetic_counts(text)
            broad_checked, broad_error = broad_arithmetic_counts(text)
            step = answer_step_consistency(text)
            rows.append(
                [
                    row_scores[index] - row_scores.mean(),
                    counts.get(answers[index], 0) / 4
                    if answers[index] is not None
                    else 0.0,
                    row_scores[peers].mean(),
                    row_scores[peers].max(),
                    narrow_checked,
                    narrow_error,
                    broad_checked,
                    broad_error,
                    step is True,
                    step is False,
                    step is None,
                    np.log1p(len(NUMBER.findall(text))),
                    np.log1p(sum(text.count(operator) for operator in OPERATORS)),
                    np.log1p(len(text)),
                    np.log1p(text.count("=")),
                    index == 0,
                    index == 1,
                    index == 2,
                    index == 3,
                ]
            )
    return np.asarray(rows, dtype=np.float32)


def main():
    train_bank, train_arrays = load_inputs("train")
    val_bank, val_arrays = load_inputs("validation")
    train_y = labels("train", train_bank)
    val_y = labels("validation", val_bank)
    train_x = train_arrays["embeddings"].astype(np.float32)
    val_x = val_arrays["embeddings"].astype(np.float32)
    train_x -= train_x.mean(axis=1, keepdims=True)
    val_x -= val_x.mean(axis=1, keepdims=True)
    flat = train_x.reshape(-1, 1024)
    mean = flat.mean(axis=0)
    scale = flat.std(axis=0)
    scale[scale < 1e-6] = 1.0
    train_x = (train_x - mean) / scale
    val_x = (val_x - mean) / scale

    oof = np.empty((len(train_bank), 4), dtype=np.float32)
    fold_ids = np.arange(len(train_bank)) % 5
    for fold in range(5):
        train_mask = fold_ids != fold
        held_out = ~train_mask
        features = train_x[train_mask].reshape(-1, 1024)
        targets = train_y[train_mask].reshape(-1).astype(np.float32)
        gram = features.T @ features / len(features)
        target = features.T @ targets / len(features)
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        weights = eigenvectors @ (
            (eigenvectors.T @ target) / (eigenvalues + 0.3)
        )
        oof[held_out] = train_x[held_out] @ weights

    full_train = linear_scores(
        train_bank, train_arrays, CODE / "runs/model_k_reasoning_density", relative=True
    )
    full_val = linear_scores(
        val_bank, val_arrays, CODE / "runs/model_k_reasoning_density", relative=True
    )
    train_k = reasoning_density_scores(
        train_bank, train_arrays, CODE / "runs/model_k_reasoning_density"
    )
    val_k = reasoning_density_scores(
        val_bank, val_arrays, CODE / "runs/model_k_reasoning_density"
    )
    oof_k = oof + (train_k - full_train)

    covariance = train_x.reshape(-1, 1024).T @ train_x.reshape(-1, 1024)
    covariance /= len(train_x) * 4
    _, eigenvectors = np.linalg.eigh(covariance)
    projection = eigenvectors[:, -64:]
    train_pca = (train_x @ projection).reshape(-1, 64)
    val_pca = (val_x @ projection).reshape(-1, 64)
    train_meta = meta_features(train_bank, oof_k)
    val_meta = meta_features(val_bank, val_k)

    base = np.asarray(select_parseable_groups(val_bank, val_k, 0.3))
    print(
        f"base={100 * val_y[np.arange(len(val_y)), base].mean():.4f}",
        flush=True,
    )
    for use_pca in (False, True):
        train_features = (
            np.concatenate([train_meta, train_pca], axis=1)
            if use_pca
            else train_meta
        )
        val_features = (
            np.concatenate([val_meta, val_pca], axis=1) if use_pca else val_meta
        )
        for leaves in (3, 7, 15):
            for iterations in (30, 60, 120):
                classifier = HistGradientBoostingClassifier(
                    max_iter=iterations,
                    learning_rate=0.05,
                    max_leaf_nodes=leaves,
                    min_samples_leaf=50,
                    l2_regularization=3,
                    random_state=7506,
                ).fit(train_features, train_y.reshape(-1))
                tree_scores = classifier.predict_proba(val_features)[:, 1].reshape(-1, 4)
                alone = np.asarray(
                    select_parseable_groups(val_bank, tree_scores, 0.3)
                )
                alone_accuracy = val_y[np.arange(len(val_y)), alone].mean()
                k_centered = val_k - val_k.mean(axis=1, keepdims=True)
                tree_centered = tree_scores - tree_scores.mean(axis=1, keepdims=True)
                k_centered /= k_centered.std()
                tree_centered /= tree_centered.std()
                best = (0.0, 0.0)
                for weight in np.arange(-0.2, 0.501, 0.025):
                    selected = np.asarray(
                        select_parseable_groups(
                            val_bank, k_centered + weight * tree_centered, 0.3
                        )
                    )
                    accuracy = val_y[np.arange(len(val_y)), selected].mean()
                    if accuracy > best[0]:
                        best = (accuracy, weight)
                print(
                    f"pca={use_pca} leaves={leaves} iterations={iterations} "
                    f"alone={100 * alone_accuracy:.4f} "
                    f"fused={100 * best[0]:.4f} weight={best[1]:.3f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
