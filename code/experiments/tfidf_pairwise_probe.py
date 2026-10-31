"""Probe a task-aligned pairwise ranker over sparse candidate text."""

from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
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


def pairwise_examples(features, labels, row_mask):
    positive = []
    negative = []
    for row_index in np.flatnonzero(row_mask):
        correct = np.flatnonzero(labels[row_index])
        wrong = np.flatnonzero(~labels[row_index])
        for correct_index in correct:
            for wrong_index in wrong:
                positive.append(4 * row_index + correct_index)
                negative.append(4 * row_index + wrong_index)
    difference = features[positive] - features[negative]
    # Symmetric examples remove any need for an intercept and avoid label bias.
    x = sparse.vstack([difference, -difference], format="csr")
    y = np.concatenate(
        [np.ones(len(positive), dtype=np.int8), np.zeros(len(positive), dtype=np.int8)]
    )
    return x, y


def oof_k_scores(bank, arrays, labels, folds):
    embeddings = arrays["embeddings"].astype(np.float32)
    embeddings -= embeddings.mean(axis=1, keepdims=True)
    scores = np.empty(labels.shape, dtype=np.float32)
    for fold in range(folds.max() + 1):
        fit = folds != fold
        held = ~fit
        fit_x = embeddings[fit].reshape(-1, 1024)
        mean = fit_x.mean(axis=0)
        scale = fit_x.std(axis=0)
        scale[scale < 1e-6] = 1.0
        fit_x = (fit_x - mean) / scale
        fit_y = labels[fit].reshape(-1).astype(np.float32)
        gram = fit_x.T @ fit_x / len(fit_x)
        target = fit_x.T @ fit_y / len(fit_x)
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        weights = eigenvectors @ (
            (eigenvectors.T @ target) / (eigenvalues + 0.3)
        )
        scores[held] = ((embeddings[held] - mean) / scale) @ weights
    checkpoint = CODE / "runs/model_k_reasoning_density"
    full_embedding = linear_scores(bank, arrays, checkpoint, relative=True)
    model_k = reasoning_density_scores(bank, arrays, checkpoint)
    scores += model_k - full_embedding
    scores -= scores.mean(axis=1, keepdims=True)
    scale = float(scores.std())
    return scores / scale, scale


def accuracy(bank, labels, scores):
    selected = np.asarray(select_parseable_groups(bank, scores, 0.3))
    return labels[np.arange(len(labels)), selected].mean()


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank)
    validation_labels = candidate_labels("validation", validation_bank)
    folds = np.arange(len(train_bank)) % 5

    train_k, k_scale = oof_k_scores(
        train_bank, train_arrays, train_labels, folds
    )
    checkpoint = CODE / "runs/model_k_reasoning_density"
    validation_k = reasoning_density_scores(
        validation_bank, validation_arrays, checkpoint
    )
    validation_k -= validation_k.mean(axis=1, keepdims=True)
    validation_k /= k_scale

    configurations = [
        ("char35", "char", (3, 5), 25000, 5),
        ("word12", "word", (1, 2), 25000, 3),
    ]
    train_texts = [
        candidate["text"]
        for row in train_bank
        for candidate in row["candidates"]
    ]
    validation_texts = [
        candidate["text"]
        for row in validation_bank
        for candidate in row["candidates"]
    ]
    for name, analyzer, ngrams, max_features, min_df in configurations:
        vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngrams,
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=True,
            dtype=np.float32,
        )
        train_x = vectorizer.fit_transform(train_texts)
        validation_x = vectorizer.transform(validation_texts)
        for c_value in (0.001, 0.01, 0.1):
            pairwise_oof = np.empty(len(train_texts), dtype=np.float32)
            for fold in range(5):
                fit_rows = folds != fold
                held_rows = ~fit_rows
                pair_x, pair_y = pairwise_examples(
                    train_x, train_labels, fit_rows
                )
                classifier = LinearSVC(
                    C=c_value,
                    fit_intercept=False,
                    dual="auto",
                    random_state=7506,
                ).fit(pair_x, pair_y)
                held = np.repeat(held_rows, 4)
                pairwise_oof[held] = classifier.decision_function(
                    train_x[held]
                )
            pairwise_oof = pairwise_oof.reshape(-1, 4)
            pairwise_oof -= pairwise_oof.mean(axis=1, keepdims=True)
            pairwise_scale = float(pairwise_oof.std())
            pairwise_oof /= pairwise_scale

            best = (-1.0, 0.0)
            for weight in np.arange(-0.20, 0.401, 0.025):
                score = accuracy(
                    train_bank, train_labels, train_k + weight * pairwise_oof
                )
                if score > best[0]:
                    best = (score, float(weight))

            pair_x, pair_y = pairwise_examples(
                train_x, train_labels, np.ones(len(train_bank), dtype=bool)
            )
            classifier = LinearSVC(
                C=c_value,
                fit_intercept=False,
                dual="auto",
                random_state=7506,
            ).fit(pair_x, pair_y)
            validation_text = classifier.decision_function(
                validation_x
            ).reshape(-1, 4)
            validation_text -= validation_text.mean(axis=1, keepdims=True)
            validation_text /= pairwise_scale
            validation_score = accuracy(
                validation_bank,
                validation_labels,
                validation_k + best[1] * validation_text,
            )
            print(
                f"features={name} c={c_value:g} "
                f"pair_oof={100 * accuracy(train_bank, train_labels, pairwise_oof):.4f} "
                f"weight={best[1]:.3f} fused_oof={100 * best[0]:.4f} "
                f"validation={100 * validation_score:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
