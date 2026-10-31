"""OOF selection of compact TF-IDF configurations and fusion weights."""

from pathlib import Path
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))

from common import DATA, load_inputs, parse_answer, read_jsonl
from linear_verifier import linear_scores
from parseable_group_verifier import select_parseable_groups
from reasoning_density_verifier import reasoning_density_scores


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


def main():
    train_bank, train_arrays = load_inputs("train")
    val_bank, val_arrays = load_inputs("validation")
    train_y = labels("train", train_bank)
    val_y = labels("validation", val_bank)
    train_text = [c["text"] for row in train_bank for c in row["candidates"]]
    val_text = [c["text"] for row in val_bank for c in row["candidates"]]
    folds = np.arange(len(train_bank)) % 5

    embeddings = train_arrays["embeddings"].astype(np.float32)
    embeddings -= embeddings.mean(axis=1, keepdims=True)
    embedding_oof = np.empty((len(train_bank), 4), dtype=np.float32)
    for fold in range(5):
        fit = folds != fold
        held_out = ~fit
        fit_x = embeddings[fit].reshape(-1, 1024)
        mean = fit_x.mean(axis=0)
        scale = fit_x.std(axis=0)
        scale[scale < 1e-6] = 1.0
        fit_x = (fit_x - mean) / scale
        fit_y = train_y[fit].reshape(-1).astype(np.float32)
        gram = fit_x.T @ fit_x / len(fit_x)
        target = fit_x.T @ fit_y / len(fit_x)
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        weights = eigenvectors @ (
            (eigenvectors.T @ target) / (eigenvalues + 0.3)
        )
        embedding_oof[held_out] = (
            (embeddings[held_out] - mean) / scale
        ) @ weights

    full_embedding = linear_scores(
        train_bank,
        train_arrays,
        CODE / "runs/model_k_reasoning_density",
        relative=True,
    )
    train_k = reasoning_density_scores(
        train_bank, train_arrays, CODE / "runs/model_k_reasoning_density"
    )
    val_k = reasoning_density_scores(
        val_bank, val_arrays, CODE / "runs/model_k_reasoning_density"
    )
    embedding_oof += train_k - full_embedding
    embedding_oof -= embedding_oof.mean(axis=1, keepdims=True)
    embedding_scale = embedding_oof.std()
    embedding_oof /= embedding_scale
    val_k = val_k - val_k.mean(axis=1, keepdims=True)
    val_k /= embedding_scale

    configurations = [
        ("char35_25k", "char", (3, 5), 25000, 5, 0.1),
        ("char35_50k", "char", (3, 5), 50000, 5, 0.1),
        ("char35_100k", "char", (3, 5), 100000, 5, 0.1),
        ("char24_50k", "char", (2, 4), 50000, 5, 0.1),
        ("char46_50k", "char", (4, 6), 50000, 5, 0.1),
        ("word12_50k", "word", (1, 2), 50000, 3, 0.03),
    ]
    for name, analyzer, ngrams, max_features, min_df, c_value in configurations:
        vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngrams,
            max_features=max_features,
            min_df=min_df,
            sublinear_tf=True,
            dtype=np.float32,
        )
        train_features = vectorizer.fit_transform(train_text)
        val_features = vectorizer.transform(val_text)
        text_oof = np.empty(len(train_text), dtype=np.float32)
        for fold in range(5):
            row_fit = folds != fold
            row_val = ~row_fit
            fit = np.repeat(row_fit, 4)
            held_out = np.repeat(row_val, 4)
            classifier = LinearSVC(
                C=c_value, dual="auto", random_state=7506
            ).fit(train_features[fit], train_y.reshape(-1)[fit])
            text_oof[held_out] = classifier.decision_function(
                train_features[held_out]
            )
        text_oof = text_oof.reshape(-1, 4)
        text_oof -= text_oof.mean(axis=1, keepdims=True)
        text_scale = text_oof.std()
        text_oof /= text_scale

        best = (0.0, 0.0)
        for weight in np.arange(-0.2, 0.201, 0.025):
            selected = np.asarray(
                select_parseable_groups(
                    train_bank, embedding_oof + weight * text_oof, 0.3
                )
            )
            accuracy = train_y[np.arange(len(train_y)), selected].mean()
            if accuracy > best[0]:
                best = (accuracy, weight)

        classifier = LinearSVC(C=c_value, dual="auto", random_state=7506).fit(
            train_features, train_y.reshape(-1)
        )
        val_text_scores = classifier.decision_function(val_features).reshape(-1, 4)
        val_text_scores -= val_text_scores.mean(axis=1, keepdims=True)
        val_text_scores /= text_scale
        val_selected = np.asarray(
            select_parseable_groups(
                val_bank, val_k + best[1] * val_text_scores, 0.3
            )
        )
        val_accuracy = val_y[np.arange(len(val_y)), val_selected].mean()
        print(
            f"{name} features={train_features.shape[1]} "
            f"oof={100 * best[0]:.4f} weight={best[1]:.3f} "
            f"validation={100 * val_accuracy:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
