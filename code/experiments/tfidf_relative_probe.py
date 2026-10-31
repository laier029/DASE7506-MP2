"""Compare absolute and within-question relative TF-IDF verifier signals."""

from collections import Counter
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


def relative_rows(features):
    """Subtract each question's mean sparse text vector from its four rows."""
    if features.shape[0] % 4:
        raise ValueError("Expected four candidate rows per question")
    group_mean = sum(features[offset::4] for offset in range(4)) * 0.25
    repeated_mean = group_mean[np.repeat(np.arange(group_mean.shape[0]), 4)]
    return (features - repeated_mean).tocsr()


def answer_structures(bank):
    return [
        "-".join(
            str(value)
            for value in sorted(
                Counter(
                    parse_answer(candidate["text"])
                    for candidate in row["candidates"]
                ).values(),
                reverse=True,
            )
        )
        for row in bank
    ]


def apply_structure_weights(scores, text_scores, structures, weights):
    result = scores.copy()
    for index, structure in enumerate(structures):
        result[index] += weights.get(structure, 0.0) * text_scores[index]
    return result


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank)
    validation_labels = candidate_labels("validation", validation_bank)
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

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=5,
        max_features=25000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    train_absolute = vectorizer.fit_transform(train_texts)
    validation_absolute = vectorizer.transform(validation_texts)
    feature_sets = {
        "absolute": (train_absolute, validation_absolute),
        "relative": (
            relative_rows(train_absolute),
            relative_rows(validation_absolute),
        ),
        "absolute_plus_relative": (
            sparse.hstack(
                [train_absolute, relative_rows(train_absolute)], format="csr"
            ),
            sparse.hstack(
                [validation_absolute, relative_rows(validation_absolute)],
                format="csr",
            ),
        ),
    }

    folds = np.arange(len(train_bank)) % 5
    embeddings = train_arrays["embeddings"].astype(np.float32)
    embeddings -= embeddings.mean(axis=1, keepdims=True)
    embedding_oof = np.empty((len(train_bank), 4), dtype=np.float32)
    for fold in range(5):
        fit_rows = folds != fold
        held_rows = ~fit_rows
        fit_x = embeddings[fit_rows].reshape(-1, 1024)
        mean = fit_x.mean(axis=0)
        scale = fit_x.std(axis=0)
        scale[scale < 1e-6] = 1.0
        fit_x = (fit_x - mean) / scale
        fit_y = train_labels[fit_rows].reshape(-1).astype(np.float32)
        gram = fit_x.T @ fit_x / len(fit_x)
        target = fit_x.T @ fit_y / len(fit_x)
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        weights = eigenvectors @ (
            (eigenvectors.T @ target) / (eigenvalues + 0.3)
        )
        embedding_oof[held_rows] = (
            (embeddings[held_rows] - mean) / scale
        ) @ weights

    checkpoint = CODE / "runs/model_k_reasoning_density"
    full_embedding = linear_scores(
        train_bank, train_arrays, checkpoint, relative=True
    )
    train_k = reasoning_density_scores(train_bank, train_arrays, checkpoint)
    validation_k = reasoning_density_scores(
        validation_bank, validation_arrays, checkpoint
    )
    embedding_oof += train_k - full_embedding
    embedding_oof -= embedding_oof.mean(axis=1, keepdims=True)
    embedding_scale = float(embedding_oof.std())
    embedding_oof /= embedding_scale
    validation_k -= validation_k.mean(axis=1, keepdims=True)
    validation_k /= embedding_scale

    train_structures = answer_structures(train_bank)
    validation_structures = answer_structures(validation_bank)
    weight_grid = np.arange(-0.20, 0.201, 0.025)
    for name, (train_features, validation_features) in feature_sets.items():
        for c_value in (0.03, 0.1, 0.3):
            text_oof = np.empty(len(train_texts), dtype=np.float32)
            for fold in range(5):
                fit_rows = folds != fold
                held_rows = ~fit_rows
                fit = np.repeat(fit_rows, 4)
                held = np.repeat(held_rows, 4)
                classifier = LinearSVC(
                    C=c_value, dual="auto", random_state=7506
                ).fit(train_features[fit], train_labels.reshape(-1)[fit])
                text_oof[held] = classifier.decision_function(
                    train_features[held]
                )
            text_oof = text_oof.reshape(-1, 4)
            text_oof -= text_oof.mean(axis=1, keepdims=True)
            text_scale = float(text_oof.std())
            text_oof /= text_scale

            base_selected = np.asarray(
                select_parseable_groups(train_bank, embedding_oof, 0.3)
            )
            best_global = (
                train_labels[np.arange(len(train_labels)), base_selected].mean(),
                0.0,
            )
            for weight in weight_grid:
                selected = np.asarray(
                    select_parseable_groups(
                        train_bank, embedding_oof + weight * text_oof, 0.3
                    )
                )
                accuracy = train_labels[
                    np.arange(len(train_labels)), selected
                ].mean()
                if accuracy > best_global[0]:
                    best_global = (accuracy, float(weight))

            structure_weights = {}
            for structure in sorted(set(train_structures)):
                mask = np.asarray(
                    [value == structure for value in train_structures]
                )
                structure_bank = [
                    row for row, keep in zip(train_bank, mask) if keep
                ]
                best = (-1.0, 0.0)
                for weight in weight_grid:
                    selected = np.asarray(
                        select_parseable_groups(
                            structure_bank,
                            embedding_oof[mask] + weight * text_oof[mask],
                            0.3,
                        )
                    )
                    accuracy = train_labels[mask][
                        np.arange(mask.sum()), selected
                    ].mean()
                    if accuracy > best[0]:
                        best = (accuracy, float(weight))
                structure_weights[structure] = best[1]

            oof_structured = apply_structure_weights(
                embedding_oof,
                text_oof,
                train_structures,
                structure_weights,
            )
            oof_selected = np.asarray(
                select_parseable_groups(train_bank, oof_structured, 0.3)
            )
            oof_accuracy = train_labels[
                np.arange(len(train_labels)), oof_selected
            ].mean()

            classifier = LinearSVC(
                C=c_value, dual="auto", random_state=7506
            ).fit(train_features, train_labels.reshape(-1))
            validation_text_scores = classifier.decision_function(
                validation_features
            ).reshape(-1, 4)
            validation_text_scores -= validation_text_scores.mean(
                axis=1, keepdims=True
            )
            validation_text_scores /= text_scale
            validation_structured = apply_structure_weights(
                validation_k,
                validation_text_scores,
                validation_structures,
                structure_weights,
            )
            validation_selected = np.asarray(
                select_parseable_groups(
                    validation_bank, validation_structured, 0.3
                )
            )
            validation_accuracy = validation_labels[
                np.arange(len(validation_labels)), validation_selected
            ].mean()
            print(
                f"features={name} c={c_value:.2f} "
                f"global_oof={100 * best_global[0]:.4f} "
                f"global_weight={best_global[1]:.3f} "
                f"structured_oof={100 * oof_accuracy:.4f} "
                f"validation={100 * validation_accuracy:.4f} "
                f"weights={structure_weights}",
                flush=True,
            )


if __name__ == "__main__":
    main()
