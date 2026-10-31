"""Probe whether candidates use the question's quantities in calculations."""

from collections import Counter
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))

from common import DATA, load_inputs, parse_answer, read_jsonl
from calculation_coverage import (
    NUMBER,
    WORD_PATTERN,
    calculation_quantities,
    calculation_context,
    normalize_number,
    quantities,
    reasoning_body,
)
from linear_verifier import linear_scores
from parseable_group_verifier import select_parseable_groups
from reasoning_density_verifier import reasoning_density_scores
from text_fusion_verifier import text_fusion_scores


def coverage_features(bank):
    output = np.zeros((len(bank), 4, 5), dtype=np.float32)
    for row_index, row in enumerate(bank):
        question_values = quantities(row["question"], include_words=True)
        question_counts = Counter(question_values)
        unique_question = set(question_values)
        for candidate_index, candidate in enumerate(row["candidates"]):
            body = reasoning_body(candidate["text"])
            calculation = calculation_context(candidate["text"])
            body_counts = Counter(quantities(body, include_words=True))
            calculation_counts = Counter(
                calculation_quantities(calculation, include_words=True)
            )
            if question_values:
                body_occurrence = sum(
                    min(count, body_counts[value])
                    for value, count in question_counts.items()
                ) / len(question_values)
                calculation_occurrence = sum(
                    min(count, calculation_counts[value])
                    for value, count in question_counts.items()
                ) / len(question_values)
            else:
                body_occurrence = calculation_occurrence = 0.0
            if unique_question:
                body_unique = sum(
                    value in body_counts for value in unique_question
                ) / len(unique_question)
                calculation_unique = sum(
                    value in calculation_counts for value in unique_question
                ) / len(unique_question)
            else:
                body_unique = calculation_unique = 0.0
            output[row_index, candidate_index] = (
                body_occurrence,
                calculation_occurrence,
                body_unique,
                calculation_unique,
                np.log1p(len(quantities(calculation, include_words=False))),
            )
    return output


def standardize_rows(values):
    centered = values - values.mean(axis=1, keepdims=True)
    return centered / (values.std(axis=1, keepdims=True) + 1e-6)


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


def apply_structure_text(scores, text, structures, weights):
    output = scores.copy()
    for index, structure in enumerate(structures):
        output[index] += weights.get(structure, -0.05) * text[index]
    return output


def oof_l_scores(bank, arrays, labels, folds):
    embeddings = arrays["embeddings"].astype(np.float32)
    embeddings -= embeddings.mean(axis=1, keepdims=True)
    embedding_oof = np.empty(labels.shape, dtype=np.float32)
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
        embedding_oof[held] = (
            (embeddings[held] - mean) / scale
        ) @ weights

    checkpoint = CODE / "runs/model_k_reasoning_density"
    full_embedding = linear_scores(bank, arrays, checkpoint, relative=True)
    model_k = reasoning_density_scores(bank, arrays, checkpoint)
    embedding_oof += model_k - full_embedding
    embedding_oof -= embedding_oof.mean(axis=1, keepdims=True)
    embedding_scale = float(embedding_oof.std())
    embedding_oof /= embedding_scale

    texts = [candidate["text"] for row in bank for candidate in row["candidates"]]
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=5,
        max_features=25000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    text_x = vectorizer.fit_transform(texts)
    text_oof = np.empty(len(texts), dtype=np.float32)
    for fold in range(folds.max() + 1):
        fit_rows = folds != fold
        held_rows = ~fit_rows
        fit = np.repeat(fit_rows, 4)
        held = np.repeat(held_rows, 4)
        classifier = LinearSVC(C=0.1, dual="auto", random_state=7506).fit(
            text_x[fit], labels.reshape(-1)[fit]
        )
        text_oof[held] = classifier.decision_function(text_x[held])
    text_oof = text_oof.reshape(-1, 4)
    text_oof -= text_oof.mean(axis=1, keepdims=True)
    text_oof /= float(text_oof.std())
    structure_weights = {
        "4": 0.0,
        "3-1": -0.025,
        "2-2": 0.025,
        "2-1-1": -0.05,
        "1-1-1-1": -0.10,
    }
    return apply_structure_text(
        embedding_oof,
        text_oof,
        answer_structures(bank),
        structure_weights,
    ), embedding_scale


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank)
    validation_labels = candidate_labels("validation", validation_bank)
    folds = np.arange(len(train_bank)) % 5
    train_scores, embedding_scale = oof_l_scores(
        train_bank, train_arrays, train_labels, folds
    )
    validation_scores = text_fusion_scores(
        validation_bank,
        validation_arrays,
        CODE / "runs/model_l_text_fusion",
    )
    # text_fusion_scores already divides Model K by the same stored OOF scale.
    del embedding_scale

    train_features = coverage_features(train_bank)
    validation_features = coverage_features(validation_bank)
    names = (
        "body_occurrence",
        "calculation_occurrence",
        "body_unique",
        "calculation_unique",
        "calculation_number_count",
    )
    for feature_index, name in enumerate(names):
        train_feature = standardize_rows(train_features[:, :, feature_index])
        validation_feature = standardize_rows(
            validation_features[:, :, feature_index]
        )
        best = (-1.0, 0.0)
        for weight in np.arange(-0.20, 0.201, 0.01):
            selected = np.asarray(
                select_parseable_groups(
                    train_bank,
                    train_scores + weight * train_feature,
                    0.3,
                )
            )
            accuracy = train_labels[
                np.arange(len(train_labels)), selected
            ].mean()
            if accuracy > best[0]:
                best = (accuracy, float(weight))
        validation_selected = np.asarray(
            select_parseable_groups(
                validation_bank,
                validation_scores + best[1] * validation_feature,
                0.3,
            )
        )
        validation_accuracy = validation_labels[
            np.arange(len(validation_labels)), validation_selected
        ].mean()
        print(
            f"feature={name} oof={100 * best[0]:.4f} "
            f"weight={best[1]:.3f} validation={100 * validation_accuracy:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
