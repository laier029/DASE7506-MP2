"""Probe a compact learned ranker over distinct final-answer groups."""

from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import re
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))

from answer_step_consistency import answer_step_consistency
from arithmetic_consistency import arithmetic_counts, broad_arithmetic_counts
from common import DATA, load_inputs, parse_answer, read_jsonl
from linear_verifier import linear_scores
from reasoning_density_verifier import reasoning_density_scores


NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
OPERATORS = (" + ", " - ", "\\times", "×", "\\div", "÷")


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


def oof_model_k(bank, arrays, labels, folds):
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
        normalized = (fit_x - mean) / scale
        fit_y = labels[fit].reshape(-1).astype(np.float32)
        gram = normalized.T @ normalized / len(normalized)
        target = normalized.T @ fit_y / len(normalized)
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        weights = eigenvectors @ (
            (eigenvectors.T @ target) / (eigenvalues + 0.3)
        )
        scores[held] = ((embeddings[held] - mean) / scale) @ weights

    checkpoint = CODE / "runs/model_k_reasoning_density"
    full_embedding = linear_scores(bank, arrays, checkpoint, relative=True)
    full_k = reasoning_density_scores(bank, arrays, checkpoint)
    scores += full_k - full_embedding
    return scores


def text_scores(train_bank, validation_bank, labels, folds):
    train_text = [
        candidate["text"] for row in train_bank for candidate in row["candidates"]
    ]
    validation_text = [
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
    train_x = vectorizer.fit_transform(train_text)
    validation_x = vectorizer.transform(validation_text)
    oof = np.empty(len(train_text), dtype=np.float32)
    for fold in range(folds.max() + 1):
        fit_rows = folds != fold
        held_rows = ~fit_rows
        fit = np.repeat(fit_rows, 4)
        held = np.repeat(held_rows, 4)
        classifier = LinearSVC(C=0.1, dual="auto", random_state=7506).fit(
            train_x[fit], labels.reshape(-1)[fit]
        )
        oof[held] = classifier.decision_function(train_x[held])
    classifier = LinearSVC(C=0.1, dual="auto", random_state=7506).fit(
        train_x, labels.reshape(-1)
    )
    validation = classifier.decision_function(validation_x)
    return oof.reshape(-1, 4), validation.reshape(-1, 4)


def candidate_static_features(bank):
    features = np.zeros((len(bank), 4, 10), dtype=np.float32)
    for row_index, row in enumerate(bank):
        for candidate_index, candidate in enumerate(row["candidates"]):
            text = candidate["text"]
            narrow_checked, narrow_bad = arithmetic_counts(text)
            broad_checked, broad_bad = broad_arithmetic_counts(text)
            step = answer_step_consistency(text)
            features[row_index, candidate_index] = (
                np.log1p(len(text)),
                np.log1p(len(NUMBER.findall(text))),
                np.log1p(sum(text.count(operator) for operator in OPERATORS)),
                np.log1p(narrow_checked),
                float(narrow_bad > 0),
                np.log1p(broad_checked),
                float(broad_bad > 0),
                float(step is True),
                float(step is False),
                float(step is None),
            )
    return features


def group_features(bank, base_scores, text, static, labels=None):
    rows = []
    question_indices = []
    candidate_indices = []
    targets = []
    for question_index, (row, row_scores, row_text, row_static) in enumerate(
        zip(bank, base_scores, text, static)
    ):
        groups = defaultdict(list)
        for index, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate["text"])
            if answer is not None:
                groups[answer].append(index)
        structure = sorted((len(indices) for indices in groups.values()), reverse=True)
        row_mean = float(row_scores.mean())
        row_std = float(row_scores.std()) + 1e-6
        for answer, indices in groups.items():
            values = row_scores[indices]
            text_values = row_text[indices]
            static_values = row_static[indices]
            best = indices[int(values.argmax())]
            try:
                answer_value = float(Fraction(answer))
                answer_log = np.sign(answer_value) * np.log1p(abs(answer_value))
                answer_integer = float(Fraction(answer).denominator == 1)
            except (ValueError, ZeroDivisionError, OverflowError):
                answer_log = 0.0
                answer_integer = 0.0
            top = np.sort(values)[::-1]
            top_gap = float(top[0] - top[1]) if len(top) > 1 else 0.0
            group_row = [
                len(indices) / 4,
                float(values.max()),
                float(values.mean()),
                float(values.min()),
                float(values.std()),
                float((values.max() - row_mean) / row_std),
                float((values.mean() - row_mean) / row_std),
                top_gap,
                float(text_values.max()),
                float(text_values.mean()),
                float(text_values.min()),
                float(text_values.std()),
                answer_log,
                answer_integer,
                float(best == 0),
                float(best == 1),
                float(best == 2),
                float(best == 3),
            ]
            group_row.extend(static_values.max(axis=0).tolist())
            group_row.extend(static_values.mean(axis=0).tolist())
            group_row.extend(
                [
                    float(structure == [4]),
                    float(structure == [3, 1]),
                    float(structure == [2, 2]),
                    float(structure == [2, 1, 1]),
                    float(structure == [1, 1, 1, 1]),
                ]
            )
            rows.append(group_row)
            question_indices.append(question_index)
            candidate_indices.append(best)
            if labels is not None:
                targets.append(float(labels[question_index, indices].any()))
    output = (
        np.asarray(rows, dtype=np.float32),
        np.asarray(question_indices),
        np.asarray(candidate_indices),
    )
    if labels is not None:
        return (*output, np.asarray(targets, dtype=np.float32))
    return output


def select_groups(scores, questions, candidates, n_questions):
    selected = np.empty(n_questions, dtype=np.int64)
    for question in range(n_questions):
        mask = questions == question
        group_rows = np.flatnonzero(mask)
        winner = group_rows[int(scores[mask].argmax())]
        selected[question] = candidates[winner]
    return selected


def fit_ridge(train_x, train_y, score_x, alpha):
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    model = Ridge(alpha=alpha, random_state=7506).fit(
        (train_x - mean) / scale, train_y
    )
    return model.predict((score_x - mean) / scale)


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank)
    validation_labels = candidate_labels("validation", validation_bank)
    folds = np.arange(len(train_bank)) % 5
    train_k = oof_model_k(train_bank, train_arrays, train_labels, folds)
    validation_k = reasoning_density_scores(
        validation_bank,
        validation_arrays,
        CODE / "runs/model_k_reasoning_density",
    )
    train_text, validation_text = text_scores(
        train_bank, validation_bank, train_labels, folds
    )
    train_static = candidate_static_features(train_bank)
    validation_static = candidate_static_features(validation_bank)
    train_x, train_questions, train_candidates, train_y = group_features(
        train_bank, train_k, train_text, train_static, train_labels
    )
    validation_x, validation_questions, validation_candidates = group_features(
        validation_bank, validation_k, validation_text, validation_static
    )

    for alpha in (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
        oof_scores = np.empty(len(train_x), dtype=np.float32)
        for fold in range(5):
            fit = folds[train_questions] != fold
            held = ~fit
            oof_scores[held] = fit_ridge(
                train_x[fit], train_y[fit], train_x[held], alpha
            )
        train_selected = select_groups(
            oof_scores, train_questions, train_candidates, len(train_bank)
        )
        oof_accuracy = train_labels[
            np.arange(len(train_labels)), train_selected
        ].mean()
        validation_scores = fit_ridge(
            train_x, train_y, validation_x, alpha
        )
        validation_selected = select_groups(
            validation_scores,
            validation_questions,
            validation_candidates,
            len(validation_bank),
        )
        validation_accuracy = validation_labels[
            np.arange(len(validation_labels)), validation_selected
        ].mean()
        print(
            f"alpha={alpha:g} groups={len(train_x)} "
            f"oof={100 * oof_accuracy:.4f} "
            f"validation={100 * validation_accuracy:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
