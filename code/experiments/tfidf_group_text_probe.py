"""Probe direct text classification and fusion at final-answer-group level."""

from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from common import load_inputs, parse_answer
from group_ranker_probe import candidate_labels, oof_model_k
from reasoning_density_verifier import reasoning_density_scores


def structure(row):
    return "-".join(
        str(value)
        for value in sorted(
            Counter(
                parse_answer(candidate["text"])
                for candidate in row["candidates"]
            ).values(),
            reverse=True,
        )
    )


def make_groups(bank, candidate_features, candidate_scores, labels):
    feature_rows = []
    questions = []
    structures = []
    candidates = []
    targets = []
    base_scores = []
    for question, (row, row_scores) in enumerate(zip(bank, candidate_scores)):
        groups = defaultdict(list)
        for index, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate["text"])
            if answer is not None:
                groups[answer].append(index)
        row_structure = structure(row)
        for indices in groups.values():
            flat = [4 * question + index for index in indices]
            feature_rows.append(
                sparse.csr_matrix(candidate_features[flat].sum(axis=0))
                * (1.0 / len(flat))
            )
            values = row_scores[indices]
            questions.append(question)
            structures.append(row_structure)
            candidates.append(indices[int(values.argmax())])
            targets.append(float(labels[question, indices].any()))
            base_scores.append(float(values.max() + 0.3 * values.mean()))
    return (
        sparse.vstack(feature_rows, format="csr"),
        np.asarray(questions),
        np.asarray(structures),
        np.asarray(candidates),
        np.asarray(targets, dtype=np.float32),
        np.asarray(base_scores, dtype=np.float32),
    )


def center_by_question(values, questions):
    output = values.copy()
    for question in np.unique(questions):
        mask = questions == question
        output[mask] -= output[mask].mean()
    return output


def select_accuracy(scores, questions, targets, n_questions):
    correct = 0
    for question in range(n_questions):
        rows = np.flatnonzero(questions == question)
        winner = rows[int(scores[rows].argmax())]
        correct += int(targets[winner])
    return correct / n_questions


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    folds = np.arange(len(train_bank)) % 5
    train_k = oof_model_k(train_bank, train_arrays, train_labels, folds)
    validation_k = reasoning_density_scores(
        validation_bank,
        validation_arrays,
        CODE / "runs/model_k_reasoning_density",
    )

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
    train_candidate_x = vectorizer.fit_transform(train_texts)
    validation_candidate_x = vectorizer.transform(validation_texts)
    (
        train_x,
        train_questions,
        train_structures,
        train_candidates,
        train_y,
        train_base,
    ) = make_groups(train_bank, train_candidate_x, train_k, train_labels)
    (
        validation_x,
        validation_questions,
        validation_structures,
        validation_candidates,
        validation_y,
        validation_base,
    ) = make_groups(
        validation_bank,
        validation_candidate_x,
        validation_k,
        validation_labels,
    )
    del train_candidates, validation_candidates

    train_base = center_by_question(train_base, train_questions)
    base_scale = float(train_base.std())
    train_base /= base_scale
    validation_base = center_by_question(validation_base, validation_questions)
    validation_base /= base_scale
    active_structures = sorted(set(train_structures) - {"4"})

    for c_value in (0.001, 0.01, 0.1, 0.3):
        text_oof = np.zeros(train_x.shape[0], dtype=np.float32)
        validation_text = np.zeros(validation_x.shape[0], dtype=np.float32)
        for row_structure in active_structures:
            for fold in range(5):
                fit = (
                    (folds[train_questions] != fold)
                    & (train_structures == row_structure)
                )
                held = (
                    (folds[train_questions] == fold)
                    & (train_structures == row_structure)
                )
                classifier = LinearSVC(
                    C=c_value, dual="auto", random_state=7506
                ).fit(train_x[fit], train_y[fit])
                text_oof[held] = classifier.decision_function(train_x[held])
            fit = train_structures == row_structure
            classifier = LinearSVC(
                C=c_value, dual="auto", random_state=7506
            ).fit(train_x[fit], train_y[fit])
            held = validation_structures == row_structure
            validation_text[held] = classifier.decision_function(
                validation_x[held]
            )

        text_oof = center_by_question(text_oof, train_questions)
        validation_text = center_by_question(
            validation_text, validation_questions
        )
        scales = {}
        for row_structure in active_structures:
            scale = float(text_oof[train_structures == row_structure].std())
            scales[row_structure] = scale if scale > 1e-6 else 1.0
            text_oof[train_structures == row_structure] /= scales[row_structure]
            validation_text[validation_structures == row_structure] /= scales[
                row_structure
            ]

        weights = {"4": 0.0}
        for row_structure in active_structures:
            question_mask = np.asarray(
                [structure(row) == row_structure for row in train_bank]
            )
            group_mask = train_structures == row_structure
            # Reindex this structure's question IDs for the local selector.
            local_questions = train_questions[group_mask]
            unique_questions = np.flatnonzero(question_mask)
            remap = {question: i for i, question in enumerate(unique_questions)}
            local_questions = np.asarray(
                [remap[question] for question in local_questions]
            )
            best = (-1.0, 0.0)
            for weight in np.arange(-0.30, 0.301, 0.025):
                score = select_accuracy(
                    train_base[group_mask] + weight * text_oof[group_mask],
                    local_questions,
                    train_y[group_mask],
                    len(unique_questions),
                )
                if score > best[0]:
                    best = (score, float(weight))
            weights[row_structure] = best[1]

        train_scores = train_base.copy()
        validation_scores = validation_base.copy()
        for row_structure, weight in weights.items():
            train_scores[train_structures == row_structure] += (
                weight * text_oof[train_structures == row_structure]
            )
            validation_scores[validation_structures == row_structure] += (
                weight * validation_text[validation_structures == row_structure]
            )
        train_accuracy = select_accuracy(
            train_scores, train_questions, train_y, len(train_bank)
        )
        validation_accuracy = select_accuracy(
            validation_scores,
            validation_questions,
            validation_y,
            len(validation_bank),
        )
        print(
            f"c={c_value:g} oof={100 * train_accuracy:.4f} "
            f"validation={100 * validation_accuracy:.4f} weights={weights}",
            flush=True,
        )


if __name__ == "__main__":
    main()
