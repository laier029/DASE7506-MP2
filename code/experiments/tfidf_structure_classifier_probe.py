"""Probe separate text-classification heads for each answer structure."""

from collections import Counter
import argparse
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
from answer_step_consistency import _reasoning_prefix
from group_ranker_probe import candidate_labels, oof_model_k
from parseable_group_verifier import select_parseable_groups
from reasoning_density_verifier import reasoning_density_scores


def structures(bank):
    return np.asarray(
        [
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
    )


def centered_scaled(scores, row_structures, scales=None):
    output = scores - scores.mean(axis=1, keepdims=True)
    if scales is None:
        scales = {}
        for structure in set(row_structures):
            scale = float(output[row_structures == structure].std())
            scales[structure] = scale if scale > 1e-6 else 1.0
    for structure, scale in scales.items():
        output[row_structures == structure] /= scale
    return output, scales


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-seed", type=int)
    parser.add_argument("--c", type=float, action="append")
    parser.add_argument("--reasoning-only", action="store_true")
    parser.add_argument(
        "--analyzer", choices=("char", "word", "both"), default="char"
    )
    args = parser.parse_args()
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    if args.fold_seed is None:
        folds = np.arange(len(train_bank)) % 5
    else:
        rng = np.random.default_rng(args.fold_seed)
        order = rng.permutation(len(train_bank))
        folds = np.empty(len(train_bank), dtype=np.int64)
        folds[order] = np.arange(len(train_bank)) % 5
    train_structures = structures(train_bank)
    validation_structures = structures(validation_bank)

    train_k = oof_model_k(train_bank, train_arrays, train_labels, folds)
    validation_k = reasoning_density_scores(
        validation_bank,
        validation_arrays,
        CODE / "runs/model_k_reasoning_density",
    )
    train_k -= train_k.mean(axis=1, keepdims=True)
    k_scale = float(train_k.std())
    train_k /= k_scale
    validation_k -= validation_k.mean(axis=1, keepdims=True)
    validation_k /= k_scale

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
    if args.reasoning_only:
        train_texts = [_reasoning_prefix(text) for text in train_texts]
        validation_texts = [_reasoning_prefix(text) for text in validation_texts]
    if args.analyzer == "both":
        char_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=5,
            max_features=25000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=3,
            max_features=25000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        train_x = sparse.hstack(
            [
                char_vectorizer.fit_transform(train_texts),
                word_vectorizer.fit_transform(train_texts),
            ],
            format="csr",
        )
        validation_x = sparse.hstack(
            [
                char_vectorizer.transform(validation_texts),
                word_vectorizer.transform(validation_texts),
            ],
            format="csr",
        )
    else:
        vectorizer = TfidfVectorizer(
            analyzer=args.analyzer,
            ngram_range=(3, 5) if args.analyzer == "char" else (1, 2),
            min_df=5 if args.analyzer == "char" else 3,
            max_features=25000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        train_x = vectorizer.fit_transform(train_texts)
        validation_x = vectorizer.transform(validation_texts)
    active_structures = sorted(set(train_structures) - {"4"})

    for c_value in (args.c or (0.01, 0.03, 0.1, 0.3)):
        text_oof = np.zeros((len(train_bank), 4), dtype=np.float32)
        validation_text = np.zeros((len(validation_bank), 4), dtype=np.float32)
        for structure in active_structures:
            for fold in range(5):
                fit_rows = (folds != fold) & (train_structures == structure)
                held_rows = (folds == fold) & (train_structures == structure)
                fit = np.repeat(fit_rows, 4)
                held = np.repeat(held_rows, 4)
                classifier = LinearSVC(
                    C=c_value, dual="auto", random_state=7506
                ).fit(train_x[fit], train_labels.reshape(-1)[fit])
                text_oof[held_rows] = classifier.decision_function(
                    train_x[held]
                ).reshape(-1, 4)

            fit_rows = train_structures == structure
            fit = np.repeat(fit_rows, 4)
            classifier = LinearSVC(
                C=c_value, dual="auto", random_state=7506
            ).fit(train_x[fit], train_labels.reshape(-1)[fit])
            validation_rows = validation_structures == structure
            validation = np.repeat(validation_rows, 4)
            validation_text[validation_rows] = classifier.decision_function(
                validation_x[validation]
            ).reshape(-1, 4)

        text_oof, text_scales = centered_scaled(text_oof, train_structures)
        validation_text, _ = centered_scaled(
            validation_text, validation_structures, text_scales
        )
        weights = {"4": 0.0}
        for structure in active_structures:
            mask = train_structures == structure
            structure_bank = [
                row for row, keep in zip(train_bank, mask) if keep
            ]
            best = (-1.0, 0.0)
            for weight in np.arange(-0.30, 0.301, 0.025):
                selected = np.asarray(
                    select_parseable_groups(
                        structure_bank,
                        train_k[mask] + weight * text_oof[mask],
                        0.3,
                    )
                )
                accuracy = train_labels[mask][
                    np.arange(mask.sum()), selected
                ].mean()
                if accuracy > best[0]:
                    best = (accuracy, float(weight))
            weights[structure] = best[1]

        train_scores = train_k.copy()
        validation_scores = validation_k.copy()
        for structure, weight in weights.items():
            train_scores[train_structures == structure] += (
                weight * text_oof[train_structures == structure]
            )
            validation_scores[validation_structures == structure] += (
                weight * validation_text[validation_structures == structure]
            )
        train_selected = np.asarray(
            select_parseable_groups(train_bank, train_scores, 0.3)
        )
        validation_selected = np.asarray(
            select_parseable_groups(validation_bank, validation_scores, 0.3)
        )
        train_accuracy = train_labels[
            np.arange(len(train_labels)), train_selected
        ].mean()
        validation_accuracy = validation_labels[
            np.arange(len(validation_labels)), validation_selected
        ].mean()
        fixed_weights = {
            "4": 0.0,
            "1-1-1-1": -0.10,
            "2-1-1": -0.10,
            "2-2": -0.225,
            "3-1": 0.025,
        }
        fixed_train_scores = train_k.copy()
        fixed_validation_scores = validation_k.copy()
        for structure, weight in fixed_weights.items():
            fixed_train_scores[train_structures == structure] += (
                weight * text_oof[train_structures == structure]
            )
            fixed_validation_scores[validation_structures == structure] += (
                weight * validation_text[validation_structures == structure]
            )
        base_selected = np.asarray(
            select_parseable_groups(train_bank, train_k, 0.3)
        )
        fixed_train_selected = np.asarray(
            select_parseable_groups(train_bank, fixed_train_scores, 0.3)
        )
        fixed_validation_selected = np.asarray(
            select_parseable_groups(
                validation_bank, fixed_validation_scores, 0.3
            )
        )
        base_accuracy = train_labels[
            np.arange(len(train_labels)), base_selected
        ].mean()
        fixed_train_accuracy = train_labels[
            np.arange(len(train_labels)), fixed_train_selected
        ].mean()
        fixed_validation_accuracy = validation_labels[
            np.arange(len(validation_labels)), fixed_validation_selected
        ].mean()
        print(
            f"c={c_value:g} oof={100 * train_accuracy:.4f} "
            f"validation={100 * validation_accuracy:.4f} weights={weights} "
            f"base_oof={100 * base_accuracy:.4f} "
            f"fixed_oof={100 * fixed_train_accuracy:.4f} "
            f"fixed_validation={100 * fixed_validation_accuracy:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
