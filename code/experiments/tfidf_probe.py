"""Probe discriminative word- and character-ngram correctness scores."""

from pathlib import Path
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))

from common import DATA, load_inputs, parse_answer, read_jsonl
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
    train_text = [
        candidate["text"] for row in train_bank for candidate in row["candidates"]
    ]
    val_text = [
        candidate["text"] for row in val_bank for candidate in row["candidates"]
    ]
    model_k = reasoning_density_scores(
        val_bank, val_arrays, CODE / "runs/model_k_reasoning_density"
    )

    settings = {
        "word": dict(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=3,
            max_features=50000,
            sublinear_tf=True,
        ),
        "char": dict(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=5,
            max_features=50000,
            sublinear_tf=True,
        ),
    }
    for name, configuration in settings.items():
        vectorizer = TfidfVectorizer(dtype=np.float32, **configuration)
        train_features = vectorizer.fit_transform(train_text)
        val_features = vectorizer.transform(val_text)
        print(
            f"{name} features={train_features.shape[1]} "
            f"train_nnz={train_features.nnz}",
            flush=True,
        )
        for regularization in (0.01, 0.03, 0.1, 0.3, 1.0):
            classifier = LinearSVC(
                C=regularization, dual="auto", random_state=7506
            ).fit(train_features, train_y.reshape(-1))
            scores = classifier.decision_function(val_features).reshape(-1, 4)
            selected = np.asarray(select_parseable_groups(val_bank, scores, 0.3))
            accuracy = val_y[np.arange(len(val_y)), selected].mean()

            k_centered = model_k - model_k.mean(axis=1, keepdims=True)
            text_centered = scores - scores.mean(axis=1, keepdims=True)
            k_centered /= k_centered.std()
            text_centered /= text_centered.std()
            best = (0.0, 0.0)
            for weight in np.arange(-0.2, 0.501, 0.025):
                fused = np.asarray(
                    select_parseable_groups(
                        val_bank, k_centered + weight * text_centered, 0.3
                    )
                )
                fused_accuracy = val_y[np.arange(len(val_y)), fused].mean()
                if fused_accuracy > best[0]:
                    best = (fused_accuracy, weight)
            print(
                f"{name} C={regularization:g} alone={100 * accuracy:.4f} "
                f"fused={100 * best[0]:.4f} weight={best[1]:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
