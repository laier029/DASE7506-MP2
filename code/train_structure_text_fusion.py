"""Train Model M's answer-structure-specific character TF-IDF heads."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / "models"))

from common import DATA, load_inputs, parse_answer, read_jsonl
from answer_step_consistency import _reasoning_prefix
from linear_verifier import linear_scores
from reasoning_density_verifier import reasoning_density_scores


STRUCTURE_WEIGHTS = {
    "4": 0.0,
    "3-1": 0.025,
    "2-2": -0.225,
    "2-1-1": -0.10,
    "1-1-1-1": -0.10,
}


def candidate_labels(bank, references):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=25000)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--reasoning-only", action="store_true")
    args = parser.parse_args()

    tick = time.perf_counter()
    bank, arrays = load_inputs("train")
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    labels = candidate_labels(bank, references)
    structures = answer_structures(bank)
    texts = [candidate["text"] for row in bank for candidate in row["candidates"]]
    if args.reasoning_only:
        texts = [_reasoning_prefix(text) for text in texts]
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=5,
        max_features=args.max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    text_features = vectorizer.fit_transform(texts)
    fold_ids = np.arange(len(bank)) % args.folds

    embeddings = arrays["embeddings"].astype(np.float32)
    embeddings -= embeddings.mean(axis=1, keepdims=True)
    embedding_oof = np.empty((len(bank), 4), dtype=np.float32)
    for fold in range(args.folds):
        fit_rows = fold_ids != fold
        held_rows = ~fit_rows
        fit_x = embeddings[fit_rows].reshape(-1, 1024)
        mean = fit_x.mean(axis=0)
        scale = fit_x.std(axis=0)
        scale[scale < 1e-6] = 1.0
        fit_x = (fit_x - mean) / scale
        fit_y = labels[fit_rows].reshape(-1).astype(np.float32)
        gram = fit_x.T @ fit_x / len(fit_x)
        target = fit_x.T @ fit_y / len(fit_x)
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        weights = eigenvectors @ (
            (eigenvectors.T @ target) / (eigenvalues + 0.3)
        )
        embedding_oof[held_rows] = (
            (embeddings[held_rows] - mean) / scale
        ) @ weights

    full_embedding = linear_scores(
        bank, arrays, args.base_checkpoint, relative=True
    )
    model_k_scores = reasoning_density_scores(bank, arrays, args.base_checkpoint)
    embedding_oof += model_k_scores - full_embedding
    embedding_oof -= embedding_oof.mean(axis=1, keepdims=True)
    embedding_scale = float(embedding_oof.std())

    active_structures = sorted(set(structures) - {"4"})
    text_oof = np.zeros((len(bank), 4), dtype=np.float32)
    classifiers = {}
    for structure in active_structures:
        for fold in range(args.folds):
            fit_rows = (fold_ids != fold) & (structures == structure)
            held_rows = (fold_ids == fold) & (structures == structure)
            fit = np.repeat(fit_rows, 4)
            held = np.repeat(held_rows, 4)
            classifier = LinearSVC(
                C=args.c, dual="auto", random_state=7506
            ).fit(text_features[fit], labels.reshape(-1)[fit])
            text_oof[held_rows] = classifier.decision_function(
                text_features[held]
            ).reshape(-1, 4)

        fit_rows = structures == structure
        fit = np.repeat(fit_rows, 4)
        classifiers[structure] = LinearSVC(
            C=args.c, dual="auto", random_state=7506
        ).fit(text_features[fit], labels.reshape(-1)[fit])

    text_oof -= text_oof.mean(axis=1, keepdims=True)
    text_scales = {}
    for structure in active_structures:
        scale = float(text_oof[structures == structure].std())
        text_scales[structure] = scale if scale > 1e-6 else 1.0

    artifact = {
        "vectorizer": vectorizer,
        "classifiers": classifiers,
        "embedding_score_scale": embedding_scale,
        "text_score_scales": text_scales,
        "structure_weights": STRUCTURE_WEIGHTS,
        "reasoning_only": args.reasoning_only,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.base_checkpoint / "model.npz", args.output / "model.npz")
    artifact_path = args.output / "structure_text_model.joblib"
    joblib.dump(artifact, artifact_path, compress=0)
    asset_bytes = (args.output / "model.npz").stat().st_size + artifact_path.stat().st_size
    metadata = {
        "method": "model_k_plus_structure_specific_character_tfidf",
        "training_split": "train",
        "folds_for_scale": args.folds,
        "character_ngram_range": [3, 5],
        "max_features": args.max_features,
        "min_document_frequency": 5,
        "linear_svc_c": args.c,
        "structure_weights": STRUCTURE_WEIGHTS,
        "reasoning_only": args.reasoning_only,
        "embedding_score_scale": embedding_scale,
        "text_score_scales": text_scales,
        "asset_bytes_uncompressed": asset_bytes,
        "training_seconds": time.perf_counter() - tick,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
