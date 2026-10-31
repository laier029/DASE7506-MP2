"""Train the compact character TF-IDF component used by Model L."""

import argparse
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
from linear_verifier import linear_scores
from reasoning_density_verifier import reasoning_density_scores


def candidate_labels(bank, references):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--text-weight", type=float, default=-0.05)
    parser.add_argument("--max-features", type=int, default=25000)
    parser.add_argument("--structure-aware", action="store_true")
    args = parser.parse_args()

    tick = time.perf_counter()
    bank, arrays = load_inputs("train")
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    labels = candidate_labels(bank, references)
    texts = [candidate["text"] for row in bank for candidate in row["candidates"]]
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
    text_oof = np.empty(len(texts), dtype=np.float32)
    embedding_oof = np.empty((len(bank), 4), dtype=np.float32)
    embeddings = arrays["embeddings"].astype(np.float32)
    embeddings -= embeddings.mean(axis=1, keepdims=True)

    for fold in range(args.folds):
        row_train = fold_ids != fold
        row_val = ~row_train
        candidate_train = np.repeat(row_train, 4)
        candidate_val = np.repeat(row_val, 4)
        fold_classifier = LinearSVC(C=0.1, dual="auto", random_state=7506).fit(
            text_features[candidate_train], labels.reshape(-1)[candidate_train]
        )
        text_oof[candidate_val] = fold_classifier.decision_function(
            text_features[candidate_val]
        )

        fit_x = embeddings[row_train].reshape(-1, 1024)
        mean = fit_x.mean(axis=0)
        scale = fit_x.std(axis=0)
        scale[scale < 1e-6] = 1.0
        fit_x = (fit_x - mean) / scale
        fit_y = labels[row_train].reshape(-1).astype(np.float32)
        gram = fit_x.T @ fit_x / len(fit_x)
        target = fit_x.T @ fit_y / len(fit_x)
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        weights = eigenvectors @ (
            (eigenvectors.T @ target) / (eigenvalues + 0.3)
        )
        embedding_oof[row_val] = (
            (embeddings[row_val] - mean) / scale
        ) @ weights

    full_embedding = linear_scores(
        bank, arrays, args.base_checkpoint, relative=True
    )
    model_k_scores = reasoning_density_scores(bank, arrays, args.base_checkpoint)
    embedding_oof += model_k_scores - full_embedding
    text_oof = text_oof.reshape(-1, 4)
    embedding_oof -= embedding_oof.mean(axis=1, keepdims=True)
    text_oof -= text_oof.mean(axis=1, keepdims=True)
    embedding_scale = float(embedding_oof.std())
    text_scale = float(text_oof.std())

    classifier = LinearSVC(C=0.1, dual="auto", random_state=7506).fit(
        text_features, labels.reshape(-1)
    )
    structure_weights = (
        {
            "4": 0.0,
            "3-1": -0.025,
            "2-2": 0.025,
            "2-1-1": -0.05,
            "1-1-1-1": -0.10,
        }
        if args.structure_aware
        else None
    )
    artifact = {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "embedding_score_scale": embedding_scale,
        "text_score_scale": text_scale,
        "text_weight": args.text_weight,
        "structure_weights": structure_weights,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.base_checkpoint / "model.npz", args.output / "model.npz")
    artifact_path = args.output / "text_model.joblib"
    joblib.dump(artifact, artifact_path, compress=0)
    asset_bytes = (args.output / "model.npz").stat().st_size + artifact_path.stat().st_size
    metadata = {
        "method": "model_k_plus_negative_character_tfidf",
        "training_split": "train",
        "folds_for_scale": args.folds,
        "character_ngram_range": [3, 5],
        "max_features": args.max_features,
        "min_document_frequency": 5,
        "linear_svc_c": 0.1,
        "text_weight": args.text_weight,
        "structure_weights": structure_weights,
        "embedding_score_scale": embedding_scale,
        "text_score_scale": text_scale,
        "asset_bytes_uncompressed": asset_bytes,
        "training_seconds": time.perf_counter() - tick,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
