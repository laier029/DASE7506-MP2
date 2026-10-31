"""Train a deterministic ridge scorer on supplied training embeddings."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from common import DATA, load_inputs, parse_answer, read_jsonl


def candidate_labels(bank, references):
    """Return one correctness label per candidate in bank order."""
    labels = []
    for row, reference in zip(bank, references):
        target = parse_answer(reference["answer"])
        labels.extend(
            parse_answer(candidate["text"]) == target
            for candidate in row["candidates"]
        )
    return np.asarray(labels, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", choices=("raw", "relative"), required=True)
    parser.add_argument("--regularization", type=float, required=True)
    parser.add_argument("--consensus-weight", type=float, default=0.0)
    parser.add_argument("--arithmetic-penalty", type=float, default=0.0)
    parser.add_argument("--answer-step-penalty", type=float, default=0.0)
    parser.add_argument("--group-mean-weight", type=float, default=0.0)
    parser.add_argument("--broad-arithmetic-penalty", type=float, default=0.0)
    parser.add_argument("--number-count-weight", type=float, default=0.0)
    parser.add_argument("--operation-count-weight", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.regularization <= 0:
        raise ValueError("Regularization must be positive.")

    tick = time.perf_counter()
    bank, arrays = load_inputs("train")
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    labels = candidate_labels(bank, references)

    embeddings = arrays["embeddings"].astype(np.float32, copy=True)
    relative = args.representation == "relative"
    if relative:
        embeddings -= embeddings.mean(axis=1, keepdims=True)

    features = embeddings.reshape(-1, embeddings.shape[-1])
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    features = (features - mean) / scale

    sample_count = len(features)
    gram = (features.T @ features) / sample_count
    target = (features.T @ labels) / sample_count
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    projected_target = eigenvectors.T @ target
    weights = eigenvectors @ (
        projected_target / (eigenvalues + args.regularization)
    )

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "model.npz"
    np.savez(
        model_path,
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        weights=weights.astype(np.float32),
        relative=np.asarray(relative, dtype=np.uint8),
        regularization=np.asarray(args.regularization, dtype=np.float32),
        consensus_weight=np.asarray(args.consensus_weight, dtype=np.float32),
        arithmetic_penalty=np.asarray(args.arithmetic_penalty, dtype=np.float32),
        answer_step_penalty=np.asarray(args.answer_step_penalty, dtype=np.float32),
        group_mean_weight=np.asarray(args.group_mean_weight, dtype=np.float32),
        broad_arithmetic_penalty=np.asarray(
            args.broad_arithmetic_penalty, dtype=np.float32
        ),
        number_count_weight=np.asarray(args.number_count_weight, dtype=np.float32),
        operation_count_weight=np.asarray(
            args.operation_count_weight, dtype=np.float32
        ),
    )

    metadata = {
        "method": "ridge_candidate_correctness_scorer",
        "representation": args.representation,
        "regularization": args.regularization,
        "consensus_weight": args.consensus_weight,
        "arithmetic_penalty": args.arithmetic_penalty,
        "answer_step_penalty": args.answer_step_penalty,
        "group_mean_weight": args.group_mean_weight,
        "broad_arithmetic_penalty": args.broad_arithmetic_penalty,
        "number_count_weight": args.number_count_weight,
        "operation_count_weight": args.operation_count_weight,
        "training_split": "train",
        "questions": len(bank),
        "candidate_examples": len(labels),
        "positive_rate": float(labels.mean()),
        "training_seconds": time.perf_counter() - tick,
        "model_file": model_path.name,
        "model_bytes": model_path.stat().st_size,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
