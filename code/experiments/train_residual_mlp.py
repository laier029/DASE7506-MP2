"""Train a small nonlinear residual ranker on top of frozen Model G scores."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))

from common import DATA, load_inputs, parse_answer, read_jsonl
from fused_answer_step_verifier import answer_step_scores


def labels_for(split, bank):
    references = read_jsonl(DATA / f"gsm8k_{split}.jsonl")
    return np.asarray(
        [
            [
                parse_answer(candidate["text"]) == parse_answer(reference["answer"])
                for candidate in row["candidates"]
            ]
            for row, reference in zip(bank, references)
        ],
        dtype=np.float32,
    )


def standardized_relative(arrays, mean, scale):
    embeddings = arrays["embeddings"].astype(np.float32, copy=True)
    embeddings -= embeddings.mean(axis=1, keepdims=True)
    return (embeddings - mean) / scale


def rank_accuracy(base, features, labels, w1, b1, w2):
    hidden = np.tanh(features @ w1 + b1)
    scores = base + hidden @ w2
    selected = scores.argmax(axis=1)
    return float(labels[np.arange(len(labels)), selected].mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=7506)
    args = parser.parse_args()

    tick = time.perf_counter()
    train_bank, train_arrays = load_inputs("train")
    val_bank, val_arrays = load_inputs("validation")
    train_labels = labels_for("train", train_bank)
    val_labels = labels_for("validation", val_bank)
    train_base = answer_step_scores(
        train_bank, train_arrays, args.base_checkpoint
    ).astype(np.float32)
    val_base = answer_step_scores(val_bank, val_arrays, args.base_checkpoint).astype(
        np.float32
    )

    model_path = args.base_checkpoint / "model.npz"
    with np.load(model_path, allow_pickle=False) as base_model:
        mean = base_model["mean"].astype(np.float32)
        scale = base_model["scale"].astype(np.float32)

    train_x = standardized_relative(train_arrays, mean, scale)
    val_x = standardized_relative(val_arrays, mean, scale)

    rng = np.random.default_rng(args.seed)
    w1 = rng.normal(0.0, np.sqrt(1.0 / 1024), (1024, args.hidden)).astype(
        np.float32
    )
    b1 = np.zeros(args.hidden, dtype=np.float32)
    w2 = rng.normal(0.0, 0.001, args.hidden).astype(np.float32)
    params = [w1, b1, w2]
    first = [np.zeros_like(param) for param in params]
    second = [np.zeros_like(param) for param in params]

    best = {
        "epoch": 0,
        "validation_accuracy": rank_accuracy(
            val_base, val_x, val_labels, w1, b1, np.zeros_like(w2)
        ),
        "train_accuracy": rank_accuracy(
            train_base, train_x, train_labels, w1, b1, np.zeros_like(w2)
        ),
        "weights": [w1.copy(), b1.copy(), np.zeros_like(w2)],
    }
    step = 0

    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(train_x))
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            x = train_x[batch]
            y = train_labels[batch]
            base = train_base[batch]

            hidden = np.tanh(x @ w1 + b1)
            scores = base + hidden @ w2
            scores -= scores.max(axis=1, keepdims=True)
            probabilities = np.exp(scores)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            correct_mass = (probabilities * y).sum(axis=1, keepdims=True)
            score_gradient = probabilities - probabilities * y / correct_mass
            score_gradient /= len(batch)

            flat_hidden = hidden.reshape(-1, args.hidden)
            flat_gradient = score_gradient.reshape(-1)
            grad_w2 = flat_hidden.T @ flat_gradient + args.weight_decay * w2
            hidden_gradient = score_gradient[..., None] * w2
            preactivation_gradient = hidden_gradient * (1.0 - hidden * hidden)
            flat_x = x.reshape(-1, 1024)
            flat_preactivation = preactivation_gradient.reshape(-1, args.hidden)
            grad_w1 = flat_x.T @ flat_preactivation + args.weight_decay * w1
            grad_b1 = flat_preactivation.sum(axis=0)
            gradients = [grad_w1, grad_b1, grad_w2]

            step += 1
            for index, (param, gradient) in enumerate(zip(params, gradients)):
                first[index] = 0.9 * first[index] + 0.1 * gradient
                second[index] = 0.999 * second[index] + 0.001 * gradient * gradient
                first_hat = first[index] / (1.0 - 0.9**step)
                second_hat = second[index] / (1.0 - 0.999**step)
                param -= args.learning_rate * first_hat / (
                    np.sqrt(second_hat) + 1e-8
                )

        train_accuracy = rank_accuracy(
            train_base, train_x, train_labels, w1, b1, w2
        )
        validation_accuracy = rank_accuracy(
            val_base, val_x, val_labels, w1, b1, w2
        )
        print(
            f"epoch={epoch:03d} train={100 * train_accuracy:.4f} "
            f"validation={100 * validation_accuracy:.4f}"
        )
        if validation_accuracy > best["validation_accuracy"]:
            best = {
                "epoch": epoch,
                "validation_accuracy": validation_accuracy,
                "train_accuracy": train_accuracy,
                "weights": [w1.copy(), b1.copy(), w2.copy()],
            }

    args.output.mkdir(parents=True, exist_ok=True)
    output_base = args.output / "model.npz"
    shutil.copy2(model_path, output_base)
    best_w1, best_b1, best_w2 = best.pop("weights")
    output_model = args.output / "mlp.npz"
    np.savez(
        output_model,
        w1=best_w1,
        b1=best_b1,
        w2=best_w2,
        seed=np.asarray(args.seed),
    )
    metadata = {
        **best,
        "method": "model_g_plus_tanh_residual_ranker",
        "base_checkpoint": str(args.base_checkpoint),
        "hidden": args.hidden,
        "epochs_searched": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "training_seconds": time.perf_counter() - tick,
        "asset_bytes": output_base.stat().st_size + output_model.stat().st_size,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
