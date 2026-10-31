"""Shared inference code for the embedding-only linear verifiers."""

from pathlib import Path

import numpy as np


def linear_scores(bank, arrays, checkpoint, *, relative):
    """Load a frozen ridge scorer and return one score per candidate."""
    if checkpoint is None:
        raise ValueError("A checkpoint directory or .npz file is required.")

    checkpoint = Path(checkpoint)
    model_path = checkpoint / "model.npz" if checkpoint.is_dir() else checkpoint

    with np.load(model_path, allow_pickle=False) as model:
        mean = model["mean"]
        scale = model["scale"]
        weights = model["weights"]
        trained_relative = bool(model["relative"].item())

    if trained_relative != relative:
        raise ValueError("Checkpoint representation does not match the method.")

    embeddings = arrays["embeddings"].astype(np.float32, copy=True)
    if embeddings.shape != (len(bank), 4, 1024):
        raise ValueError("Expected embeddings with shape [N, 4, 1024].")

    if relative:
        embeddings -= embeddings.mean(axis=1, keepdims=True)

    features = (embeddings - mean) / scale
    return features @ weights


def predict_with_linear_model(bank, arrays, checkpoint, *, relative):
    """Select the highest-scoring candidate from a frozen ridge scorer."""
    scores = linear_scores(bank, arrays, checkpoint, relative=relative)
    return scores.argmax(axis=1).tolist()
