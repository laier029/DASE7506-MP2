"""Model G scores plus a small tanh residual network over relative embeddings."""

from pathlib import Path

import numpy as np

from fused_answer_step_verifier import answer_step_scores


def predict(bank, arrays, checkpoint):
    checkpoint = Path(checkpoint)
    scores = answer_step_scores(bank, arrays, checkpoint)

    with np.load(checkpoint / "model.npz", allow_pickle=False) as base:
        mean = base["mean"]
        scale = base["scale"]
    with np.load(checkpoint / "mlp.npz", allow_pickle=False) as model:
        w1 = model["w1"]
        b1 = model["b1"]
        w2 = model["w2"]

    embeddings = arrays["embeddings"].astype(np.float32, copy=True)
    embeddings -= embeddings.mean(axis=1, keepdims=True)
    features = (embeddings - mean) / scale
    scores += np.tanh(features @ w1 + b1) @ w2
    return scores.argmax(axis=1).tolist()
