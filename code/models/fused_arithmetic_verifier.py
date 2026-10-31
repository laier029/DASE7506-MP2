"""Model F: fused verifier with a conservative arithmetic-error penalty."""

from pathlib import Path

import numpy as np

from arithmetic_consistency import arithmetic_counts
from fused_relative_consensus import fused_scores


def arithmetic_scores(bank, arrays, checkpoint):
    """Return fused scores after penalizing high-confidence arithmetic errors."""
    scores = fused_scores(bank, arrays, checkpoint)

    checkpoint = Path(checkpoint)
    model_path = checkpoint / "model.npz" if checkpoint.is_dir() else checkpoint
    with np.load(model_path, allow_pickle=False) as model:
        arithmetic_penalty = float(model["arithmetic_penalty"].item())

    has_error = np.zeros((len(bank), 4), dtype=np.float32)
    for row_index, row in enumerate(bank):
        for candidate_index, candidate in enumerate(row["candidates"]):
            _, incorrect = arithmetic_counts(candidate["text"])
            has_error[row_index, candidate_index] = float(incorrect > 0)

    return scores - arithmetic_penalty * has_error


def predict(bank, arrays, checkpoint):
    """Select the candidate with the highest arithmetic-adjusted score."""
    return arithmetic_scores(bank, arrays, checkpoint).argmax(axis=1).tolist()
