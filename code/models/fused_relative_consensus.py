"""Model E: relative-embedding scores fused with final-answer consensus."""

from collections import Counter
from pathlib import Path

import numpy as np

from common import parse_answer
from linear_verifier import linear_scores


def fused_scores(bank, arrays, checkpoint):
    """Return relative-linear scores with normalized answer-frequency credit."""
    scores = linear_scores(bank, arrays, checkpoint, relative=True)

    checkpoint = Path(checkpoint)
    model_path = checkpoint / "model.npz" if checkpoint.is_dir() else checkpoint
    with np.load(model_path, allow_pickle=False) as model:
        consensus_weight = float(model["consensus_weight"].item())

    consensus = np.zeros((len(bank), 4), dtype=np.float32)
    for row_index, row in enumerate(bank):
        answers = [parse_answer(candidate["text"]) for candidate in row["candidates"]]
        counts = Counter(answer for answer in answers if answer is not None)
        consensus[row_index] = [
            counts.get(answer, 0) / 4 if answer is not None else 0.0
            for answer in answers
        ]

    return scores + consensus_weight * consensus


def predict(bank, arrays, checkpoint):
    """Select the candidate with the highest fused score."""
    scores = fused_scores(bank, arrays, checkpoint)
    return scores.argmax(axis=1).tolist()
