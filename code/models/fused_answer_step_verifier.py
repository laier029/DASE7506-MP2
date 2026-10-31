"""Model G: Model F plus final-answer/last-calculation consistency."""

from pathlib import Path

import numpy as np

from answer_step_consistency import answer_step_consistency
from fused_arithmetic_verifier import arithmetic_scores


def answer_step_scores(bank, arrays, checkpoint):
    """Return Model G scores before selecting the maximum."""
    scores = arithmetic_scores(bank, arrays, checkpoint)

    checkpoint = Path(checkpoint)
    model_path = checkpoint / "model.npz" if checkpoint.is_dir() else checkpoint
    with np.load(model_path, allow_pickle=False) as model:
        answer_step_penalty = float(model["answer_step_penalty"].item())

    mismatch = np.zeros((len(bank), 4), dtype=np.float32)
    for row_index, row in enumerate(bank):
        for candidate_index, candidate in enumerate(row["candidates"]):
            consistent = answer_step_consistency(candidate["text"])
            mismatch[row_index, candidate_index] = float(consistent is False)

    return scores - answer_step_penalty * mismatch


def predict(bank, arrays, checkpoint):
    """Penalize an explicit mismatch between answer and last calculation."""
    return answer_step_scores(bank, arrays, checkpoint).argmax(axis=1).tolist()

