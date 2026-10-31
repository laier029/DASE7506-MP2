"""Model J: Model I plus a lower-weight unit-aware arithmetic check."""

from pathlib import Path

import numpy as np

from arithmetic_consistency import arithmetic_counts, broad_arithmetic_counts
from fused_answer_step_verifier import answer_step_scores
from parseable_group_verifier import select_parseable_groups


def broad_arithmetic_scores(bank, arrays, checkpoint):
    """Return Model J candidate scores before answer-group aggregation."""
    scores = answer_step_scores(bank, arrays, checkpoint)
    checkpoint = Path(checkpoint)
    model_path = checkpoint / "model.npz" if checkpoint.is_dir() else checkpoint
    with np.load(model_path, allow_pickle=False) as model:
        broad_penalty = float(model["broad_arithmetic_penalty"].item())

    broad_only_error = np.zeros((len(bank), 4), dtype=np.float32)
    for row_index, row in enumerate(bank):
        for candidate_index, candidate in enumerate(row["candidates"]):
            text = candidate["text"]
            narrow_incorrect = arithmetic_counts(text)[1]
            broad_incorrect = broad_arithmetic_counts(text)[1]
            broad_only_error[row_index, candidate_index] = float(
                broad_incorrect > 0 and narrow_incorrect == 0
            )

    return scores - broad_penalty * broad_only_error


def predict(bank, arrays, checkpoint):
    """Apply a small extra penalty to arithmetic errors hidden by unit labels."""
    scores = broad_arithmetic_scores(bank, arrays, checkpoint)
    checkpoint = Path(checkpoint)
    model_path = checkpoint / "model.npz" if checkpoint.is_dir() else checkpoint
    with np.load(model_path, allow_pickle=False) as model:
        group_mean_weight = float(model["group_mean_weight"].item())
    return select_parseable_groups(bank, scores, group_mean_weight)
