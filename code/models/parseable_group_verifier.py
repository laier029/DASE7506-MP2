"""Model I: Model H with protocol-safe rejection of unparseable answers."""

from collections import defaultdict
from pathlib import Path

import numpy as np

from common import parse_answer
from fused_answer_step_verifier import answer_step_scores


def select_parseable_groups(bank, scores, group_mean_weight):
    """Apply Model H aggregation while excluding unparseable candidates."""
    selected = []
    for row, row_scores in zip(bank, scores):
        groups = defaultdict(list)
        for index, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate["text"])
            if answer is not None:
                groups[answer].append(index)

        # The benchmark guarantees at least one correct candidate, whose final
        # answer must therefore be parseable by the official scorer.
        if not groups:
            raise ValueError("Question has no candidate with a parseable answer.")

        best_group_score = None
        best_candidate = None
        for indices in groups.values():
            values = row_scores[indices]
            candidate = indices[int(values.argmax())]
            group_score = float(values.max() + group_mean_weight * values.mean())
            if best_group_score is None or group_score > best_group_score:
                best_group_score = group_score
                best_candidate = candidate
        selected.append(best_candidate)

    return selected


def predict(bank, arrays, checkpoint):
    """Select only candidates whose final numeric answer can be scored."""
    scores = answer_step_scores(bank, arrays, checkpoint)
    checkpoint = Path(checkpoint)
    model_path = checkpoint / "model.npz" if checkpoint.is_dir() else checkpoint
    with np.load(model_path, allow_pickle=False) as model:
        group_mean_weight = float(model["group_mean_weight"].item())
    return select_parseable_groups(bank, scores, group_mean_weight)
