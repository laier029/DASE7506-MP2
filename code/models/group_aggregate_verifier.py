"""Model H: score distinct final-answer groups before selecting a trace."""

from collections import defaultdict
from pathlib import Path

import numpy as np

from common import parse_answer
from fused_answer_step_verifier import answer_step_scores


def predict(bank, arrays, checkpoint):
    """Combine the best and mean trace quality for each final-answer group."""
    scores = answer_step_scores(bank, arrays, checkpoint)
    checkpoint = Path(checkpoint)
    model_path = checkpoint / "model.npz" if checkpoint.is_dir() else checkpoint
    with np.load(model_path, allow_pickle=False) as model:
        group_mean_weight = float(model["group_mean_weight"].item())

    selected = []
    for row, row_scores in zip(bank, scores):
        groups = defaultdict(list)
        for index, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate["text"])
            key = ("missing", index) if answer is None else ("answer", answer)
            groups[key].append(index)

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
