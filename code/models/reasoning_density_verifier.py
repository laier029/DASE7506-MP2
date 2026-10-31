"""Model K: Model J plus small within-question reasoning-density features."""

from pathlib import Path
import re

import numpy as np

from broad_arithmetic_group_verifier import broad_arithmetic_scores
from parseable_group_verifier import select_parseable_groups


_NUMBER_TOKEN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_OPERATORS = (" + ", " - ", "\\times", "×", "\\div", "÷")


def _standardize_within_question(values):
    centered = values - values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, keepdims=True)
    return centered / (scale + 1e-6)


def reasoning_density_scores(bank, arrays, checkpoint):
    """Return Model K candidate scores before answer-group aggregation."""
    scores = broad_arithmetic_scores(bank, arrays, checkpoint)
    checkpoint = Path(checkpoint)
    model_path = checkpoint / "model.npz" if checkpoint.is_dir() else checkpoint
    with np.load(model_path, allow_pickle=False) as model:
        number_weight = float(model["number_count_weight"].item())
        operation_weight = float(model["operation_count_weight"].item())

    number_count = np.zeros((len(bank), 4), dtype=np.float32)
    operation_count = np.zeros((len(bank), 4), dtype=np.float32)
    for row_index, row in enumerate(bank):
        for candidate_index, candidate in enumerate(row["candidates"]):
            text = candidate["text"]
            number_count[row_index, candidate_index] = np.log1p(
                len(_NUMBER_TOKEN.findall(text))
            )
            operation_count[row_index, candidate_index] = np.log1p(
                sum(text.count(operator) for operator in _OPERATORS)
            )

    scores += number_weight * _standardize_within_question(number_count)
    scores += operation_weight * _standardize_within_question(operation_count)
    return scores


def predict(bank, arrays, checkpoint):
    """Reward modest numeric and operational detail relative to peer traces."""
    scores = reasoning_density_scores(bank, arrays, checkpoint)
    checkpoint = Path(checkpoint)
    model_path = checkpoint / "model.npz" if checkpoint.is_dir() else checkpoint
    with np.load(model_path, allow_pickle=False) as model:
        group_mean_weight = float(model["group_mean_weight"].item())
    return select_parseable_groups(bank, scores, group_mean_weight)
