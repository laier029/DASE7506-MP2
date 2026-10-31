"""Conservative question-conditioned answer-rank signal."""

from fractions import Fraction
import re

import numpy as np

from common import parse_answer
from structure_text_fusion_verifier import answer_structures


REMAINDER_TARGET = re.compile(
    r"\b(?:left|remain(?:s|ing|ed)?|difference|saved|savings|behind)\b",
    re.I,
)
REMAINDER_RANK_WEIGHT = -0.025
DISAGREEMENT_REMAINDER_RANK_WEIGHT = -0.10


def _standardize_rows(values):
    return (values - values.mean(axis=1, keepdims=True)) / (
        values.std(axis=1, keepdims=True) + 1e-6
    )


def remainder_answer_rank(bank):
    """Return within-question numeric ranks only for remainder-like targets."""
    output = np.zeros((len(bank), 4), dtype=np.float32)
    for row_index, row in enumerate(bank):
        if REMAINDER_TARGET.search(row["question"]) is None:
            continue
        values = []
        for candidate in row["candidates"]:
            try:
                values.append(float(Fraction(parse_answer(candidate["text"]))))
            except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                values.append(np.nan)
        finite = np.asarray([value for value in values if np.isfinite(value)])
        if len(set(finite.tolist())) <= 1:
            continue
        for candidate_index, value in enumerate(values):
            if np.isfinite(value):
                output[row_index, candidate_index] = float(
                    (finite < value).sum()
                ) / max(len(finite) - 1, 1)
    return _standardize_rows(output)


def semantic_rank_adjusted_scores(bank, scores):
    """Apply the OOF-stable, validation-neutral weak rank preference."""
    feature = remainder_answer_rank(bank)
    output = scores + REMAINDER_RANK_WEIGHT * feature
    structures = answer_structures(bank)
    rows = np.isin(structures, ("2-2", "2-1-1"))
    output[rows] += (
        DISAGREEMENT_REMAINDER_RANK_WEIGHT - REMAINDER_RANK_WEIGHT
    ) * feature[rows]
    return output
