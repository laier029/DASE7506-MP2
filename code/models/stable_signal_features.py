"""Stable weak signals retained after five-fold robustness screening."""

from collections import Counter
import re

import numpy as np

from arithmetic_consistency import broad_arithmetic_counts
from calculation_coverage import calculation_context, calculation_quantities
from structure_text_fusion_verifier import answer_structures


IMPLICIT_PATTERNS = (
    (re.compile(r"\bboth\b", re.I), "2"),
    (re.compile(r"\bparents\b", re.I), "2"),
    (re.compile(r"\bround trips?\b", re.I), "2"),
    (re.compile(r"\bsecond coat\b", re.I), "2"),
    (re.compile(r"\b(?:daily|per day|each day)\b.*\bweek\b", re.I), "7"),
    (re.compile(r"\b(?:monthly|per month|each month)\b.*\byear\b", re.I), "12"),
    (re.compile(r"\bhours?\b.*\bminutes?\b", re.I), "60"),
    (re.compile(r"\bminutes?\b.*\bseconds?\b", re.I), "60"),
)

IMPLICIT_WEIGHT = 0.60
BROAD_2_1_1_WEIGHT = -0.15


def _standardize_rows(values):
    return (values - values.mean(axis=1, keepdims=True)) / (
        values.std(axis=1, keepdims=True) + 1e-6
    )


def implicit_factor_coverage(bank):
    """Measure use of factors implied by natural-language units or plurality."""
    output = np.zeros((len(bank), 4), dtype=np.float32)
    for row_index, row in enumerate(bank):
        required = Counter(
            value
            for pattern, value in IMPLICIT_PATTERNS
            if pattern.search(row["question"])
        )
        if not required:
            continue
        for candidate_index, candidate in enumerate(row["candidates"]):
            used = Counter(
                calculation_quantities(
                    calculation_context(candidate["text"]), include_words=True
                )
            )
            output[row_index, candidate_index] = sum(
                min(count, used[value]) for value, count in required.items()
            ) / sum(required.values())
    return output


def broad_problem_indicator(bank):
    """Return candidate broad-arithmetic error indicators."""
    return np.asarray(
        [
            [
                float(broad_arithmetic_counts(candidate["text"])[1] > 0)
                for candidate in row["candidates"]
            ]
            for row in bank
        ],
        dtype=np.float32,
    )


def stable_signal_adjusted_scores(bank, scores):
    """Add robust implicit-factor coverage and a scoped soft penalty."""
    output = scores.copy()
    output += IMPLICIT_WEIGHT * _standardize_rows(
        implicit_factor_coverage(bank)
    )
    broad = _standardize_rows(broad_problem_indicator(bank))
    rows = answer_structures(bank) == "2-1-1"
    output[rows] += BROAD_2_1_1_WEIGHT * broad[rows]
    return output
