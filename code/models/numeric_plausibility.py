"""High-precision final-answer plausibility gates learned from training data."""

from collections import defaultdict
from fractions import Fraction
import re

from common import parse_answer


_NEGATIVE_INPUT = re.compile(r"(?<!\w)-\s*\d")


def fractional_answer_groups(bank):
    """Return parsed answer groups whose canonical value is non-integral."""
    rejected = []
    for row in bank:
        groups = defaultdict(list)
        for candidate_index, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate["text"])
            if answer is not None:
                groups[answer].append(candidate_index)
        rejected.append(
            {
                answer
                for answer in groups
                if Fraction(answer).denominator != 1
            }
        )
    return rejected


def safe_nonpositive_answer_groups(bank):
    """Reject nonpositive answers unless the question contains a negative input."""
    rejected = []
    for row in bank:
        row_rejected = set()
        if _NEGATIVE_INPUT.search(row["question"]) is None:
            for candidate in row["candidates"]:
                answer = parse_answer(candidate["text"])
                if answer is not None and Fraction(answer) <= 0:
                    row_rejected.add(answer)
        rejected.append(row_rejected)
    return rejected


def implausible_answer_groups(bank):
    """Combine the independently zero-false-negative numeric gates."""
    fractional = fractional_answer_groups(bank)
    nonpositive = safe_nonpositive_answer_groups(bank)
    return [left | right for left, right in zip(fractional, nonpositive)]
