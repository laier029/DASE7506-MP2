"""Recover from protocol answers hijacked by intermediate Markdown headings."""

from fractions import Fraction
import re

import numpy as np

from common import parse_answer


_BOXED = re.compile(
    r"\\boxed\s*\{\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)\s*\}"
)


def boxed_answer(text):
    """Return the canonical numeric value of the last simple boxed answer."""
    matches = _BOXED.findall(str(text))
    if not matches:
        return None
    try:
        return str(Fraction(matches[-1].replace(",", "")))
    except (ValueError, ZeroDivisionError):
        return None


def has_declared_answer_mismatch(text):
    """Whether protocol parsing and the last explicit box disagree."""
    boxed = boxed_answer(text)
    return boxed is not None and parse_answer(text) != boxed


def redirect_selected_mismatch(bank, scores, choices):
    """Redirect only a selected mismatched candidate to its boxed-answer group.

    If no candidate exposes the boxed value through the fixed protocol parser,
    preserve the original choice as a safe fallback.
    """
    output = list(choices)
    for row_index, (row, choice) in enumerate(zip(bank, choices)):
        selected_text = row["candidates"][choice]["text"]
        if not has_declared_answer_mismatch(selected_text):
            continue
        target = boxed_answer(selected_text)
        indices = [
            index
            for index, candidate in enumerate(row["candidates"])
            if parse_answer(candidate["text"]) == target
        ]
        if not indices:
            continue
        values = scores[row_index, indices]
        output[row_index] = indices[int(np.argmax(values))]
    return output


def redirect_consistent_boxed_endorsement(bank, scores, choices):
    """Use a unique boxed target endorsed by any parser-mismatched candidate.

    Multiple mismatched candidates must agree. Missing, conflicting or
    protocol-invisible targets preserve the original choice.
    """
    output = list(choices)
    for row_index, row in enumerate(bank):
        targets = {
            boxed_answer(candidate["text"])
            for candidate in row["candidates"]
            if has_declared_answer_mismatch(candidate["text"])
        }
        targets.discard(None)
        if len(targets) != 1:
            continue
        target = next(iter(targets))
        indices = [
            index
            for index, candidate in enumerate(row["candidates"])
            if parse_answer(candidate["text"]) == target
        ]
        if not indices:
            continue
        values = scores[row_index, indices]
        output[row_index] = indices[int(np.argmax(values))]
    return output
