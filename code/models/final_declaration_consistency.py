"""Detect answer groups whose explicit final declaration conflicts with scoring."""

from fractions import Fraction
import re

from common import parse_answer


_FINAL_HEADING = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:[*_]{0,2})?(?:✅\s*)?"
    r"(?:final\s+answer|answer)\s*:"
)
_BOXED = re.compile(
    r"\\boxed\s*\{\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)\s*\}"
)
_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


def declared_final_answer(text):
    """Return the first declared number after the last Final Answer heading."""
    matches = list(_FINAL_HEADING.finditer(str(text)))
    if not matches:
        return None
    tail = str(text)[matches[-1].end() :]
    boxes = _BOXED.findall(tail)
    if boxes:
        token = boxes[-1]
    else:
        match = _NUMBER.search(tail)
        token = match.group() if match else None
    if token is None:
        return None
    try:
        return str(Fraction(token.replace(",", "")))
    except (ValueError, ZeroDivisionError):
        return None


def has_final_declaration_mismatch(text):
    declared = declared_final_answer(text)
    return declared is not None and parse_answer(text) != declared


def final_declaration_mismatch_groups(bank):
    """Reject only groups for which every supporting candidate conflicts."""
    output = []
    for row in bank:
        groups = {}
        for candidate_index, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate["text"])
            if answer is not None:
                groups.setdefault(answer, []).append(candidate_index)
        output.append(
            {
                answer
                for answer, indices in groups.items()
                if all(
                    has_final_declaration_mismatch(
                        row["candidates"][index]["text"]
                    )
                    for index in indices
                )
            }
        )
    return output
