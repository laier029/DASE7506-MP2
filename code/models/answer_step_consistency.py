"""Compare a candidate's declared answer with its last explicit calculation."""

import math
from fractions import Fraction
import re

from common import parse_answer
from arithmetic_consistency import _normalize, _safe_numeric_value


_FINAL_HEADING = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:[*_]{0,2})?(?:✅\s*)?"
    r"(?:final\s+answer|answer)\s*:"
)
_HASH_ANSWER = re.compile(r"(?m)^\s*####")


def _reasoning_prefix(text):
    """Remove the declared-answer section so it cannot verify itself."""
    headings = list(_FINAL_HEADING.finditer(text))
    hash_answers = list(_HASH_ANSWER.finditer(text))
    if headings:
        return text[: headings[-1].start()]
    if hash_answers:
        return text[: hash_answers[-1].start()]
    return text


def _numeric_rhs_value(part):
    """Parse a numeric RHS, allowing Markdown and trailing unit words."""
    part = re.sub(r"\\text\s*\{[^{}]*\}", " ", part)
    part = re.sub(r"\\(?:mathrm|operatorname)\s*\{[^{}]*\}", " ", part)
    part = part.replace("**", "").replace("__", "").replace("`", "")
    part = part.replace("\\%", "").replace("%", "")
    part = part.strip().strip("{}.,;:!?")

    try:
        return _safe_numeric_value(part)
    except (SyntaxError, ValueError, ZeroDivisionError):
        pass

    # Accept a numeric expression followed only by a unit label, such as
    # "384 dollars".  Do not extract a number from arbitrary prose.
    match = re.fullmatch(r"([0-9,.+\-*/()\s]+)\s+[A-Za-z][A-Za-z /-]*", part)
    if not match:
        return None
    try:
        return _safe_numeric_value(match.group(1))
    except (SyntaxError, ValueError, ZeroDivisionError):
        return None


def calculation_values(text):
    """Return numeric right-hand-side results in reasoning order."""
    reasoning = _reasoning_prefix(str(text))
    normalized = _normalize(reasoning)
    normalized = normalized.replace("\\approx", "=")
    normalized = normalized.replace("\\Rightarrow", "\n")
    normalized = normalized.replace("\\implies", "\n")
    normalized = normalized.replace("\\\\", "\n")
    # Currency is commonly escaped inside LaTeX (for example ``\\$120``).
    # Remove the escape and symbol together so a stray backslash does not make
    # an otherwise numeric right-hand side unparsable.
    normalized = normalized.replace("\\$", "")
    normalized = normalized.replace("$", " ")

    values = []
    for line in normalized.splitlines():
        if "=" not in line:
            continue
        for part in line.split("=")[1:]:
            value = _numeric_rhs_value(part)
            if value is not None:
                values.append(value)
    return values


def last_step_value(text):
    """Return the rightmost numeric result of the last explicit equation."""
    values = calculation_values(text)
    return values[-1] if values else None


def answer_step_consistency(text):
    """Return True/False for comparable values, otherwise None."""
    answer = parse_answer(text)
    last_value = last_step_value(text)
    if answer is None or last_value is None:
        return None
    answer_value = float(Fraction(answer))
    if math.isclose(answer_value, last_value, rel_tol=1e-6, abs_tol=1e-6):
        return True
    # Integer answers are often explicitly rounded from a decimal last step.
    if Fraction(answer).denominator == 1 and math.isclose(
        answer_value, round(last_value), rel_tol=0.0, abs_tol=1e-6
    ):
        return True
    return False

