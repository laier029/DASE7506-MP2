"""Measure whether a candidate actually uses the question's quantities."""

from collections import Counter
from fractions import Fraction
import re

import numpy as np


NUMBER = re.compile(r"(?<![\w.])(?:\d+(?:,\d{3})*|\.\d+)(?:\.\d+)?(?:/\d+)?")
# Keep LaTeX delimiters separate from ordinary currency markers such as
# ``$20``. A permissive ``$...$`` pattern can otherwise pair two prices and
# incorrectly treat all prose between them as one mathematical expression.
MATH_FRAGMENT = re.compile(
    r"(?<!\\)\$\$(?P<block>.+?)(?<!\\)\$\$|"
    r"(?<![\\$])\$(?![$\d])(?P<inline>[^\n$]+?)(?<!\\)\$(?!\$)",
    re.DOTALL,
)
FINAL_MARKER = re.compile(
    r"(?:####|###\s*(?:final\s+answer|answer)|final\s+answer\s*:|"
    r"answer\s*:)",
    re.IGNORECASE,
)
# Do not confuse a Markdown bullet with subtraction.
OPERATOR = re.compile(
    r"(?:=|\+|(?<=[\d)])\s*-\s*(?=[\d(])|/|\\times|\\div|×|÷)"
)
NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
    "hundred": "100",
    "thousand": "1000",
    "twice": "2",
    "double": "2",
    "half": "1/2",
    "quarter": "1/4",
    "dozen": "12",
}
WORD_PATTERN = re.compile(
    r"\b(" + "|".join(NUMBER_WORDS) + r")\b", re.IGNORECASE
)
SIMPLE_LATEX_FRACTION = re.compile(
    r"\\frac\s*\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}"
    r"\s*\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}"
)


def normalize_number(token):
    token = token.replace(",", "")
    if token.startswith("."):
        token = "0" + token
    try:
        return str(Fraction(token))
    except (ValueError, ZeroDivisionError):
        return token


def quantities(text, *, include_words=False):
    values = [normalize_number(match.group()) for match in NUMBER.finditer(text)]
    if include_words:
        values.extend(
            normalize_number(NUMBER_WORDS[match.group().lower()])
            for match in WORD_PATTERN.finditer(text)
        )
    return values


def calculation_quantities(text, *, include_words=False):
    """Return calculated values plus fraction numerator/denominator inputs."""
    normalized = SIMPLE_LATEX_FRACTION.sub(r"\1/\2", text)
    values = quantities(normalized, include_words=include_words)
    for match in NUMBER.finditer(normalized):
        token = match.group().replace(",", "")
        if "/" not in token:
            continue
        numerator, denominator = token.split("/", 1)
        values.extend((normalize_number(numerator), normalize_number(denominator)))
    return values


def reasoning_body(text):
    match = FINAL_MARKER.search(text)
    return text[: match.start()] if match else text


def calculation_context(text):
    body = reasoning_body(text)
    fragments = [
        match.group("block") or match.group("inline")
        for match in MATH_FRAGMENT.finditer(body)
        if OPERATOR.search(match.group("block") or match.group("inline"))
    ]
    # Math fragments have already been collected.  Remove them before scanning
    # plain-text lines so an inline equation is not counted a second time.
    nonmath_body = MATH_FRAGMENT.sub(" ", body)
    lines = [line for line in nonmath_body.splitlines() if OPERATOR.search(line)]
    return "\n".join(fragments + lines)


def calculation_unique_coverage(bank):
    """Return fraction of unique question quantities used in calculations."""
    output = np.zeros((len(bank), 4), dtype=np.float32)
    for row_index, row in enumerate(bank):
        question_values = set(quantities(row["question"], include_words=True))
        for candidate_index, candidate in enumerate(row["candidates"]):
            calculation_values = set(
                calculation_quantities(
                    calculation_context(candidate["text"]),
                    include_words=True,
                )
            )
            output[row_index, candidate_index] = (
                sum(value in calculation_values for value in question_values)
                / max(len(question_values), 1)
            )
    return output


def calculation_occurrence_coverage(bank):
    """Return occurrence-aware question-quantity calculation coverage."""
    output = np.zeros((len(bank), 4), dtype=np.float32)
    for row_index, row in enumerate(bank):
        question_values = quantities(row["question"], include_words=True)
        question_counts = Counter(question_values)
        for candidate_index, candidate in enumerate(row["candidates"]):
            calculation_counts = Counter(
                calculation_quantities(
                    calculation_context(candidate["text"]),
                    include_words=True,
                )
            )
            output[row_index, candidate_index] = sum(
                min(count, calculation_counts[value])
                for value, count in question_counts.items()
            ) / max(len(question_values), 1)
    return output
