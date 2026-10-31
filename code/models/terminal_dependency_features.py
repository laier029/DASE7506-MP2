"""Terminal arithmetic-dependency coverage for cumulative targets."""

import ast
from fractions import Fraction
import math
import re

import numpy as np

from answer_step_consistency import _reasoning_prefix
from arithmetic_consistency import _normalize, _safe_numeric_value
from calculation_coverage import quantities
from common import parse_answer
from structure_text_fusion_verifier import answer_structures


CUMULATIVE_TARGET = re.compile(r"\b(?:now|currently|everything)\b", re.I)
DEPENDENCY_WEIGHT = 0.40
THREE_ONE_DEPENDENCY_WEIGHT = 0.50


def _standardize_rows(values):
    return (values - values.mean(axis=1, keepdims=True)) / (
        values.std(axis=1, keepdims=True) + 1e-6
    )


def _numeric_constants(expression):
    tree = ast.parse(expression.strip().replace(",", ""), mode="eval")
    return [
        float(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float)
    ]


def _equation_pairs(text):
    normalized = _normalize(_reasoning_prefix(text))
    normalized = normalized.replace("\\Rightarrow", "\n")
    normalized = normalized.replace("\\implies", "\n")
    normalized = normalized.replace("\\\\", "\n")
    normalized = normalized.replace("$", " ")
    normalized = re.sub(r"\\text\s*\{[^{}]*\}", " ", normalized)
    output = []
    for line in normalized.splitlines():
        if "=" not in line:
            continue
        parts = line.split("=")
        for index in range(1, len(parts)):
            lhs_match = re.search(r"([0-9,.+\-*/()\s]+)$", parts[index - 1])
            rhs_match = re.match(r"\s*([0-9,.+\-*/()\s]+)", parts[index])
            lhs = lhs_match.group(1).strip() if lhs_match else ""
            rhs = rhs_match.group(1).strip() if rhs_match else ""
            try:
                lhs_value = _safe_numeric_value(lhs)
                rhs_value = _safe_numeric_value(rhs)
                constants = _numeric_constants(lhs)
            except (SyntaxError, ValueError, ZeroDivisionError):
                continue
            if math.isclose(lhs_value, rhs_value, rel_tol=1e-3, abs_tol=1e-3):
                output.append((rhs_value, constants))
    return output


def terminal_dependency_coverage(bank):
    """Return expanded leaf counts for the equation producing each answer."""
    output = np.zeros((len(bank), 4), dtype=np.float32)
    for row_index, row in enumerate(bank):
        if CUMULATIVE_TARGET.search(row["question"]) is None:
            continue
        question_values = set()
        for token in quantities(row["question"], include_words=True):
            try:
                question_values.add(round(float(Fraction(token)), 8))
            except (ValueError, ZeroDivisionError, OverflowError):
                continue
        for candidate_index, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate["text"])
            if answer is None:
                continue
            try:
                answer_value = float(Fraction(answer))
            except (ValueError, ZeroDivisionError, OverflowError):
                continue
            dependencies = {}
            final_size = 0.0
            for result, constants in _equation_pairs(candidate["text"]):
                size = 0.0
                for constant in constants:
                    key = round(constant, 8)
                    if key in dependencies and key not in question_values:
                        size += dependencies[key]
                    else:
                        size += 1.0
                key = round(result, 8)
                dependencies[key] = max(size, dependencies.get(key, 0.0))
                if math.isclose(result, answer_value, rel_tol=1e-6, abs_tol=1e-6):
                    final_size = size
            output[row_index, candidate_index] = final_size
    return _standardize_rows(output)


def terminal_dependency_adjusted_scores(bank, scores):
    feature = terminal_dependency_coverage(bank)
    output = scores + DEPENDENCY_WEIGHT * feature
    rows = answer_structures(bank) == "3-1"
    output[rows] += (
        THREE_ONE_DEPENDENCY_WEIGHT - DEPENDENCY_WEIGHT
    ) * feature[rows]
    return output
