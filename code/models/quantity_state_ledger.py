"""Generic arithmetic dependency ledger for question-conditioned verification.

The ledger deliberately avoids target-specific keyword rules.  It reconstructs
the arithmetic dependency chain that produces a candidate's parsed final
answer, then describes how much of the question's numerical state reaches that
answer.  A downstream learned ranker can decide when deep or broad dependency
coverage is useful for the wording of a particular question.
"""

from collections import Counter
from fractions import Fraction
import ast
import math
import re

import numpy as np

from answer_step_consistency import _reasoning_prefix
from arithmetic_consistency import _normalize, _safe_numeric_value
from calculation_coverage import quantities
from common import parse_answer


FEATURE_NAMES = (
    "has_final_equation",
    "terminal_position",
    "calculations_after",
    "conservative_leaf_log",
    "aggressive_leaf_log",
    "conservative_depth",
    "aggressive_depth",
    "question_unique_coverage",
    "question_occurrence_coverage",
    "extra_leaf_ratio",
    "terminal_leaf_growth",
    "expanded_intermediate_ratio",
    "final_add_share",
    "final_subtract_share",
    "final_multiply_share",
    "final_divide_share",
    "final_multi_operator",
    "equation_count_log",
)


_TRAILING_EXPRESSION = re.compile(r"([0-9,.+\-*/()\s]+)$")
_LEADING_EXPRESSION = re.compile(r"\s*([0-9,.+\-*/()\s]+)")


def _round_value(value):
    return round(float(value), 8)


def _fraction_value(token):
    try:
        return _round_value(Fraction(token))
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None


def _expression_summary(expression):
    """Return value, numeric leaves and arithmetic operator counts."""
    cleaned = expression.strip().replace(",", "")
    tree = ast.parse(cleaned, mode="eval")
    value = float(_safe_numeric_value(cleaned))
    constants = []
    operators = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            constants.append(float(node.value))
        elif isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Add):
                operators["add"] += 1
            elif isinstance(node.op, ast.Sub):
                operators["subtract"] += 1
            elif isinstance(node.op, ast.Mult):
                operators["multiply"] += 1
            elif isinstance(node.op, ast.Div):
                operators["divide"] += 1
    return value, constants, operators


def _equation_steps(text):
    """Extract ordered arithmetic equations as result/dependency records."""
    normalized = _normalize(_reasoning_prefix(text))
    normalized = normalized.replace("\\Rightarrow", "\n")
    normalized = normalized.replace("\\implies", "\n")
    normalized = normalized.replace("\\\\", "\n")
    normalized = normalized.replace("$", " ")
    normalized = normalized.replace("**", "")
    normalized = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", normalized)
    normalized = re.sub(r"\\text\s*\{[^{}]*\}", " ", normalized)
    output = []
    for line in normalized.splitlines():
        if "=" not in line:
            continue
        parts = line.split("=")
        for index in range(1, len(parts)):
            left_match = _TRAILING_EXPRESSION.search(parts[index - 1])
            right_match = _LEADING_EXPRESSION.match(parts[index])
            left = left_match.group(1).strip() if left_match else ""
            right = right_match.group(1).strip() if right_match else ""
            if not left or not right:
                continue
            try:
                left_value, left_constants, left_ops = _expression_summary(left)
                right_value, right_constants, right_ops = _expression_summary(right)
            except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
                continue
            if not math.isclose(
                left_value, right_value, rel_tol=1e-3, abs_tol=1e-3
            ):
                continue
            left_operator_count = sum(left_ops.values())
            right_operator_count = sum(right_ops.values())
            if right_operator_count > left_operator_count:
                result = left_value
                constants = right_constants
                operators = right_ops
            else:
                result = right_value
                constants = left_constants
                operators = left_ops
            if not constants:
                continue
            output.append(
                {
                    "result": _round_value(result),
                    "constants": [_round_value(value) for value in constants],
                    "operators": operators,
                }
            )
    return output


def _question_counter(question):
    values = (
        _fraction_value(token)
        for token in quantities(question, include_words=True)
    )
    return Counter(value for value in values if value is not None)


def _expanded_steps(text, question_values):
    """Expand intermediate results into conservative and aggressive ledgers."""
    nodes = {}
    output = []
    for raw in _equation_steps(text):
        conservative = Counter()
        aggressive = Counter()
        conservative_depth = aggressive_depth = 1.0
        expanded = 0
        for constant in raw["constants"]:
            prior = nodes.get(constant)
            if prior is None:
                conservative[constant] += 1
                aggressive[constant] += 1
                continue
            aggressive.update(prior["aggressive"])
            aggressive_depth = max(
                aggressive_depth, prior["aggressive_depth"] + 1.0
            )
            expanded += 1
            if constant in question_values:
                conservative[constant] += 1
            else:
                conservative.update(prior["conservative"])
                conservative_depth = max(
                    conservative_depth, prior["conservative_depth"] + 1.0
                )
        state = {
            **raw,
            "conservative": conservative,
            "aggressive": aggressive,
            "conservative_depth": conservative_depth,
            "aggressive_depth": aggressive_depth,
            "expanded": expanded,
        }
        nodes[raw["result"]] = state
        output.append(state)
    return output


def _coverage(leaves, question_values):
    if not question_values:
        return 0.0, 0.0
    unique = sum(value in leaves for value in question_values) / len(question_values)
    occurrence = sum(
        min(count, leaves.get(value, 0))
        for value, count in question_values.items()
    ) / sum(question_values.values())
    return float(unique), float(occurrence)


def _candidate_features(question, text):
    question_values = _question_counter(question)
    steps = _expanded_steps(text, question_values)
    output = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    output[-1] = np.log1p(len(steps))
    answer = _fraction_value(parse_answer(text))
    if answer is None or not steps:
        return output
    matches = [
        index
        for index, step in enumerate(steps)
        if math.isclose(step["result"], answer, rel_tol=1e-6, abs_tol=1e-6)
    ]
    if not matches:
        return output
    final_index = matches[-1]
    final = steps[final_index]
    conservative = final["conservative"]
    aggressive = final["aggressive"]
    unique_coverage, occurrence_coverage = _coverage(
        conservative, question_values
    )
    leaf_count = sum(conservative.values())
    aggressive_count = sum(aggressive.values())
    extra = sum(
        count for value, count in conservative.items() if value not in question_values
    )
    prior_leaf_max = max(
        (sum(step["conservative"].values()) for step in steps[:final_index]),
        default=0,
    )
    growth = leaf_count - prior_leaf_max
    operator_total = sum(final["operators"].values())
    output[:] = (
        1.0,
        (final_index + 1) / len(steps),
        (len(steps) - final_index - 1) / len(steps),
        np.log1p(leaf_count),
        np.log1p(aggressive_count),
        final["conservative_depth"],
        final["aggressive_depth"],
        unique_coverage,
        occurrence_coverage,
        extra / max(leaf_count, 1),
        np.sign(growth) * np.log1p(abs(growth)),
        final["expanded"] / max(len(final["constants"]), 1),
        final["operators"]["add"] / max(operator_total, 1),
        final["operators"]["subtract"] / max(operator_total, 1),
        final["operators"]["multiply"] / max(operator_total, 1),
        final["operators"]["divide"] / max(operator_total, 1),
        float(operator_total > 1),
        np.log1p(len(steps)),
    )
    return output


def quantity_state_ledger_features(bank):
    """Return generic ledger features with shape ``[N, 4, F]``."""
    return np.asarray(
        [
            [
                _candidate_features(row["question"], candidate["text"])
                for candidate in row["candidates"]
            ]
            for row in bank
        ],
        dtype=np.float32,
    )
