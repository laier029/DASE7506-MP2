"""Probe whether a final total depends on enough prior arithmetic branches."""

import ast
from fractions import Fraction
import math
from pathlib import Path
import re
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from answer_step_consistency import _reasoning_prefix
from arithmetic_consistency import _normalize, _safe_numeric_value
from calculation_coverage import quantities
from common import load_inputs, parse_answer
from dimension_consistency_probe import choices, correct
from group_ranker_probe import candidate_labels
from model_t_probe_utils import load_or_build_model_t_oof, standardize_rows
from stable_signal_features import stable_signal_adjusted_scores
from stable_signal_verifier import stable_signal_scores


TARGETS = {
    "total": re.compile(r"\b(?:total|altogether|in all|all together|overall|combined)\b", re.I),
    "now_everything": re.compile(r"\b(?:now|currently|everything)\b", re.I),
}


def numeric_constants(expression):
    tree = ast.parse(expression.strip().replace(",", ""), mode="eval")
    output = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            output.append(float(node.value))
    return output


def equation_pairs(text):
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
            lhs = parts[index - 1].strip()
            rhs = parts[index].strip()
            lhs_match = re.search(r"([0-9,.+\-*/()\s]+)$", lhs)
            rhs_match = re.match(r"\s*([0-9,.+\-*/()\s]+)", rhs)
            lhs_expression = lhs_match.group(1).strip() if lhs_match else ""
            rhs_expression = rhs_match.group(1).strip() if rhs_match else ""
            try:
                lhs_value = _safe_numeric_value(lhs_expression)
                rhs_value = _safe_numeric_value(rhs_expression)
                constants = numeric_constants(lhs_expression)
            except (SyntaxError, ValueError, ZeroDivisionError):
                continue
            if not math.isclose(lhs_value, rhs_value, rel_tol=1e-3, abs_tol=1e-3):
                continue
            output.append((rhs_value, constants))
    return output


def dependency_features(bank, pattern):
    output = np.zeros((len(bank), 4, 2), dtype=np.float32)
    for row_index, row in enumerate(bank):
        if pattern.search(row["question"]) is None:
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
            depths = {}
            final_size = final_depth = 0.0
            for result, constants in equation_pairs(candidate["text"]):
                size = 0.0
                depth = 1.0
                for constant in constants:
                    key = round(constant, 8)
                    if key in dependencies and key not in question_values:
                        size += dependencies[key]
                        depth = max(depth, depths[key] + 1.0)
                    else:
                        size += 1.0
                key = round(result, 8)
                dependencies[key] = max(size, dependencies.get(key, 0.0))
                depths[key] = max(depth, depths.get(key, 0.0))
                if math.isclose(result, answer_value, rel_tol=1e-6, abs_tol=1e-6):
                    final_size = size
                    final_depth = depth
            output[row_index, candidate_index] = (final_size, final_depth)
    return output


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    train_scores = [
        stable_signal_adjusted_scores(train_bank, scores)
        for scores in load_or_build_model_t_oof(train_bank, train_arrays, train_labels)
    ]
    validation_scores = stable_signal_scores(
        validation_bank, validation_arrays, CODE / "runs/model_x_full_cascade"
    )
    empty_train = [set() for _ in train_bank]
    empty_validation = [set() for _ in validation_bank]
    base_train_choices = [choices(train_bank, scores, empty_train) for scores in train_scores]
    base_train = [correct(train_labels, selected) for selected in base_train_choices]
    base_validation_choices = choices(
        validation_bank, validation_scores, empty_validation
    )
    base_validation = correct(validation_labels, base_validation_choices)
    weights = (-0.20, -0.10, -0.05, -0.025, 0.0, 0.025, 0.05, 0.10, 0.20)
    for target_name, pattern in TARGETS.items():
        train_raw = dependency_features(train_bank, pattern)
        validation_raw = dependency_features(validation_bank, pattern)
        for feature_index, feature_name in enumerate(("leaf_count", "depth")):
            train_feature = standardize_rows(train_raw[:, :, feature_index])
            validation_feature = standardize_rows(validation_raw[:, :, feature_index])
            trials = []
            for weight in weights:
                selected = choices(
                    train_bank, train_scores[0] + weight * train_feature, empty_train
                )
                trials.append((correct(train_labels, selected), -abs(weight), weight))
            _, _, weight = max(trials)
            deltas = []
            for scores, base in zip(train_scores, base_train):
                selected = choices(
                    train_bank, scores + weight * train_feature, empty_train
                )
                deltas.append(correct(train_labels, selected) - base)
            proposed = choices(
                validation_bank,
                validation_scores + weight * validation_feature,
                empty_validation,
            )
            print(
                f"target={target_name} feature={feature_name} weight={weight:+.3f} "
                f"support={int((np.ptp(train_raw[:, :, feature_index], axis=1) > 0).sum())}/"
                f"{int((np.ptp(validation_raw[:, :, feature_index], axis=1) > 0).sum())} "
                f"oof_deltas={deltas} "
                f"validation_delta={correct(validation_labels, proposed) - base_validation:+d} "
                f"validation_changes={np.flatnonzero(proposed != base_validation_choices).tolist()}",
                flush=True,
            )


if __name__ == "__main__":
    main()
