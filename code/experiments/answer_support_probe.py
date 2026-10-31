"""Probe whether the declared answer is supported elsewhere in the trace."""

from fractions import Fraction
from pathlib import Path
import math
import re
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from answer_step_consistency import _reasoning_prefix, calculation_values
from common import load_inputs, parse_answer
from parseable_group_verifier import select_parseable_groups
from question_number_coverage_probe import candidate_labels, oof_l_scores
from text_fusion_verifier import text_fusion_scores


NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


def matches_answer(answer, value):
    answer_value = float(Fraction(answer))
    if math.isclose(answer_value, value, rel_tol=1e-6, abs_tol=1e-6):
        return True
    return Fraction(answer).denominator == 1 and math.isclose(
        answer_value, round(value), rel_tol=0.0, abs_tol=1e-6
    )


def body_numbers(text):
    output = []
    for match in NUMBER.finditer(_reasoning_prefix(text)):
        try:
            output.append(float(Fraction(match.group().replace(",", ""))))
        except (ValueError, ZeroDivisionError):
            pass
    return output


def answer_support_features(bank):
    output = np.zeros((len(bank), 4, 7), dtype=np.float32)
    for row_index, row in enumerate(bank):
        for candidate_index, candidate in enumerate(row["candidates"]):
            text = candidate["text"]
            answer = parse_answer(text)
            if answer is None:
                continue
            equations = calculation_values(text)
            numbers = body_numbers(text)
            equation_matches = [
                matches_answer(answer, value) for value in equations
            ]
            number_matches = [matches_answer(answer, value) for value in numbers]
            last_equation_support = bool(equation_matches and equation_matches[-1])
            last_number_support = bool(number_matches and number_matches[-1])
            output[row_index, candidate_index] = (
                float(any(equation_matches)),
                sum(equation_matches) / max(len(equation_matches), 1),
                float(last_equation_support),
                float(any(number_matches)),
                sum(number_matches) / max(len(number_matches), 1),
                float(last_number_support),
                np.log1p(sum(number_matches)),
            )
    return output


def standardize_rows(values):
    return (values - values.mean(axis=1, keepdims=True)) / (
        values.std(axis=1, keepdims=True) + 1e-6
    )


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank)
    validation_labels = candidate_labels("validation", validation_bank)
    folds = np.arange(len(train_bank)) % 5
    train_scores, _ = oof_l_scores(
        train_bank, train_arrays, train_labels, folds
    )
    validation_scores = text_fusion_scores(
        validation_bank,
        validation_arrays,
        CODE / "runs/model_l_text_fusion",
    )
    train_features = answer_support_features(train_bank)
    validation_features = answer_support_features(validation_bank)
    names = (
        "any_equation_support",
        "equation_support_fraction",
        "last_equation_support",
        "any_body_number_support",
        "body_number_support_fraction",
        "last_body_number_support",
        "body_support_count",
    )
    for feature_index, name in enumerate(names):
        train_feature = standardize_rows(train_features[:, :, feature_index])
        validation_feature = standardize_rows(
            validation_features[:, :, feature_index]
        )
        best = (-1.0, 0.0)
        for weight in np.arange(-0.20, 0.201, 0.01):
            selected = np.asarray(
                select_parseable_groups(
                    train_bank,
                    train_scores + weight * train_feature,
                    0.3,
                )
            )
            accuracy = train_labels[
                np.arange(len(train_labels)), selected
            ].mean()
            if accuracy > best[0]:
                best = (accuracy, float(weight))
        selected = np.asarray(
            select_parseable_groups(
                validation_bank,
                validation_scores + best[1] * validation_feature,
                0.3,
            )
        )
        validation_accuracy = validation_labels[
            np.arange(len(validation_labels)), selected
        ].mean()
        print(
            f"feature={name} oof={100 * best[0]:.4f} "
            f"weight={best[1]:.3f} validation={100 * validation_accuracy:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
