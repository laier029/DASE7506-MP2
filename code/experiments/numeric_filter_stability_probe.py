"""Check fractional-answer rejection across random Model O OOF folds."""

from fractions import Fraction
from pathlib import Path
import sys

import joblib
import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from calculation_coverage import calculation_unique_coverage
from cascade_stability_probe import (
    answer_structures,
    balanced_random_folds,
    model_m_oof,
)
from coverage_group_filter_probe import accuracy, answer_groups, select
from common import load_inputs
from group_ranker_probe import candidate_labels
from numeric_plausibility_probe import rejected_groups


def fractional_rule(question, value, groups):
    return value.denominator != 1


def apply_model_o_coverage(bank, scores, structures):
    coverage = calculation_unique_coverage(bank)
    normalized = (coverage - coverage.mean(axis=1, keepdims=True)) / (
        coverage.std(axis=1, keepdims=True) + 1e-6
    )
    adjusted = scores.copy()
    rows = structures == "2-1-1"
    adjusted[rows] += 0.4 * normalized[rows]
    return adjusted


def main():
    bank, arrays = load_inputs("train")
    labels = candidate_labels("train", bank).astype(bool)
    structures = answer_structures(bank)
    artifact = joblib.load(
        CODE / "runs/model_o_coverage_cascade/structure_text_model.joblib"
    )
    texts = [
        candidate["text"]
        for row in bank
        for candidate in row["candidates"]
    ]
    text_features = artifact["vectorizer"].transform(texts)
    rejected = rejected_groups(bank, fractional_rule)

    for seed in (7506, 11, 29, 47, 83):
        folds = (
            np.arange(len(bank)) % 5
            if seed == 7506
            else balanced_random_folds(len(bank), seed)
        )
        scores = model_m_oof(
            bank, arrays, labels, folds, text_features, structures
        )
        scores = apply_model_o_coverage(bank, scores, structures)
        base = select(bank, scores, [set() for _ in bank])
        filtered = select(bank, scores, rejected)
        print(
            f"seed={seed} base={accuracy(labels, base)} "
            f"filtered={accuracy(labels, filtered)} "
            f"delta={accuracy(labels, filtered) - accuracy(labels, base):+d}",
            flush=True,
        )


if __name__ == "__main__":
    main()
