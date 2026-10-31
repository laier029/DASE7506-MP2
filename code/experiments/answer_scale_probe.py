"""Probe answer magnitude and within-question numeric-rank priors."""

from fractions import Fraction
from pathlib import Path
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from calculation_coverage import quantities
from common import load_inputs, parse_answer
from coverage_cascade_verifier import coverage_adjusted_scores
from group_ranker_probe import candidate_labels
from model_t_probe_utils import FastFinalSelector, load_or_build_model_t_oof, standardize_rows
from occurrence_plausibility_verifier import occurrence_adjusted_scores
from structure_text_fusion_verifier import answer_structures, structure_text_scores
from target_omitted_factor_verifier import target_adjusted_scores


def answer_scale_features(bank):
    output = np.zeros((len(bank), 4, 7), dtype=np.float32)
    for row_index, row in enumerate(bank):
        values = []
        for candidate in row["candidates"]:
            try:
                values.append(float(Fraction(parse_answer(candidate["text"]))))
            except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                values.append(np.nan)
        finite = np.asarray([value for value in values if np.isfinite(value)])
        logs = np.sign(finite) * np.log1p(np.abs(finite))
        median = float(np.median(logs)) if len(logs) else 0.0
        question_values = []
        for value in quantities(row["question"], include_words=True):
            try:
                question_values.append(abs(float(Fraction(value))))
            except (ValueError, ZeroDivisionError, OverflowError):
                continue
        question_max = max(question_values, default=1.0)
        for candidate_index, value in enumerate(values):
            if not np.isfinite(value):
                continue
            signed_log = np.sign(value) * np.log1p(abs(value))
            rank = float((finite < value).sum()) / max(len(finite) - 1, 1)
            output[row_index, candidate_index] = (
                signed_log,
                np.log1p(abs(value)),
                rank,
                abs(signed_log - median),
                float(value == finite.max()),
                float(value == finite.min()),
                np.log1p(abs(value) / max(question_max, 1e-6)),
            )
    return output


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    score_sets = load_or_build_model_t_oof(train_bank, train_arrays, train_labels)
    train_selector = FastFinalSelector(train_bank)
    validation_selector = FastFinalSelector(validation_bank)
    validation_scores = structure_text_scores(
        validation_bank, validation_arrays, CODE / "runs/model_t_final_cascade"
    )
    validation_scores = coverage_adjusted_scores(validation_bank, validation_scores)
    validation_scores = occurrence_adjusted_scores(validation_bank, validation_scores)
    validation_scores = target_adjusted_scores(validation_bank, validation_scores)
    train_base = [train_selector.correct_count(train_labels, s) for s in score_sets]
    validation_base = validation_selector.correct_count(validation_labels, validation_scores)
    train_raw = answer_scale_features(train_bank)
    validation_raw = answer_scale_features(validation_bank)
    train_structures = answer_structures(train_bank)
    validation_structures = answer_structures(validation_bank)
    names = ("signed_log", "absolute_log", "rank", "median_distance", "is_max", "is_min", "question_ratio")
    weights = np.round(np.arange(-0.5, 0.501, 0.025), 3)
    scopes = ("all", "3-1", "2-2", "2-1-1", "1-1-1-1")
    for feature_index, name in enumerate(names):
        train_z = standardize_rows(train_raw[:, :, feature_index])
        validation_z = standardize_rows(validation_raw[:, :, feature_index])
        for scope in scopes:
            train_feature = np.zeros_like(train_z)
            validation_feature = np.zeros_like(validation_z)
            train_rows = np.ones(len(train_bank), dtype=bool) if scope == "all" else train_structures == scope
            validation_rows = np.ones(len(validation_bank), dtype=bool) if scope == "all" else validation_structures == scope
            train_feature[train_rows] = train_z[train_rows]
            validation_feature[validation_rows] = validation_z[validation_rows]
            trials = []
            for weight in weights:
                count = train_selector.correct_count(train_labels, score_sets[0] + weight * train_feature)
                trials.append((count, -abs(float(weight)), float(weight)))
            _, _, weight = max(trials)
            if weight == 0:
                continue
            train_deltas = [
                train_selector.correct_count(train_labels, s + weight * train_feature) - base
                for s, base in zip(score_sets, train_base)
            ]
            validation_delta = validation_selector.correct_count(
                validation_labels, validation_scores + weight * validation_feature
            ) - validation_base
            print(
                f"feature={name} scope={scope} weight={weight:+.3f} "
                f"train_deltas={train_deltas} validation_delta={validation_delta:+d}",
                flush=True,
            )


if __name__ == "__main__":
    main()
