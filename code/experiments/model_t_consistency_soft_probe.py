"""Recheck arithmetic and final-step soft penalties on top of Model T."""

from pathlib import Path
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from answer_step_consistency import answer_step_consistency
from arithmetic_consistency import (
    arithmetic_counts,
    broad_arithmetic_counts,
    plain_text_arithmetic_counts,
)
from common import load_inputs
from coverage_cascade_verifier import coverage_adjusted_scores
from group_ranker_probe import candidate_labels
from model_t_probe_utils import (
    FastFinalSelector,
    load_or_build_model_t_oof,
    standardize_rows,
)
from occurrence_plausibility_verifier import occurrence_adjusted_scores
from structure_text_fusion_verifier import answer_structures, structure_text_scores
from target_omitted_factor_verifier import target_adjusted_scores


def consistency_features(bank):
    output = np.zeros((len(bank), 4, 8), dtype=np.float32)
    for row_index, row in enumerate(bank):
        for candidate_index, candidate in enumerate(row["candidates"]):
            text = candidate["text"]
            narrow_checked, narrow_bad = arithmetic_counts(text)
            broad_checked, broad_bad = broad_arithmetic_counts(text)
            plain_checked, plain_bad = plain_text_arithmetic_counts(text)
            step = answer_step_consistency(text)
            output[row_index, candidate_index] = (
                float(narrow_bad > 0),
                narrow_bad / max(narrow_checked, 1),
                float(broad_bad > 0),
                broad_bad / max(broad_checked, 1),
                float(plain_bad > 0),
                plain_bad / max(plain_checked, 1),
                float(step is False),
                float(narrow_bad + broad_bad + plain_bad > 0 or step is False),
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
    train_features = consistency_features(train_bank)
    validation_features = consistency_features(validation_bank)
    train_structures = answer_structures(train_bank)
    validation_structures = answer_structures(validation_bank)
    names = (
        "narrow_any",
        "narrow_rate",
        "broad_any",
        "broad_rate",
        "plain_any",
        "plain_rate",
        "step_mismatch",
        "any_problem",
    )
    weights = np.round(np.arange(-0.5, 0.001, 0.025), 3)
    scopes = ("all", "3-1", "2-2", "2-1-1", "1-1-1-1")
    for feature_index, name in enumerate(names):
        train_raw = standardize_rows(train_features[:, :, feature_index])
        validation_raw = standardize_rows(validation_features[:, :, feature_index])
        for scope in scopes:
            train_feature = np.zeros_like(train_raw)
            validation_feature = np.zeros_like(validation_raw)
            train_rows = np.ones(len(train_bank), dtype=bool) if scope == "all" else train_structures == scope
            validation_rows = np.ones(len(validation_bank), dtype=bool) if scope == "all" else validation_structures == scope
            train_feature[train_rows] = train_raw[train_rows]
            validation_feature[validation_rows] = validation_raw[validation_rows]
            trials = []
            for weight in weights:
                count = train_selector.correct_count(
                    train_labels, score_sets[0] + weight * train_feature
                )
                trials.append((count, -abs(float(weight)), float(weight)))
            _, _, weight = max(trials)
            if weight == 0:
                continue
            train_deltas = [
                train_selector.correct_count(train_labels, s + weight * train_feature) - base
                for s, base in zip(score_sets, train_base)
            ]
            validation_delta = validation_selector.correct_count(
                validation_labels,
                validation_scores + weight * validation_feature,
            ) - validation_base
            print(
                f"feature={name} scope={scope} weight={weight:+.3f} "
                f"train_deltas={train_deltas} validation_delta={validation_delta:+d}",
                flush=True,
            )


if __name__ == "__main__":
    main()
