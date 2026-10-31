"""Probe whether candidate conclusions preserve the question's target concept."""

from pathlib import Path
import re
import sys

import joblib
import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from answer_step_consistency import _reasoning_prefix
from calculation_coverage import calculation_occurrence_coverage
from cascade_stability_probe import (
    answer_structures,
    balanced_random_folds,
    model_m_oof,
)
from cascade_verifier import unanimous_broad_error_groups
from common import load_inputs, parse_answer
from coverage_cascade_verifier import coverage_adjusted_scores
from group_ranker_probe import candidate_labels
from numeric_filter_stability_probe import apply_model_o_coverage
from numeric_plausibility import fractional_answer_groups
from occurrence_plausibility_verifier import occurrence_adjusted_scores
from structure_text_fusion_verifier import structure_text_scores


TARGETS = (
    "difference",
    "reduction",
    "change",
    "remaining",
    "left",
    "saving",
    "saved",
    "total",
    "sum",
    "profit",
    "cost",
    "price",
    "time",
    "distance",
    "number",
    "amount",
)


def target_features(bank):
    output = np.zeros((len(bank), 4, 4), dtype=np.float32)
    for row_index, row in enumerate(bank):
        question = row["question"].lower()
        targets = [target for target in TARGETS if target in question]
        if not targets:
            continue
        for candidate_index, candidate in enumerate(row["candidates"]):
            reasoning = _reasoning_prefix(candidate["text"]).lower()
            tail = reasoning[-300:]
            present = [target in tail for target in targets]
            explicit = [
                re.search(
                    rf"(?:final|total)\W{{0,20}}{re.escape(target)}|"
                    rf"{re.escape(target)}\W{{0,20}}(?:is|=)",
                    tail,
                )
                is not None
                for target in targets
            ]
            headings = "\n".join(
                line for line in tail.splitlines() if "#" in line or "**" in line
            )
            heading = [target in headings for target in targets]
            positions = [reasoning.rfind(target) for target in targets]
            output[row_index, candidate_index] = (
                sum(present) / len(targets),
                sum(explicit) / len(targets),
                sum(heading) / len(targets),
                max(positions) / max(len(reasoning), 1),
            )
    return output


def standardize_rows(values):
    return (values - values.mean(axis=1, keepdims=True)) / (
        values.std(axis=1, keepdims=True) + 1e-6
    )


def allowed_candidates(bank):
    arithmetic = unanimous_broad_error_groups(bank)
    fractional = fractional_answer_groups(bank)
    allowed = np.zeros((len(bank), 4), dtype=bool)
    for row_index, row in enumerate(bank):
        rejected = arithmetic[row_index] | fractional[row_index]
        for candidate_index, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate["text"])
            allowed[row_index, candidate_index] = (
                answer is not None and answer not in rejected
            )
        if not allowed[row_index].any():
            allowed[row_index] = [
                parse_answer(candidate["text"]) is not None
                for candidate in row["candidates"]
            ]
    return allowed


def structure_correct(labels, scores, rows, allowed):
    adjusted = scores[rows].copy()
    adjusted[~allowed[rows]] = -np.inf
    choices = adjusted.argmax(axis=1)
    return int(labels[rows][np.arange(rows.sum()), choices].sum())


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    train_structures = answer_structures(train_bank)
    validation_structures = answer_structures(validation_bank)
    train_rows = train_structures == "1-1-1-1"
    validation_rows = validation_structures == "1-1-1-1"
    train_allowed = allowed_candidates(train_bank)
    validation_allowed = allowed_candidates(validation_bank)

    artifact = joblib.load(
        CODE / "runs/model_r_omitted_factor/structure_text_model.joblib"
    )
    texts = [
        candidate["text"]
        for row in train_bank
        for candidate in row["candidates"]
    ]
    text_features = artifact["vectorizer"].transform(texts)
    occurrence = calculation_occurrence_coverage(train_bank)
    occurrence = standardize_rows(occurrence)
    score_sets = []
    for seed in (7506, 11, 29, 47, 83):
        folds = (
            np.arange(len(train_bank)) % 5
            if seed == 7506
            else balanced_random_folds(len(train_bank), seed)
        )
        scores = model_m_oof(
            train_bank,
            train_arrays,
            train_labels,
            folds,
            text_features,
            train_structures,
        )
        scores = apply_model_o_coverage(
            train_bank, scores, train_structures
        )
        scores[train_rows] += 0.225 * occurrence[train_rows]
        score_sets.append(scores)

    validation_scores = structure_text_scores(
        validation_bank,
        validation_arrays,
        CODE / "runs/model_r_omitted_factor",
    )
    validation_scores = coverage_adjusted_scores(
        validation_bank, validation_scores
    )
    validation_scores = occurrence_adjusted_scores(
        validation_bank, validation_scores
    )
    train_base = [
        structure_correct(
            train_labels, scores, train_rows, train_allowed
        )
        for scores in score_sets
    ]
    validation_base = structure_correct(
        validation_labels,
        validation_scores,
        validation_rows,
        validation_allowed,
    )
    train_features = target_features(train_bank)
    validation_features = target_features(validation_bank)
    names = ("tail_recall", "explicit_target", "heading_target", "last_position")
    weights = np.round(np.arange(-0.5, 0.501, 0.025), 3)
    for feature_index, name in enumerate(names):
        train_feature = standardize_rows(train_features[:, :, feature_index])
        validation_feature = standardize_rows(
            validation_features[:, :, feature_index]
        )
        trials = []
        for weight in weights:
            count = structure_correct(
                train_labels,
                score_sets[0] + weight * train_feature,
                train_rows,
                train_allowed,
            )
            trials.append((count, -abs(float(weight)), float(weight)))
        _, _, weight = max(trials)
        train_deltas = [
            structure_correct(
                train_labels,
                scores + weight * train_feature,
                train_rows,
                train_allowed,
            )
            - base
            for scores, base in zip(score_sets, train_base)
        ]
        validation_delta = structure_correct(
            validation_labels,
            validation_scores + weight * validation_feature,
            validation_rows,
            validation_allowed,
        ) - validation_base
        print(
            f"feature={name} weight={weight:+.3f} "
            f"train_deltas={train_deltas} validation_delta={validation_delta:+d}",
            flush=True,
        )


if __name__ == "__main__":
    main()
