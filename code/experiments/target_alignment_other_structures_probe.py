"""Test terminal target recall on Model T's grouped-answer structures."""

from pathlib import Path
import sys

import joblib
import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from calculation_coverage import calculation_occurrence_coverage
from cascade_stability_probe import (
    answer_structures,
    balanced_random_folds,
    model_m_oof,
)
from common import load_inputs
from coverage_cascade_verifier import coverage_adjusted_scores
from group_ranker_probe import candidate_labels
from numeric_filter_stability_probe import apply_model_o_coverage
from numeric_plausibility import implausible_answer_groups
from occurrence_plausibility_verifier import occurrence_adjusted_scores
from omitted_factor import select_with_omitted_factor_switch
from structure_text_fusion_verifier import structure_text_scores
from target_alignment import terminal_target_recall
from target_omitted_factor_verifier import target_adjusted_scores


def standardize_rows(values):
    return (values - values.mean(axis=1, keepdims=True)) / (
        values.std(axis=1, keepdims=True) + 1e-6
    )


def subset(bank, values, rows):
    return [value for value, keep in zip(values, rows) if keep]


def correct_count(bank, labels, scores, rejected):
    choices = np.asarray(
        select_with_omitted_factor_switch(bank, scores, 0.3, rejected)
    )
    return int(labels[np.arange(len(labels)), choices].sum())


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    train_structures = answer_structures(train_bank)
    validation_structures = answer_structures(validation_bank)
    artifact = joblib.load(
        CODE / "runs/model_t_final_cascade/structure_text_model.joblib"
    )
    texts = [
        candidate["text"]
        for row in train_bank
        for candidate in row["candidates"]
    ]
    text_features = artifact["vectorizer"].transform(texts)
    occurrence = standardize_rows(calculation_occurrence_coverage(train_bank))
    target = standardize_rows(terminal_target_recall(train_bank))
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
        scores = apply_model_o_coverage(train_bank, scores, train_structures)
        all_distinct = train_structures == "1-1-1-1"
        scores[all_distinct] += 0.225 * occurrence[all_distinct]
        scores[all_distinct] += 0.125 * target[all_distinct]
        score_sets.append(scores)

    validation_scores = structure_text_scores(
        validation_bank,
        validation_arrays,
        CODE / "runs/model_t_final_cascade",
    )
    validation_scores = coverage_adjusted_scores(
        validation_bank, validation_scores
    )
    validation_scores = occurrence_adjusted_scores(
        validation_bank, validation_scores
    )
    validation_scores = target_adjusted_scores(
        validation_bank, validation_scores
    )
    train_feature = target
    validation_feature = standardize_rows(terminal_target_recall(validation_bank))
    train_rejected_all = implausible_answer_groups(train_bank)
    validation_rejected_all = implausible_answer_groups(validation_bank)
    weights = np.round(np.arange(-0.4, 0.401, 0.025), 3)

    for structure in ("3-1", "2-2", "2-1-1"):
        train_rows = train_structures == structure
        validation_rows = validation_structures == structure
        train_subbank = subset(train_bank, train_bank, train_rows)
        validation_subbank = subset(
            validation_bank, validation_bank, validation_rows
        )
        train_sublabels = train_labels[train_rows]
        validation_sublabels = validation_labels[validation_rows]
        train_rejected = subset(
            train_bank, train_rejected_all, train_rows
        )
        validation_rejected = subset(
            validation_bank, validation_rejected_all, validation_rows
        )
        base_counts = [
            correct_count(
                train_subbank,
                train_sublabels,
                scores[train_rows],
                train_rejected,
            )
            for scores in score_sets
        ]
        validation_base = correct_count(
            validation_subbank,
            validation_sublabels,
            validation_scores[validation_rows],
            validation_rejected,
        )
        trials = []
        for weight in weights:
            count = correct_count(
                train_subbank,
                train_sublabels,
                score_sets[0][train_rows]
                + weight * train_feature[train_rows],
                train_rejected,
            )
            trials.append((count, -abs(float(weight)), float(weight)))
        _, _, weight = max(trials)
        train_deltas = [
            correct_count(
                train_subbank,
                train_sublabels,
                scores[train_rows] + weight * train_feature[train_rows],
                train_rejected,
            )
            - base
            for scores, base in zip(score_sets, base_counts)
        ]
        validation_delta = correct_count(
            validation_subbank,
            validation_sublabels,
            validation_scores[validation_rows]
            + weight * validation_feature[validation_rows],
            validation_rejected,
        ) - validation_base
        print(
            f"structure={structure} weight={weight:+.3f} "
            f"train_deltas={train_deltas} validation_delta={validation_delta:+d}",
            flush=True,
        )


if __name__ == "__main__":
    main()
