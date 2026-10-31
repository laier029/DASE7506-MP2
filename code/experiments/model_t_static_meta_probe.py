"""Probe nonlinear static survivor scores fused with Model T OOF scores."""

from collections import Counter
from fractions import Fraction
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from calculation_coverage import (
    calculation_occurrence_coverage,
    calculation_unique_coverage,
)
from cascade_stability_probe import balanced_random_folds
from common import load_inputs, parse_answer
from coverage_cascade_verifier import coverage_adjusted_scores
from group_ranker_probe import candidate_labels, candidate_static_features
from implicit_factor_probe import implicit_features
from model_t_probe_utils import (
    FastFinalSelector,
    SEEDS,
    load_or_build_model_t_oof,
    standardize_rows,
)
from occurrence_plausibility_verifier import occurrence_adjusted_scores
from structure_text_fusion_verifier import answer_structures, structure_text_scores
from target_alignment import terminal_target_recall
from target_omitted_factor_verifier import target_adjusted_scores
from target_operation_probe import operation_features


def answer_group_features(bank):
    output = np.zeros((len(bank), 4, 10), dtype=np.float32)
    for row_index, row in enumerate(bank):
        answers = [parse_answer(candidate["text"]) for candidate in row["candidates"]]
        counts = Counter(answer for answer in answers if answer is not None)
        numeric = []
        for answer in answers:
            try:
                numeric.append(float(Fraction(answer)))
            except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                numeric.append(np.nan)
        finite = np.asarray([value for value in numeric if np.isfinite(value)])
        for candidate_index, (answer, value) in enumerate(zip(answers, numeric)):
            if np.isfinite(value):
                magnitude = np.sign(value) * np.log1p(abs(value))
                rank = float((finite < value).sum()) / max(len(finite) - 1, 1)
                integral = float(Fraction(answer).denominator == 1)
                positive = float(value > 0)
                zero = float(value == 0)
            else:
                magnitude = rank = integral = positive = zero = 0.0
            output[row_index, candidate_index] = (
                counts.get(answer, 0) / 4 if answer is not None else 0.0,
                magnitude,
                rank,
                integral,
                positive,
                zero,
                float(candidate_index == 0),
                float(candidate_index == 1),
                float(candidate_index == 2),
                float(candidate_index == 3),
            )
    return output


def static_features(bank):
    structures = answer_structures(bank)
    structure_names = ("3-1", "2-2", "2-1-1", "1-1-1-1")
    structure_one_hot = np.zeros((len(bank), 4, len(structure_names)), dtype=np.float32)
    for index, name in enumerate(structure_names):
        structure_one_hot[structures == name, :, index] = 1.0
    return np.concatenate(
        [
            candidate_static_features(bank),
            calculation_unique_coverage(bank)[:, :, None],
            calculation_occurrence_coverage(bank)[:, :, None],
            terminal_target_recall(bank)[:, :, None],
            operation_features(bank),
            implicit_features(bank),
            answer_group_features(bank),
            structure_one_hot,
        ],
        axis=2,
    )


def make_classifier(leaves, iterations):
    return HistGradientBoostingClassifier(
        max_iter=iterations,
        learning_rate=0.05,
        max_leaf_nodes=leaves,
        min_samples_leaf=100,
        l2_regularization=5.0,
        random_state=7506,
    )


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    score_sets = load_or_build_model_t_oof(
        train_bank, train_arrays, train_labels
    )
    train_selector = FastFinalSelector(train_bank)
    validation_selector = FastFinalSelector(validation_bank)
    validation_scores = structure_text_scores(
        validation_bank,
        validation_arrays,
        CODE / "runs/model_t_final_cascade",
    )
    validation_scores = coverage_adjusted_scores(validation_bank, validation_scores)
    validation_scores = occurrence_adjusted_scores(validation_bank, validation_scores)
    validation_scores = target_adjusted_scores(validation_bank, validation_scores)
    train_x = static_features(train_bank)
    validation_x = static_features(validation_bank)
    structures = answer_structures(train_bank)
    active_rows = structures != "4"
    flat_active = np.repeat(active_rows, 4)
    target = train_labels.reshape(-1)
    weights = np.round(np.arange(0.0, 0.501, 0.025), 3)

    for leaves, iterations in ((3, 60), (7, 60), (7, 120)):
        meta_sets = []
        for seed in SEEDS:
            folds = (
                np.arange(len(train_bank)) % 5
                if seed == 7506
                else balanced_random_folds(len(train_bank), seed)
            )
            meta = np.zeros((len(train_bank), 4), dtype=np.float32)
            for fold in range(5):
                fit_rows = (folds != fold) & active_rows
                held_rows = (folds == fold) & active_rows
                fit = np.repeat(fit_rows, 4)
                held = np.repeat(held_rows, 4)
                classifier = make_classifier(leaves, iterations).fit(
                    train_x.reshape(-1, train_x.shape[-1])[fit], target[fit]
                )
                meta.reshape(-1)[held] = classifier.predict_proba(
                    train_x.reshape(-1, train_x.shape[-1])[held]
                )[:, 1]
            meta_sets.append(standardize_rows(meta))

        base_counts = [
            train_selector.correct_count(train_labels, scores)
            for scores in score_sets
        ]
        trials = []
        for weight in weights:
            count = train_selector.correct_count(
                train_labels, score_sets[0] + weight * meta_sets[0]
            )
            trials.append((count, -float(weight), float(weight)))
        _, _, weight = max(trials)
        train_deltas = [
            train_selector.correct_count(
                train_labels, scores + weight * meta
            )
            - base
            for scores, meta, base in zip(score_sets, meta_sets, base_counts)
        ]

        full = make_classifier(leaves, iterations).fit(
            train_x.reshape(-1, train_x.shape[-1])[flat_active],
            target[flat_active],
        )
        validation_meta = np.zeros((len(validation_bank), 4), dtype=np.float32)
        validation_active = answer_structures(validation_bank) != "4"
        held = np.repeat(validation_active, 4)
        validation_meta.reshape(-1)[held] = full.predict_proba(
            validation_x.reshape(-1, validation_x.shape[-1])[held]
        )[:, 1]
        validation_meta = standardize_rows(validation_meta)
        validation_base = validation_selector.correct_count(
            validation_labels, validation_scores
        )
        validation_delta = validation_selector.correct_count(
            validation_labels,
            validation_scores + weight * validation_meta,
        ) - validation_base
        print(
            f"leaves={leaves} iterations={iterations} weight={weight:.3f} "
            f"train_deltas={train_deltas} validation_delta={validation_delta:+d}",
            flush=True,
        )

        structure_weights = {}
        fixed_adjustment = np.zeros_like(meta_sets[0])
        validation_adjustment = np.zeros_like(validation_meta)
        train_structures = answer_structures(train_bank)
        validation_structures = answer_structures(validation_bank)
        for structure in ("3-1", "2-2", "2-1-1", "1-1-1-1"):
            rows = train_structures == structure
            validation_rows = validation_structures == structure
            trials = []
            for candidate_weight in weights:
                adjustment = np.zeros_like(meta_sets[0])
                adjustment[rows] = candidate_weight * meta_sets[0][rows]
                count = train_selector.correct_count(
                    train_labels, score_sets[0] + adjustment
                )
                trials.append(
                    (count, -float(candidate_weight), float(candidate_weight))
                )
            _, _, structure_weight = max(trials)
            structure_weights[structure] = structure_weight
            fixed_adjustment[rows] = structure_weight * meta_sets[0][rows]
            validation_adjustment[validation_rows] = (
                structure_weight * validation_meta[validation_rows]
            )
        structured_deltas = []
        for scores, meta, base in zip(score_sets, meta_sets, base_counts):
            adjustment = np.zeros_like(meta)
            for structure, structure_weight in structure_weights.items():
                rows = train_structures == structure
                adjustment[rows] = structure_weight * meta[rows]
            structured_deltas.append(
                train_selector.correct_count(
                    train_labels, scores + adjustment
                )
                - base
            )
        structured_validation_delta = validation_selector.correct_count(
            validation_labels,
            validation_scores + validation_adjustment,
        ) - validation_base
        print(
            f"  structure_weights={structure_weights} "
            f"train_deltas={structured_deltas} "
            f"validation_delta={structured_validation_delta:+d}",
            flush=True,
        )


if __name__ == "__main__":
    main()
