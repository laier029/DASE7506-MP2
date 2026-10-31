"""Probe margin-gated coverage switches from 3-vote groups to singletons."""

from collections import defaultdict
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
from cascade_verifier import unanimous_broad_error_groups
from common import load_inputs, parse_answer
from coverage_cascade_verifier import coverage_adjusted_scores
from group_ranker_probe import candidate_labels
from numeric_filter_stability_probe import apply_model_o_coverage
from numeric_plausibility import fractional_answer_groups
from occurrence_plausibility_verifier import occurrence_adjusted_scores
from question_number_coverage_probe import coverage_features
from structure_text_fusion_verifier import structure_text_scores


def answer_groups(row):
    groups = defaultdict(list)
    for index, candidate in enumerate(row["candidates"]):
        answer = parse_answer(candidate["text"])
        if answer is not None:
            groups[answer].append(index)
    return groups


def switch_cases(bank, labels, scores, coverage, aggregation):
    arithmetic = unanimous_broad_error_groups(bank)
    fractional = fractional_answer_groups(bank)
    structures = answer_structures(bank)
    gaps = []
    margins = []
    deltas = []
    rows = []
    for row_index, (row, row_scores) in enumerate(zip(bank, scores)):
        if structures[row_index] != "3-1":
            continue
        groups = answer_groups(row)
        rejected = arithmetic[row_index] | fractional[row_index]
        survivors = {
            answer: indices
            for answer, indices in groups.items()
            if answer not in rejected
        }
        if not survivors:
            survivors = groups
        if sorted(map(len, survivors.values())) != [1, 3]:
            continue
        scored = []
        for answer, indices in survivors.items():
            values = row_scores[indices]
            candidate = indices[int(values.argmax())]
            group_score = float(values.max() + 0.3 * values.mean())
            scored.append((group_score, answer, indices, candidate))
        scored.sort(reverse=True)
        selected = scored[0]
        if len(selected[2]) != 3:
            continue
        singleton = next(item for item in scored if len(item[2]) == 1)
        selected_coverage = getattr(np, aggregation)(
            coverage[row_index, selected[2]]
        )
        singleton_coverage = coverage[row_index, singleton[2][0]]
        gaps.append(float(singleton_coverage - selected_coverage))
        margins.append(float(selected[0] - singleton[0]))
        deltas.append(
            int(labels[row_index, singleton[3]])
            - int(labels[row_index, selected[3]])
        )
        rows.append(row_index)
    return (
        np.asarray(gaps),
        np.asarray(margins),
        np.asarray(deltas),
        np.asarray(rows),
    )


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    train_structures = answer_structures(train_bank)
    artifact = joblib.load(
        CODE / "runs/model_q_occurrence_plausibility/structure_text_model.joblib"
    )
    texts = [
        candidate["text"]
        for row in train_bank
        for candidate in row["candidates"]
    ]
    text_features = artifact["vectorizer"].transform(texts)

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
        # Occurrence scoring is inactive on 3-1, so it need not be recomputed.
        score_sets.append(scores)

    validation_scores = structure_text_scores(
        validation_bank,
        validation_arrays,
        CODE / "runs/model_q_occurrence_plausibility",
    )
    validation_scores = coverage_adjusted_scores(
        validation_bank, validation_scores
    )
    validation_scores = occurrence_adjusted_scores(
        validation_bank, validation_scores
    )
    train_coverage = calculation_unique_coverage(train_bank)
    validation_coverage = calculation_unique_coverage(validation_bank)

    gap_grid = np.arange(0.05, 0.501, 0.05)
    margin_grid = np.arange(0.1, 2.501, 0.1)
    for aggregation in ("max", "mean", "min"):
        train_cases = [
            switch_cases(
                train_bank, train_labels, scores, train_coverage, aggregation
            )
            for scores in score_sets
        ]
        fixed_gap, fixed_margin, fixed_delta, _ = train_cases[0]
        trials = []
        for gap_threshold in gap_grid:
            for margin_threshold in margin_grid:
                mask = (fixed_gap >= gap_threshold) & (
                    fixed_margin <= margin_threshold
                )
                delta = int(fixed_delta[mask].sum())
                trials.append(
                    (
                        delta,
                        -int(mask.sum()),
                        gap_threshold,
                        -margin_threshold,
                    )
                )
        best_delta, _, gap_threshold, negative_margin = max(trials)
        margin_threshold = -negative_margin
        train_deltas = []
        train_switches = []
        for gaps, margins, deltas, _ in train_cases:
            mask = (gaps >= gap_threshold) & (margins <= margin_threshold)
            train_deltas.append(int(deltas[mask].sum()))
            train_switches.append(int(mask.sum()))
        val_gap, val_margin, val_delta, val_rows = switch_cases(
            validation_bank,
            validation_labels,
            validation_scores,
            validation_coverage,
            aggregation,
        )
        val_mask = (val_gap >= gap_threshold) & (
            val_margin <= margin_threshold
        )
        print(
            f"aggregation={aggregation} gap={gap_threshold:.2f} "
            f"margin={margin_threshold:.2f} fixed_delta={best_delta:+d} "
            f"train_deltas={train_deltas} train_switches={train_switches} "
            f"validation_delta={int(val_delta[val_mask].sum()):+d} "
            f"validation_switches={int(val_mask.sum())} "
            f"validation_rows={val_rows[val_mask].tolist()}",
            flush=True,
        )


if __name__ == "__main__":
    main()
