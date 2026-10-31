"""Reusable Model T OOF scores and a fast static final selector for probes."""

from fractions import Fraction
from pathlib import Path
import sys

import joblib
import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from calculation_coverage import (
    calculation_context,
    calculation_occurrence_coverage,
    calculation_quantities,
    quantities,
)
from cascade_stability_probe import balanced_random_folds, model_m_oof
from cascade_verifier import unanimous_broad_error_groups
from numeric_filter_stability_probe import apply_model_o_coverage
from numeric_plausibility import implausible_answer_groups
from omitted_factor import _answer_groups, _fraction_set
from structure_text_fusion_verifier import answer_structures
from target_alignment import terminal_target_recall


CACHE = CODE / "runs/diagnostics/model_t_oof_scores_currency_fix.npz"
SEEDS = (7506, 11, 29, 47, 83)


def standardize_rows(values):
    return (values - values.mean(axis=1, keepdims=True)) / (
        values.std(axis=1, keepdims=True) + 1e-6
    )


def load_or_build_model_t_oof(bank, arrays, labels):
    """Return five Model T pre-selection OOF score tensors."""
    if CACHE.exists():
        with np.load(CACHE, allow_pickle=False) as saved:
            scores = saved["scores"]
        if scores.shape == (len(SEEDS), len(bank), 4):
            return [score.copy() for score in scores]

    artifact = joblib.load(
        CODE / "runs/model_t_final_cascade/structure_text_model.joblib"
    )
    texts = [c["text"] for row in bank for c in row["candidates"]]
    text_features = artifact["vectorizer"].transform(texts)
    structures = answer_structures(bank)
    occurrence = standardize_rows(calculation_occurrence_coverage(bank))
    target = standardize_rows(terminal_target_recall(bank))
    all_scores = []
    for seed in SEEDS:
        folds = (
            np.arange(len(bank)) % 5
            if seed == 7506
            else balanced_random_folds(len(bank), seed)
        )
        scores = model_m_oof(
            bank, arrays, labels, folds, text_features, structures
        )
        scores = apply_model_o_coverage(bank, scores, structures)
        rows = structures == "1-1-1-1"
        scores[rows] += 0.225 * occurrence[rows]
        scores[rows] += 0.125 * target[rows]
        all_scores.append(scores.astype(np.float32))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, scores=np.stack(all_scores))
    return all_scores


class FastFinalSelector:
    """Precompute text-only state used by Model T's final group selector."""

    def __init__(self, bank):
        self.bank = bank
        arithmetic = unanimous_broad_error_groups(bank)
        implausible = implausible_answer_groups(bank)
        self.rows = []
        for row_index, row in enumerate(bank):
            groups = _answer_groups(row)
            rejected = arithmetic[row_index] | implausible[row_index]
            survivors = {
                answer: indices
                for answer, indices in groups.items()
                if answer not in rejected
            }
            if not survivors:
                survivors = groups
            question_values = _fraction_set(
                quantities(row["question"], include_words=True)
            )
            candidate_values = [
                _fraction_set(
                    calculation_quantities(
                        calculation_context(candidate["text"]),
                        include_words=True,
                    )
                )
                for candidate in row["candidates"]
            ]
            self.rows.append((survivors, question_values, candidate_values))

    def select(self, scores, group_mean_weight=0.3):
        choices = []
        for row_scores, state in zip(scores, self.rows):
            survivors, question_values, candidate_values = state
            scored = []
            for answer, indices in survivors.items():
                values = row_scores[indices]
                candidate = indices[int(values.argmax())]
                scored.append(
                    (
                        float(
                            values.max() + group_mean_weight * values.mean()
                        ),
                        answer,
                        indices,
                        candidate,
                    )
                )
            scored.sort(reverse=True)
            selected = scored[0]
            if len(selected[2]) >= 2:
                selected_value = Fraction(selected[1])
                if selected_value != 0:
                    selected_used = set().union(
                        *(candidate_values[index] for index in selected[2])
                    )
                    alternatives = []
                    for item in scored[1:]:
                        alternative_value = Fraction(item[1])
                        if alternative_value == 0:
                            continue
                        ratio = abs(alternative_value / selected_value)
                        if ratio < 1:
                            ratio = 1 / ratio
                        if ratio <= 1 or ratio not in question_values:
                            continue
                        if ratio in selected_used:
                            continue
                        alternative_used = set().union(
                            *(candidate_values[index] for index in item[2])
                        )
                        if ratio not in alternative_used:
                            continue
                        if selected[0] - item[0] > 1.25:
                            continue
                        alternatives.append(item)
                    if alternatives:
                        selected = max(alternatives)
            choices.append(selected[3])
        return np.asarray(choices)

    def correct_count(self, labels, scores, group_mean_weight=0.3):
        choices = self.select(scores, group_mean_weight)
        return int(labels[np.arange(len(labels)), choices].sum())
