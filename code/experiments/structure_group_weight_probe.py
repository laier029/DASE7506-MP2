"""Probe structure-specific answer-group mean weights for Model T."""

from pathlib import Path
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from common import load_inputs
from coverage_cascade_verifier import coverage_adjusted_scores
from group_ranker_probe import candidate_labels
from model_t_probe_utils import FastFinalSelector, load_or_build_model_t_oof
from occurrence_plausibility_verifier import occurrence_adjusted_scores
from structure_text_fusion_verifier import answer_structures, structure_text_scores
from target_omitted_factor_verifier import target_adjusted_scores


def correct(labels, choices):
    return int(labels[np.arange(len(labels)), choices].sum())


def mixed_choices(selector, scores, rows, weight):
    base = selector.select(scores, 0.3)
    alternative = selector.select(scores, weight)
    base[rows] = alternative[rows]
    return base


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    score_sets = load_or_build_model_t_oof(train_bank, train_arrays, train_labels)
    train_selector = FastFinalSelector(train_bank)
    validation_selector = FastFinalSelector(validation_bank)
    train_structures = answer_structures(train_bank)
    validation_structures = answer_structures(validation_bank)
    validation_scores = structure_text_scores(
        validation_bank, validation_arrays, CODE / "runs/model_t_final_cascade"
    )
    validation_scores = coverage_adjusted_scores(validation_bank, validation_scores)
    validation_scores = occurrence_adjusted_scores(validation_bank, validation_scores)
    validation_scores = target_adjusted_scores(validation_bank, validation_scores)
    weights = np.round(np.arange(0.0, 0.601, 0.025), 3)
    for structure in ("3-1", "2-2", "2-1-1", "1-1-1-1"):
        train_rows = train_structures == structure
        validation_rows = validation_structures == structure
        trials = []
        for weight in weights:
            choices = mixed_choices(
                train_selector, score_sets[0], train_rows, weight
            )
            trials.append((correct(train_labels, choices), -abs(weight - 0.3), weight))
        _, _, weight = max(trials)
        train_counts = [
            correct(
                train_labels,
                mixed_choices(train_selector, scores, train_rows, weight),
            )
            for scores in score_sets
        ]
        validation_count = correct(
            validation_labels,
            mixed_choices(
                validation_selector,
                validation_scores,
                validation_rows,
                weight,
            ),
        )
        print(
            f"structure={structure} weight={weight:.3f} "
            f"train={train_counts} validation={validation_count}",
            flush=True,
        )


if __name__ == "__main__":
    main()
