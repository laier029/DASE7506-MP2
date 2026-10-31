"""Probe low-margin majority-to-singleton switches for 3-1 structures."""

from collections import defaultdict
from pathlib import Path
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from common import load_inputs, parse_answer
from coverage_cascade_verifier import coverage_adjusted_scores
from group_ranker_probe import candidate_labels
from minority_fallback_probe import base_choices, rejection_sets
from model_t_probe_utils import load_or_build_model_t_oof
from occurrence_plausibility_verifier import occurrence_adjusted_scores
from stable_signal_features import stable_signal_adjusted_scores
from structure_text_fusion_verifier import structure_text_scores
from target_omitted_factor_verifier import target_adjusted_scores


def switch(bank, scores, choices, rejected, threshold):
    output = choices.copy()
    changes = []
    for row_index, (row, row_scores, choice) in enumerate(zip(bank, scores, choices)):
        groups = defaultdict(list)
        for candidate_index, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate["text"])
            if answer is not None:
                groups[answer].append(candidate_index)
        if sorted(map(len, groups.values()), reverse=True) != [3, 1]:
            continue
        selected_answer = parse_answer(row["candidates"][choice]["text"])
        if selected_answer not in groups or len(groups[selected_answer]) != 3:
            continue
        singleton_answer, singleton_indices = next(
            (answer, indices)
            for answer, indices in groups.items()
            if len(indices) == 1
        )
        if singleton_answer in rejected[row_index]:
            continue
        majority_indices = groups[selected_answer]
        majority_values = row_scores[majority_indices]
        singleton_values = row_scores[singleton_indices]
        majority_score = float(majority_values.max() + 0.3 * majority_values.mean())
        singleton_score = float(singleton_values.max() + 0.3 * singleton_values.mean())
        if majority_score - singleton_score > threshold:
            continue
        output[row_index] = singleton_indices[int(singleton_values.argmax())]
        changes.append(row_index)
    return output, changes


def correct(labels, choices):
    return int(labels[np.arange(len(labels)), choices].sum())


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    score_sets = [
        stable_signal_adjusted_scores(train_bank, scores)
        for scores in load_or_build_model_t_oof(train_bank, train_arrays, train_labels)
    ]
    validation_scores = structure_text_scores(
        validation_bank, validation_arrays, CODE / "runs/model_t_final_cascade"
    )
    validation_scores = coverage_adjusted_scores(validation_bank, validation_scores)
    validation_scores = occurrence_adjusted_scores(validation_bank, validation_scores)
    validation_scores = target_adjusted_scores(validation_bank, validation_scores)
    validation_scores = stable_signal_adjusted_scores(validation_bank, validation_scores)
    train_rejected, train_additional = rejection_sets(train_bank)
    validation_rejected, validation_additional = rejection_sets(validation_bank)
    train_base = [base_choices(train_bank, scores, train_additional) for scores in score_sets]
    validation_base = base_choices(validation_bank, validation_scores, validation_additional)
    thresholds = np.round(np.arange(0.0, 3.001, 0.1), 3)
    trials = []
    for threshold in thresholds:
        choices, changes = switch(
            train_bank, score_sets[0], train_base[0], train_rejected, threshold
        )
        trials.append((correct(train_labels, choices), -len(changes), -threshold, threshold))
    _, _, _, threshold = max(trials)
    train_results = []
    for scores, base in zip(score_sets, train_base):
        choices, changes = switch(
            train_bank, scores, base, train_rejected, threshold
        )
        train_results.append(
            (correct(train_labels, choices) - correct(train_labels, base), len(changes))
        )
    validation_choices, validation_changes = switch(
        validation_bank,
        validation_scores,
        validation_base,
        validation_rejected,
        threshold,
    )
    print(
        f"threshold={threshold:.2f} train={train_results} "
        f"validation_delta={correct(validation_labels, validation_choices) - correct(validation_labels, validation_base):+d} "
        f"validation_rows={validation_changes}"
    )


if __name__ == "__main__":
    main()
