"""Probe uncertainty-gated majority fallback for 2-1-1 answer structures."""

from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from cascade_verifier import unanimous_broad_error_groups
from common import load_inputs, parse_answer
from coverage_cascade_verifier import coverage_adjusted_scores
from declared_answer_consistency import redirect_consistent_boxed_endorsement
from final_declaration_consistency import final_declaration_mismatch_groups
from group_ranker_probe import candidate_labels
from model_t_probe_utils import load_or_build_model_t_oof
from numeric_plausibility import implausible_answer_groups
from occurrence_plausibility_verifier import occurrence_adjusted_scores
from omitted_factor import select_with_omitted_factor_switch
from stable_signal_features import stable_signal_adjusted_scores
from structure_text_fusion_verifier import structure_text_scores
from target_omitted_factor_verifier import target_adjusted_scores


def rejection_sets(bank):
    arithmetic = unanimous_broad_error_groups(bank)
    numeric = implausible_answer_groups(bank)
    declaration = final_declaration_mismatch_groups(bank)
    return [
        first | second | third
        for first, second, third in zip(arithmetic, numeric, declaration)
    ], [first | second for first, second in zip(numeric, declaration)]


def base_choices(bank, scores, additional):
    choices = select_with_omitted_factor_switch(bank, scores, 0.3, additional)
    return np.asarray(
        redirect_consistent_boxed_endorsement(bank, scores, choices)
    )


def fallback(bank, scores, choices, rejected, threshold):
    output = choices.copy()
    changes = []
    for row_index, (row, row_scores, choice) in enumerate(zip(bank, scores, choices)):
        groups = defaultdict(list)
        for candidate_index, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate["text"])
            if answer is not None:
                groups[answer].append(candidate_index)
        structure = sorted((len(indices) for indices in groups.values()), reverse=True)
        if structure != [2, 1, 1]:
            continue
        selected_answer = parse_answer(row["candidates"][choice]["text"])
        if selected_answer not in groups or len(groups[selected_answer]) != 1:
            continue
        majority_answer = next(
            answer for answer, indices in groups.items() if len(indices) == 2
        )
        if majority_answer in rejected[row_index]:
            continue
        selected_indices = groups[selected_answer]
        majority_indices = groups[majority_answer]
        selected_values = row_scores[selected_indices]
        majority_values = row_scores[majority_indices]
        selected_score = float(selected_values.max() + 0.3 * selected_values.mean())
        majority_score = float(majority_values.max() + 0.3 * majority_values.mean())
        if selected_score - majority_score > threshold:
            continue
        output[row_index] = majority_indices[int(majority_values.argmax())]
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
    thresholds = np.round(np.arange(-1.0, 2.001, 0.1), 3)
    trials = []
    for threshold in thresholds:
        choices, changes = fallback(
            train_bank,
            score_sets[0],
            train_base[0],
            train_rejected,
            threshold,
        )
        trials.append(
            (correct(train_labels, choices), -len(changes), -abs(threshold), threshold)
        )
    _, _, _, threshold = max(trials)
    train_results = []
    for scores, base in zip(score_sets, train_base):
        choices, changes = fallback(
            train_bank, scores, base, train_rejected, threshold
        )
        train_results.append(
            (correct(train_labels, choices) - correct(train_labels, base), len(changes))
        )
    validation_choices, validation_changes = fallback(
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
