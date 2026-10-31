"""Probe conflicts between protocol parsing and explicit Final Answer sections."""

from fractions import Fraction
from pathlib import Path
import re
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from common import load_inputs, parse_answer
from coverage_cascade_verifier import coverage_adjusted_scores
from group_ranker_probe import candidate_labels
from model_t_probe_utils import FastFinalSelector, load_or_build_model_t_oof
from occurrence_plausibility_verifier import occurrence_adjusted_scores
from numeric_plausibility import implausible_answer_groups
from omitted_factor import select_with_omitted_factor_switch
from stable_signal_features import stable_signal_adjusted_scores
from structure_text_fusion_verifier import structure_text_scores
from target_omitted_factor_verifier import target_adjusted_scores


FINAL = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:[*_]{0,2})?(?:✅\s*)?"
    r"(?:final\s+answer|answer)\s*:"
)
BOXED = re.compile(
    r"\\boxed\s*\{\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)\s*\}"
)
NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


def declared_final_answer(text):
    matches = list(FINAL.finditer(text))
    if not matches:
        return None
    tail = text[matches[-1].end() :]
    boxes = BOXED.findall(tail)
    if boxes:
        token = boxes[-1]
    else:
        match = NUMBER.search(tail)
        token = match.group() if match else None
    if token is None:
        return None
    try:
        return str(Fraction(token.replace(",", "")))
    except (ValueError, ZeroDivisionError):
        return None


def mismatch_targets(bank):
    output = []
    for row in bank:
        row_targets = []
        for candidate in row["candidates"]:
            target = declared_final_answer(candidate["text"])
            row_targets.append(
                target if target is not None and target != parse_answer(candidate["text"]) else None
            )
        output.append(row_targets)
    return output


def redirect_selected(bank, scores, choices, targets):
    output = choices.copy()
    changed = []
    for row_index, (row, choice) in enumerate(zip(bank, choices)):
        target = targets[row_index][choice]
        if target is None:
            continue
        indices = [
            index
            for index, candidate in enumerate(row["candidates"])
            if parse_answer(candidate["text"]) == target
        ]
        if not indices:
            continue
        output[row_index] = indices[int(scores[row_index, indices].argmax())]
        changed.append(row_index)
    return output, changed


def redirect_any_consistent(bank, scores, choices, targets):
    output = choices.copy()
    changed = []
    for row_index, row in enumerate(bank):
        row_targets = {target for target in targets[row_index] if target is not None}
        if len(row_targets) != 1:
            continue
        target = next(iter(row_targets))
        indices = [
            index
            for index, candidate in enumerate(row["candidates"])
            if parse_answer(candidate["text"]) == target
        ]
        if not indices:
            continue
        replacement = indices[int(scores[row_index, indices].argmax())]
        if parse_answer(row["candidates"][replacement]["text"]) != parse_answer(
            row["candidates"][choices[row_index]]["text"]
        ):
            changed.append(row_index)
        output[row_index] = replacement
    return output, changed


def correct(labels, choices):
    return int(labels[np.arange(len(labels)), choices].sum())


def summarize(name, bank, labels, targets):
    total = wrong = exists = target_correct = 0
    for row_index, row in enumerate(bank):
        for candidate_index, target in enumerate(targets[row_index]):
            if target is None:
                continue
            total += 1
            wrong += int(not labels[row_index, candidate_index])
            indices = [
                index
                for index, candidate in enumerate(row["candidates"])
                if parse_answer(candidate["text"]) == target
            ]
            if indices:
                exists += 1
                target_correct += int(labels[row_index, indices[0]])
    print(
        f"{name}_flags={total} wrong={wrong} wrong_precision={wrong / max(total, 1):.4f} "
        f"visible_targets={exists} correct_targets={target_correct} "
        f"target_precision={target_correct / max(exists, 1):.4f}"
    )


def rejected_mismatch_groups(bank, targets):
    output = []
    for row, row_targets in zip(bank, targets):
        groups = {}
        for candidate_index, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate["text"])
            if answer is not None:
                groups.setdefault(answer, []).append(candidate_index)
        output.append(
            {
                answer
                for answer, indices in groups.items()
                if all(row_targets[index] is not None for index in indices)
            }
        )
    return output


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    train_targets = mismatch_targets(train_bank)
    validation_targets = mismatch_targets(validation_bank)
    summarize("train", train_bank, train_labels, train_targets)
    summarize("validation", validation_bank, validation_labels, validation_targets)
    score_sets = load_or_build_model_t_oof(train_bank, train_arrays, train_labels)
    validation_scores = structure_text_scores(
        validation_bank, validation_arrays, CODE / "runs/model_t_final_cascade"
    )
    validation_scores = coverage_adjusted_scores(validation_bank, validation_scores)
    validation_scores = occurrence_adjusted_scores(validation_bank, validation_scores)
    validation_scores = target_adjusted_scores(validation_bank, validation_scores)
    selector_train = FastFinalSelector(train_bank)
    selector_validation = FastFinalSelector(validation_bank)
    train_mismatch_rejected = rejected_mismatch_groups(train_bank, train_targets)
    validation_mismatch_rejected = rejected_mismatch_groups(
        validation_bank, validation_targets
    )
    train_implausible = implausible_answer_groups(train_bank)
    validation_implausible = implausible_answer_groups(validation_bank)
    train_combined_rejected = [
        left | right
        for left, right in zip(train_implausible, train_mismatch_rejected)
    ]
    validation_combined_rejected = [
        left | right
        for left, right in zip(
            validation_implausible, validation_mismatch_rejected
        )
    ]
    for model_name, train_scores, val_scores in (
        ("t", score_sets, validation_scores),
        (
            "u",
            [stable_signal_adjusted_scores(train_bank, scores) for scores in score_sets],
            stable_signal_adjusted_scores(validation_bank, validation_scores),
        ),
    ):
        for rule_name, rule in (
            ("selected", redirect_selected),
            ("any_consistent", redirect_any_consistent),
        ):
            train_deltas = []
            train_changes = []
            for scores in train_scores:
                base = selector_train.select(scores)
                redirected, changed = rule(train_bank, scores, base, train_targets)
                train_deltas.append(correct(train_labels, redirected) - correct(train_labels, base))
                train_changes.append(len(changed))
            validation_base = selector_validation.select(val_scores)
            validation_redirected, validation_changed = rule(
                validation_bank, val_scores, validation_base, validation_targets
            )
            print(
                f"model={model_name} rule={rule_name} train_deltas={train_deltas} "
                f"train_changes={train_changes} "
                f"validation_delta={correct(validation_labels, validation_redirected) - correct(validation_labels, validation_base):+d} "
                f"validation_changes={validation_changed}"
            )
        filter_train_deltas = []
        for scores in train_scores:
            base = np.asarray(
                select_with_omitted_factor_switch(
                    train_bank, scores, 0.3, train_implausible
                )
            )
            filtered = np.asarray(
                select_with_omitted_factor_switch(
                    train_bank, scores, 0.3, train_combined_rejected
                )
            )
            filter_train_deltas.append(
                correct(train_labels, filtered) - correct(train_labels, base)
            )
        validation_base = np.asarray(
            select_with_omitted_factor_switch(
                validation_bank, val_scores, 0.3, validation_implausible
            )
        )
        validation_filtered = np.asarray(
            select_with_omitted_factor_switch(
                validation_bank, val_scores, 0.3, validation_combined_rejected
            )
        )
        filter_changes = np.flatnonzero(
            validation_base != validation_filtered
        ).tolist()
        print(
            f"model={model_name} rule=hard_filter train_deltas={filter_train_deltas} "
            f"validation_delta={correct(validation_labels, validation_filtered) - correct(validation_labels, validation_base):+d} "
            f"validation_changes={filter_changes}"
        )


if __name__ == "__main__":
    main()
