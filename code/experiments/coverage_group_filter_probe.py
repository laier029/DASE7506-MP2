"""Probe high-precision group rejection from unanimously missing quantities."""

from collections import defaultdict
from pathlib import Path
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from calculation_coverage import calculation_unique_coverage
from cascade_verifier import unanimous_broad_error_groups
from common import load_inputs, parse_answer
from coverage_cascade_verifier import coverage_adjusted_scores
from group_ranker_probe import candidate_labels
from structure_text_fusion_verifier import answer_structures, structure_text_scores


def answer_groups(row):
    groups = defaultdict(list)
    for candidate_index, candidate in enumerate(row["candidates"]):
        answer = parse_answer(candidate["text"])
        if answer is not None:
            groups[answer].append(candidate_index)
    return groups


def low_coverage_groups(bank, coverage, structures, threshold, active):
    rejected = []
    for row_index, row in enumerate(bank):
        row_rejected = set()
        groups = answer_groups(row)
        if structures[row_index] not in active:
            rejected.append(row_rejected)
            continue
        for answer, indices in groups.items():
            if len(indices) < 2:
                continue
            alternatives = [
                index
                for other_answer, other_indices in groups.items()
                if other_answer != answer
                for index in other_indices
            ]
            if not alternatives:
                continue
            if coverage[row_index, indices].max() + threshold <= coverage[
                row_index, alternatives
            ].max():
                row_rejected.add(answer)
        rejected.append(row_rejected)
    return rejected


def group_statistics(bank, labels, rejected):
    total = wrong = correct_kills = 0
    for row_index, (row, row_rejected) in enumerate(zip(bank, rejected)):
        groups = answer_groups(row)
        for answer in row_rejected:
            total += 1
            wrong += int(not labels[row_index, groups[answer][0]])
        correct_answers = {
            answer
            for answer, indices in groups.items()
            if labels[row_index, indices[0]]
        }
        correct_kills += int(bool(correct_answers & row_rejected))
    return total, wrong, correct_kills


def select(bank, scores, extra_rejected):
    arithmetic_rejected = unanimous_broad_error_groups(bank)
    selected = []
    for row, row_scores, arithmetic, coverage in zip(
        bank, scores, arithmetic_rejected, extra_rejected
    ):
        groups = answer_groups(row)
        rejected = arithmetic | coverage
        survivors = {
            answer: indices
            for answer, indices in groups.items()
            if answer not in rejected
        }
        if not survivors:
            survivors = groups
        best = None
        for indices in survivors.values():
            values = row_scores[indices]
            candidate = indices[int(values.argmax())]
            item = (float(values.max() + 0.3 * values.mean()), -candidate, candidate)
            if best is None or item > best:
                best = item
        selected.append(best[2])
    return np.asarray(selected)


def accuracy(labels, selected):
    return int(labels[np.arange(len(labels)), selected].sum())


def main():
    train_bank, _ = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    train_structures = answer_structures(train_bank)
    validation_structures = answer_structures(validation_bank)
    train_coverage = calculation_unique_coverage(train_bank)
    validation_coverage = calculation_unique_coverage(validation_bank)

    validation_scores = structure_text_scores(
        validation_bank,
        validation_arrays,
        CODE / "runs/model_o_coverage_cascade",
    )
    validation_scores = coverage_adjusted_scores(
        validation_bank, validation_scores
    )
    base_rejected = [set() for _ in validation_bank]
    base = select(validation_bank, validation_scores, base_rejected)
    base_correct = accuracy(validation_labels, base)

    for active in (("3-1",), ("3-1", "2-2"), ("3-1", "2-2", "2-1-1")):
        for threshold in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
            train_rejected = low_coverage_groups(
                train_bank,
                train_coverage,
                train_structures,
                threshold,
                set(active),
            )
            validation_rejected = low_coverage_groups(
                validation_bank,
                validation_coverage,
                validation_structures,
                threshold,
                set(active),
            )
            train_total, train_wrong, train_kills = group_statistics(
                train_bank, train_labels, train_rejected
            )
            val_total, val_wrong, val_kills = group_statistics(
                validation_bank, validation_labels, validation_rejected
            )
            chosen = select(
                validation_bank, validation_scores, validation_rejected
            )
            delta = accuracy(validation_labels, chosen) - base_correct
            print(
                f"active={'+'.join(active)} threshold={threshold:.2f} "
                f"train_groups={train_total} "
                f"train_precision={100 * train_wrong / max(train_total, 1):.2f}% "
                f"train_correct_kills={train_kills} "
                f"validation_groups={val_total} "
                f"validation_precision={100 * val_wrong / max(val_total, 1):.2f}% "
                f"validation_correct_kills={val_kills} delta={delta:+d}",
                flush=True,
            )


if __name__ == "__main__":
    main()
