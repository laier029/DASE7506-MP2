"""Measure whether directly copying a question quantity predicts wrong answers."""

from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from calculation_coverage import quantities
from common import load_inputs, parse_answer
from final_cascade_verifier import predict
from group_ranker_probe import candidate_labels
from structure_text_fusion_verifier import answer_structures


def answer_groups(row):
    groups = defaultdict(list)
    for index, candidate in enumerate(row["candidates"]):
        answer = parse_answer(candidate["text"])
        if answer is not None:
            groups[answer].append(index)
    return groups


def copy_groups(bank):
    output = []
    for row in bank:
        question_values = set()
        for token in quantities(row["question"], include_words=True):
            try:
                question_values.add(Fraction(token))
            except (ValueError, ZeroDivisionError):
                continue
        output.append(
            {
                answer
                for answer in answer_groups(row)
                if Fraction(answer) in question_values
            }
        )
    return output


def statistics(bank, labels, copied):
    structures = answer_structures(bank)
    counts = defaultdict(lambda: [0, 0])
    for row_index, row in enumerate(bank):
        groups = answer_groups(row)
        for answer in copied[row_index]:
            counts[structures[row_index]][0] += 1
            counts[structures[row_index]][1] += int(
                not labels[row_index, groups[answer][0]]
            )
    return counts


def main():
    train_bank, _ = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    train_copy = copy_groups(train_bank)
    validation_copy = copy_groups(validation_bank)
    for split, counts in (
        ("train", statistics(train_bank, train_labels, train_copy)),
        (
            "validation",
            statistics(validation_bank, validation_labels, validation_copy),
        ),
    ):
        for structure, (total, wrong) in sorted(counts.items()):
            print(
                f"{split} structure={structure} copied_groups={total} "
                f"wrong_precision={100 * wrong / max(total, 1):.2f}%"
            )

    choices = np.asarray(
        predict(
            validation_bank,
            validation_arrays,
            CODE / "runs/model_t_final_cascade",
        )
    )
    structures = answer_structures(validation_bank)
    for structure in sorted(set(structures)):
        rows = np.flatnonzero(structures == structure)
        selected_copy = [
            row_index
            for row_index in rows
            if parse_answer(
                validation_bank[row_index]["candidates"][choices[row_index]]["text"]
            )
            in validation_copy[row_index]
        ]
        wrong = sum(
            not validation_labels[row_index, choices[row_index]]
            for row_index in selected_copy
        )
        print(
            f"selected structure={structure} copied={len(selected_copy)} "
            f"wrong={wrong} rows={[row for row in selected_copy if not validation_labels[row, choices[row]]]}"
        )


if __name__ == "__main__":
    main()
