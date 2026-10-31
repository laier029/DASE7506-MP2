"""Audit validation changes from terminal target-word recall."""

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
from numeric_plausibility import fractional_answer_groups
from occurrence_plausibility_verifier import occurrence_adjusted_scores
from omitted_factor import select_with_omitted_factor_switch
from structure_text_fusion_verifier import answer_structures, structure_text_scores
from target_alignment_probe import standardize_rows, target_features


def main():
    bank, arrays = load_inputs("validation")
    labels = candidate_labels("validation", bank).astype(bool)
    scores = structure_text_scores(
        bank, arrays, CODE / "runs/model_r_omitted_factor"
    )
    scores = coverage_adjusted_scores(bank, scores)
    scores = occurrence_adjusted_scores(bank, scores)
    feature = standardize_rows(target_features(bank)[:, :, 0])
    adjusted = scores.copy()
    rows = answer_structures(bank) == "1-1-1-1"
    adjusted[rows] += 0.125 * feature[rows]
    rejected = fractional_answer_groups(bank)
    base = np.asarray(
        select_with_omitted_factor_switch(bank, scores, 0.3, rejected)
    )
    changed = np.asarray(
        select_with_omitted_factor_switch(bank, adjusted, 0.3, rejected)
    )
    for row_index in np.flatnonzero(base != changed):
        row = bank[row_index]
        print(
            f"question={row_index} base={base[row_index]} "
            f"base_correct={bool(labels[row_index, base[row_index]])} "
            f"changed={changed[row_index]} "
            f"changed_correct={bool(labels[row_index, changed[row_index]])} "
            f"prompt={' '.join(row['question'].split())}"
        )
        for candidate_index, candidate in enumerate(row["candidates"]):
            print(
                f"  c={candidate_index} answer={parse_answer(candidate['text'])} "
                f"correct={bool(labels[row_index, candidate_index])} "
                f"feature={feature[row_index, candidate_index]:+.3f} "
                f"score={scores[row_index, candidate_index]:+.3f}"
            )


if __name__ == "__main__":
    main()
