"""Audit validation choices changed by 2-1-1 calculation coverage scoring."""

from pathlib import Path
import argparse
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from cascade_verifier import select_after_negative_filter
from common import load_inputs, parse_answer
from group_ranker_probe import candidate_labels
from question_number_coverage_probe import coverage_features
from structure_text_fusion_verifier import answer_structures, structure_text_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=float, default=0.25)
    args = parser.parse_args()
    bank, arrays = load_inputs("validation")
    labels = candidate_labels("validation", bank).astype(bool)
    structures = answer_structures(bank)
    scores = structure_text_scores(bank, arrays, CODE / "runs/model_n_cascade")
    raw_coverage = coverage_features(bank)[:, :, 3]
    coverage = (raw_coverage - raw_coverage.mean(axis=1, keepdims=True)) / (
        raw_coverage.std(axis=1, keepdims=True) + 1e-6
    )
    adjusted = scores.copy()
    rows = structures == "2-1-1"
    adjusted[rows] += args.weight * coverage[rows]
    base = np.asarray(select_after_negative_filter(bank, scores, 0.3))
    changed = np.asarray(select_after_negative_filter(bank, adjusted, 0.3))

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
                f"coverage={raw_coverage[row_index, candidate_index]:.3f} "
                f"z={coverage[row_index, candidate_index]:+.3f} "
                f"score={scores[row_index, candidate_index]:+.3f}"
            )


if __name__ == "__main__":
    main()
