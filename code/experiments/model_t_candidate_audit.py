"""Print Model T error candidates and the verifier signals used so far."""

import argparse
from pathlib import Path
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from arithmetic_consistency import arithmetic_counts, plain_text_arithmetic_counts
from calculation_coverage import (
    calculation_occurrence_coverage,
    calculation_unique_coverage,
)
from common import load_inputs, parse_answer
from coverage_cascade_verifier import coverage_adjusted_scores
from group_ranker_probe import candidate_labels
from numeric_plausibility import implausible_answer_groups
from occurrence_plausibility_verifier import occurrence_adjusted_scores
from omitted_factor import select_with_omitted_factor_switch
from structure_text_fusion_verifier import structure_text_scores
from target_alignment import terminal_target_recall
from target_omitted_factor_verifier import target_adjusted_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("indices", nargs="*", type=int)
    args = parser.parse_args()
    bank, arrays = load_inputs("validation")
    labels = candidate_labels("validation", bank).astype(bool)
    scores = structure_text_scores(
        bank, arrays, CODE / "runs/model_t_final_cascade"
    )
    scores = coverage_adjusted_scores(bank, scores)
    scores = occurrence_adjusted_scores(bank, scores)
    scores = target_adjusted_scores(bank, scores)
    choices = np.asarray(
        select_with_omitted_factor_switch(
            bank, scores, 0.3, implausible_answer_groups(bank)
        )
    )
    unique = calculation_unique_coverage(bank)
    occurrence = calculation_occurrence_coverage(bank)
    targets = terminal_target_recall(bank)
    wanted = set(args.indices)
    for row_index, row in enumerate(bank):
        if labels[row_index, choices[row_index]]:
            continue
        if wanted and row_index not in wanted:
            continue
        print("=" * 88)
        print(f"q={row_index} selected={choices[row_index]} question={row['question']}")
        for candidate_index, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate["text"])
            status = "GOLD" if labels[row_index, candidate_index] else "wrong"
            selected = " SELECTED" if choices[row_index] == candidate_index else ""
            narrow = arithmetic_counts(candidate["text"])
            plain = plain_text_arithmetic_counts(candidate["text"])
            print(
                f"\n[{candidate_index}] {status}{selected} answer={answer} "
                f"score={scores[row_index, candidate_index]:.4f} "
                f"unique={unique[row_index, candidate_index]:.3f} "
                f"occ={occurrence[row_index, candidate_index]:.3f} "
                f"target={targets[row_index, candidate_index]:.3f} "
                f"arith={narrow} plain={plain}"
            )
            print(candidate["text"])


if __name__ == "__main__":
    main()
