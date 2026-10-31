"""Check the marginal value of a safe nonpositive gate on Model S."""

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
from numeric_plausibility import fractional_answer_groups
from occurrence_plausibility_verifier import occurrence_adjusted_scores
from omitted_factor import select_with_omitted_factor_switch
from structure_text_fusion_verifier import structure_text_scores
from target_omitted_factor_verifier import target_adjusted_scores


NEGATIVE_INPUT = re.compile(r"(?<!\w)-\s*\d")


def safe_nonpositive_groups(bank):
    output = []
    for row in bank:
        rejected = set()
        if NEGATIVE_INPUT.search(row["question"]) is None:
            for candidate in row["candidates"]:
                answer = parse_answer(candidate["text"])
                if answer is not None and Fraction(answer) <= 0:
                    rejected.add(answer)
        output.append(rejected)
    return output


def correct_count(labels, choices):
    return int(labels[np.arange(len(labels)), choices].sum())


def main():
    bank, arrays = load_inputs("validation")
    labels = candidate_labels("validation", bank).astype(bool)
    scores = structure_text_scores(
        bank, arrays, CODE / "runs/model_s_target_alignment"
    )
    scores = coverage_adjusted_scores(bank, scores)
    scores = occurrence_adjusted_scores(bank, scores)
    scores = target_adjusted_scores(bank, scores)
    fractional = fractional_answer_groups(bank)
    nonpositive = safe_nonpositive_groups(bank)
    combined = [left | right for left, right in zip(fractional, nonpositive)]
    base = np.asarray(
        select_with_omitted_factor_switch(bank, scores, 0.3, fractional)
    )
    filtered = np.asarray(
        select_with_omitted_factor_switch(bank, scores, 0.3, combined)
    )
    changes = np.flatnonzero(base != filtered)
    print(
        f"base={correct_count(labels, base)} "
        f"filtered={correct_count(labels, filtered)} "
        f"delta={correct_count(labels, filtered) - correct_count(labels, base):+d} "
        f"changes={changes.tolist()}"
    )
    for row_index in changes:
        print(
            f"  q={row_index} base_correct={bool(labels[row_index, base[row_index]])} "
            f"filtered_correct={bool(labels[row_index, filtered[row_index]])}"
        )


if __name__ == "__main__":
    main()
