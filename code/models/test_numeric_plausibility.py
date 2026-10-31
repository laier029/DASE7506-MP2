"""Regression checks for non-integral answer-group rejection and fallback."""

from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cascade_verifier import select_after_negative_filter
from numeric_plausibility import (
    fractional_answer_groups,
    implausible_answer_groups,
    safe_nonpositive_answer_groups,
)


def row(answers):
    return {
        "candidates": [
            {"index": index, "text": f"Reasoning.\n#### {answer}"}
            for index, answer in enumerate(answers)
        ]
    }


def main():
    bank = [
        row(("1/2", "2", "4/2", "3")),
        row(("1/2", "3/2", "5/2", "7/2")),
    ]
    rejected = fractional_answer_groups(bank)
    if rejected != [{"1/2"}, {"1/2", "3/2", "5/2", "7/2"}]:
        raise AssertionError(f"unexpected fractional groups: {rejected}")
    scores = np.asarray([[3.0, 1.0, 2.0, 0.0], [4.0, 3.0, 2.0, 1.0]])
    selected = select_after_negative_filter(
        bank, scores, 0.3, additional_rejected=rejected
    )
    # 4/2 canonicalizes to the integral group "2"; an all-fractional question
    # uses the whole-question fallback and preserves the original best choice.
    if selected != [2, 0]:
        raise AssertionError(f"unexpected plausibility choices: {selected}")

    signed_bank = [
        {
            "question": "The temperature was -8 degrees. What was the average?",
            "candidates": row(("-8", "1", "2", "3"))["candidates"],
        },
        {
            "question": "How much change remains?",
            "candidates": row(("-5", "0", "5", "10"))["candidates"],
        },
    ]
    safe = safe_nonpositive_answer_groups(signed_bank)
    if safe != [set(), {"-5", "0"}]:
        raise AssertionError(f"unexpected safe nonpositive groups: {safe}")
    combined = implausible_answer_groups(signed_bank)
    if combined != [set(), {"-5", "0"}]:
        raise AssertionError(f"unexpected combined groups: {combined}")
    print("numeric plausibility checks passed")


if __name__ == "__main__":
    main()
