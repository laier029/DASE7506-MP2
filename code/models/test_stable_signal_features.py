"""Regression checks for Model U's stable weak signals."""

import numpy as np

from stable_signal_features import (
    broad_problem_indicator,
    implicit_factor_coverage,
    stable_signal_adjusted_scores,
)


def main():
    bank = [
        {
            "question": "Both guards work each day for a week.",
            "candidates": [
                {"text": "$8 * 7 * 2 = 112$\n#### 112"},
                {"text": "$8 * 7 = 56$\n#### 56"},
                {"text": "$8 + 7 = 15$\n#### 15"},
                {"text": "No calculation.\n#### 8"},
            ],
        },
        {
            "question": "Use 4 and 5 to calculate the total.",
            "candidates": [
                {"text": "$4 + 5 = 10$\n#### 10"},
                {"text": "$4 + 5 = 9$\n#### 9"},
                {"text": "$4 * 5 = 20$\n#### 20"},
                {"text": "$4 - 5 = -1$\n#### -1"},
            ],
        },
    ]
    implicit = implicit_factor_coverage(bank)
    if not implicit[0, 0] > implicit[0, 1]:
        raise AssertionError(f"implicit factor ordering failed: {implicit[0]}")
    broad = broad_problem_indicator(bank)
    if broad[1, 0] != 1 or broad[1, 1] != 0:
        raise AssertionError(f"broad arithmetic flags failed: {broad[1]}")
    scores = stable_signal_adjusted_scores(
        bank, np.zeros((2, 4), dtype=np.float32)
    )
    if scores.shape != (2, 4) or not np.isfinite(scores).all():
        raise AssertionError("stable signal scores must be finite and aligned")
    print("stable signal feature checks passed")


if __name__ == "__main__":
    main()
