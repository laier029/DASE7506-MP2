"""Regression checks for cumulative terminal dependencies."""

import numpy as np

from terminal_dependency_features import terminal_dependency_coverage


def main():
    bank = [
        {
            "question": "There are some items. How many are there now?",
            "candidates": [
                {"text": "$2 + 3 = 5$\n$5 + 4 = 9$\n#### <numeric answer> 9"},
                {"text": "$2 + 3 = 5$\n#### <numeric answer> 5"},
                {"text": "No equation.\n#### <numeric answer> 7"},
                {"text": "$4 + 4 = 8$\n#### <numeric answer> 8"},
            ],
        },
        {
            "question": "How many items are there?",
            "candidates": [
                {"text": "$2 + 3 = 5$\n#### <numeric answer> 5"}
            ] * 4,
        },
    ]
    feature = terminal_dependency_coverage(bank)
    if feature.shape != (2, 4) or not np.isfinite(feature).all():
        raise AssertionError("dependency feature must be finite and aligned")
    if feature[0, 0] <= feature[0, 1]:
        raise AssertionError(f"expanded dependency was not rewarded: {feature[0]}")
    if not np.allclose(feature[1], 0.0):
        raise AssertionError("inactive questions must remain unchanged")
    print("terminal dependency feature tests passed")


if __name__ == "__main__":
    main()
