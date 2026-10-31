"""Regression checks for the margin-gated omitted-factor switch."""

from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from omitted_factor import select_with_omitted_factor_switch


def candidate(reasoning, answer, index):
    return {"index": index, "text": f"{reasoning}\n#### {answer}"}


def main():
    row = {
        "question": (
            "Tim hires 2 bodyguards at 20 dollars for 8 hours over 7 days."
        ),
        "candidates": [
            candidate("$20 * 8 * 7 = 1120$", "1120", 0),
            candidate("$20 * 8 * 7 = 1120$", "1120", 1),
            candidate("$20 * 8 * 7 = 1120$", "1120", 2),
            candidate("$2 * 20 * 8 * 7 = 2240$", "2240", 3),
        ],
    }
    scores = np.asarray([[1.0, 0.8, 0.7, 0.5]])
    selected = select_with_omitted_factor_switch(
        [row], scores, 0.3, [set()]
    )
    if selected != [3]:
        raise AssertionError(f"expected omitted-factor switch, got {selected}")

    high_margin_scores = np.asarray([[3.0, 2.9, 2.8, 0.0]])
    selected = select_with_omitted_factor_switch(
        [row], high_margin_scores, 0.3, [set()]
    )
    if selected != [0]:
        raise AssertionError(f"high-margin majority should remain, got {selected}")
    print("omitted-factor checks passed")


if __name__ == "__main__":
    main()
