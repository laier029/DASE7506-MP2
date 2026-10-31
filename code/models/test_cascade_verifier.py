"""Regression checks for Model N's precision-first group filter."""

from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cascade_verifier import (
    select_after_negative_filter,
    unanimous_broad_error_groups,
)


def candidate(text, index):
    return {"index": index, "text": text}


def main():
    repeated_wrong = {
        "candidates": [
            candidate("$$0.6/9 = 2/15 \\text{ hours}$$\n#### 8", 0),
            candidate("$$0.6/9 = 2/15 \\text{ hours}$$\n#### 8", 1),
            candidate("$$0.6/9 = 0.0667 \\text{ hours}$$\n#### 4", 2),
            candidate("No calculation.\n#### 5", 3),
        ]
    }
    singleton_error = {
        "candidates": [
            candidate("$$1+1+1+2+1.5+1=7 \\text{ pounds}$$\n#### 7", 0),
            candidate("No calculation.\n#### 8", 1),
            candidate("No calculation.\n#### 9", 2),
            candidate("No calculation.\n#### 10", 3),
        ]
    }
    bank = [repeated_wrong, singleton_error]
    rejected = unanimous_broad_error_groups(bank)
    if rejected != [{"8"}, set()]:
        raise AssertionError(f"unexpected rejected groups: {rejected}")

    scores = np.asarray([[2.0, 1.5, 0.0, -1.0], [2.0, 1.0, 0.0, -1.0]])
    selected = select_after_negative_filter(bank, scores, 0.3)
    if selected != [2, 0]:
        raise AssertionError(f"unexpected cascade choices: {selected}")

    all_rejected = {
        "candidates": [
            candidate("#### 1", 0),
            candidate("#### 2", 1),
            candidate("#### 3", 2),
            candidate("#### 4", 3),
        ]
    }
    fallback = select_after_negative_filter(
        [all_rejected],
        np.asarray([[0.0, 1.0, 3.0, 2.0]]),
        0.3,
        additional_rejected=[{"1", "2", "3", "4"}],
    )
    if fallback != [2]:
        raise AssertionError(f"all-groups-removed fallback failed: {fallback}")
    print("cascade verifier checks passed")


if __name__ == "__main__":
    main()
