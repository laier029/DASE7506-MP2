"""Regression checks for the conservative semantic answer-rank feature."""

import numpy as np

from semantic_rank_features import remainder_answer_rank


def candidate(index, answer):
    return {"index": index, "text": f"#### <numeric answer> {answer}"}


def main():
    bank = [
        {
            "id": "active",
            "question": "How many dollars are left?",
            "candidates": [
                candidate(0, 5),
                candidate(1, 10),
                candidate(2, 10),
                candidate(3, 20),
            ],
        },
        {
            "id": "inactive",
            "question": "How many dollars are there?",
            "candidates": [
                candidate(0, 5),
                candidate(1, 10),
                candidate(2, 10),
                candidate(3, 20),
            ],
        },
    ]
    feature = remainder_answer_rank(bank)
    if feature.shape != (2, 4):
        raise AssertionError(f"unexpected feature shape: {feature.shape}")
    if not feature[0, 0] < feature[0, 1] < feature[0, 3]:
        raise AssertionError(f"answer rank ordering failed: {feature[0]}")
    if not np.isclose(feature[0, 1], feature[0, 2]):
        raise AssertionError("equal answers must receive equal ranks")
    if not np.allclose(feature[1], 0.0):
        raise AssertionError("inactive questions must remain unchanged")
    print("semantic rank feature tests passed")


if __name__ == "__main__":
    main()
