"""Regression checks for Model Z's unanimous-answer short circuit."""

import numpy as np

import robust_semantic_verifier
from robust_semantic_verifier import _unanimous_parseable_choice


def row(*answers):
    return {
        "candidates": [
            {"text": f"#### <numeric answer> {answer}"}
            for answer in answers
        ]
    }


def main():
    if _unanimous_parseable_choice(row(5, 5, 5, 5)) != 0:
        raise AssertionError("unanimous parseable answers must short-circuit")
    if _unanimous_parseable_choice(row(5, 5, 6, 5)) is not None:
        raise AssertionError("answer disagreement was not detected")
    partly_unparseable = {
        "candidates": [
            {"text": "No numeric answer"},
            {"text": "#### <numeric answer> 7"},
            {"text": "No numeric answer"},
            {"text": "#### <numeric answer> 7"},
        ]
    }
    if _unanimous_parseable_choice(partly_unparseable) != 1:
        raise AssertionError("short circuit selected an unparseable candidate")

    mixed_bank = [row(5, 5, 5, 5), row(1, 2, 3, 4)]
    arrays = {
        "embeddings": np.zeros((2, 4, 3), dtype=np.float32),
        "scalar": np.asarray(7.0, dtype=np.float32),
        "unrelated_rows": np.zeros((3, 2), dtype=np.float32),
    }
    original = robust_semantic_verifier._predict_disagreements

    def fake_disagreement_predict(active_bank, active_arrays, checkpoint):
        if len(active_bank) != 1:
            raise AssertionError("short-circuited row reached full verifier")
        if active_arrays["embeddings"].shape != (1, 4, 3):
            raise AssertionError("aligned input array was not sliced")
        if active_arrays["scalar"].ndim != 0:
            raise AssertionError("scalar metadata should remain unchanged")
        if active_arrays["unrelated_rows"].shape != (3, 2):
            raise AssertionError("unrelated array should remain unchanged")
        return [2]

    robust_semantic_verifier._predict_disagreements = fake_disagreement_predict
    try:
        choices = robust_semantic_verifier.predict(mixed_bank, arrays, None)
    finally:
        robust_semantic_verifier._predict_disagreements = original
    if choices != [0, 2] or any(type(choice) is not int for choice in choices):
        raise AssertionError(f"misaligned or non-native choices: {choices}")
    print("robust semantic verifier checks passed")


if __name__ == "__main__":
    main()
