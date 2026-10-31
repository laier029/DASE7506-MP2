"""Regression checks for protocol-aware declared-answer recovery."""

import numpy as np

from declared_answer_consistency import (
    boxed_answer,
    has_declared_answer_mismatch,
    redirect_consistent_boxed_endorsement,
    redirect_selected_mismatch,
)


def main():
    hijacked = (
        "#### Job 3\n1 toilet and 2 sinks.\n"
        "### Final Answer:\n$$\\boxed{250}$$"
    )
    if boxed_answer(hijacked) != "250":
        raise AssertionError("last boxed answer was not canonicalized")
    if not has_declared_answer_mismatch(hijacked):
        raise AssertionError("Markdown heading hijack was not detected")
    bank = [
        {
            "candidates": [
                {"text": "#### <numeric answer> 250"},
                {"text": hijacked},
                {"text": "#### <numeric answer> 300"},
                {"text": "No parseable answer"},
            ]
        }
    ]
    scores = np.asarray([[0.2, 1.0, 0.5, -1.0]], dtype=np.float32)
    redirected = redirect_selected_mismatch(bank, scores, [1])
    if redirected != [0]:
        raise AssertionError(f"unexpected redirect: {redirected}")
    preserved = redirect_selected_mismatch(bank, scores, [2])
    if preserved != [2]:
        raise AssertionError(f"safe fallback changed a consistent choice: {preserved}")
    endorsed = redirect_consistent_boxed_endorsement(bank, scores, [2])
    if endorsed != [0]:
        raise AssertionError(f"boxed endorsement was not applied: {endorsed}")

    invisible = [
        {
            "candidates": [
                {"text": hijacked},
                {"text": "#### <numeric answer> 300"},
                {"text": "#### <numeric answer> 400"},
                {"text": "#### <numeric answer> 500"},
            ]
        }
    ]
    invisible_scores = np.asarray([[1.0, 0.8, 0.2, 0.0]], dtype=np.float32)
    if redirect_consistent_boxed_endorsement(
        invisible, invisible_scores, [1]
    ) != [1]:
        raise AssertionError("protocol-invisible boxed target changed the choice")

    conflicting = [
        {
            "candidates": [
                {"text": hijacked},
                {
                    "text": (
                        "#### Job 4\n2 sinks.\n"
                        "### Final Answer:\n$$\\boxed{260}$$"
                    )
                },
                {"text": "#### <numeric answer> 300"},
                {"text": "#### <numeric answer> 400"},
            ]
        }
    ]
    conflicting_scores = np.asarray([[1.0, 0.9, 0.8, 0.0]], dtype=np.float32)
    if redirect_consistent_boxed_endorsement(
        conflicting, conflicting_scores, [2]
    ) != [2]:
        raise AssertionError("conflicting boxed targets changed the choice")
    print("declared answer consistency checks passed")


if __name__ == "__main__":
    main()
