"""Measure whether a candidate conclusion preserves the question target."""

import numpy as np

from answer_step_consistency import _reasoning_prefix


TARGETS = (
    "difference",
    "reduction",
    "change",
    "remaining",
    "left",
    "saving",
    "saved",
    "total",
    "sum",
    "profit",
    "cost",
    "price",
    "time",
    "distance",
    "number",
    "amount",
)


def terminal_target_recall(bank):
    """Return recall of question target words in the final reasoning window."""
    output = np.zeros((len(bank), 4), dtype=np.float32)
    for row_index, row in enumerate(bank):
        question = row["question"].lower()
        targets = [target for target in TARGETS if target in question]
        if not targets:
            continue
        for candidate_index, candidate in enumerate(row["candidates"]):
            tail = _reasoning_prefix(candidate["text"]).lower()[-300:]
            output[row_index, candidate_index] = sum(
                target in tail for target in targets
            ) / len(targets)
    return output
