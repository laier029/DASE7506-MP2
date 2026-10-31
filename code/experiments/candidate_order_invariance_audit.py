"""Audit whether Model Z answer selection is invariant to candidate reordering."""

import json
from pathlib import Path
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))

from common import load_inputs, parse_answer
from robust_semantic_verifier import predict


SEEDS = (7506, 11, 29, 47, 83)


def _trial(bank, arrays, checkpoint, original, seed):
    rng = np.random.default_rng(seed)
    permutations = np.stack([rng.permutation(4) for _ in bank])
    permuted_bank = []
    for row, order in zip(bank, permutations):
        copied = dict(row)
        copied["candidates"] = [
            {**row["candidates"][int(old_index)], "index": new_index}
            for new_index, old_index in enumerate(order)
        ]
        permuted_bank.append(copied)
    permuted_arrays = dict(arrays)
    permuted_arrays["embeddings"] = np.stack(
        [row[order] for row, order in zip(arrays["embeddings"], permutations)]
    )
    permuted = np.asarray(predict(permuted_bank, permuted_arrays, checkpoint))
    mapped = permutations[np.arange(len(bank)), permuted]
    candidate_changes = np.flatnonzero(original != mapped).tolist()
    answer_changes = [
        row_index
        for row_index in candidate_changes
        if parse_answer(bank[row_index]["candidates"][original[row_index]]["text"])
        != parse_answer(bank[row_index]["candidates"][mapped[row_index]]["text"])
    ]
    return {
        "seed": seed,
        "candidate_index_changes_after_mapping": candidate_changes,
        "answer_group_changes_after_mapping": answer_changes,
    }


def main():
    bank, arrays = load_inputs("validation")
    checkpoint = CODE / "runs/model_z_robust_semantic"
    original = np.asarray(predict(bank, arrays, checkpoint))
    trials = [
        _trial(bank, arrays, checkpoint, original, seed) for seed in SEEDS
    ]
    answer_changes = sorted(
        {
            index
            for trial in trials
            for index in trial["answer_group_changes_after_mapping"]
        }
    )
    report = {
        "protocol_note": "validation inputs only; no references read",
        "seeds": list(SEEDS),
        "questions": len(bank),
        "trials": trials,
        "answer_group_changes_after_mapping": answer_changes,
    }
    output = CODE / "runs/model_z_robust_semantic/order-invariance-audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
