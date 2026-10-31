"""Audit coverage and empirical safety of the unique-answer short circuit."""

import json
from pathlib import Path
import sys


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from common import load_inputs
from group_ranker_probe import candidate_labels
from robust_semantic_verifier import _unanimous_parseable_choice


def audit(split):
    bank, _ = load_inputs(split)
    labels = candidate_labels(split, bank).astype(bool)
    choices = [_unanimous_parseable_choice(row) for row in bank]
    rows = [index for index, choice in enumerate(choices) if choice is not None]
    correct = sum(labels[index, choices[index]] for index in rows)
    return {
        "questions": len(bank),
        "short_circuited": len(rows),
        "fully_verified": len(bank) - len(rows),
        "short_circuit_correct": int(correct),
        "short_circuit_accuracy": float(correct / len(rows)) if rows else None,
    }


def main():
    report = {
        "protocol_note": "train and validation only; test sealed",
        "train": audit("train"),
        "validation": audit("validation"),
    }
    output = CODE / "runs/model_z_robust_semantic/short-circuit-audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
