"""Audit empirical precision and finite-sample support of hard gates."""

import argparse
import json
import math
from pathlib import Path
import sys


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from cascade_verifier import unanimous_broad_error_groups
from common import load_inputs, parse_answer
from final_declaration_consistency import final_declaration_mismatch_groups
from group_ranker_probe import candidate_labels
from numeric_plausibility import (
    fractional_answer_groups,
    safe_nonpositive_answer_groups,
)


GATES = {
    "unanimous_broad_arithmetic": unanimous_broad_error_groups,
    "fractional_answer": fractional_answer_groups,
    "safe_nonpositive_answer": safe_nonpositive_answer_groups,
    "explicit_final_declaration_conflict": final_declaration_mismatch_groups,
}


def _audit_groups(bank, labels, rejected):
    total = 0
    wrong = 0
    for row_index, (row, answers) in enumerate(zip(bank, rejected)):
        for answer in answers:
            indices = [
                index
                for index, candidate in enumerate(row["candidates"])
                if parse_answer(candidate["text"]) == answer
            ]
            total += 1
            wrong += int(not labels[row_index, indices].any())
    precision = wrong / total if total else None
    # For n successes in n Bernoulli trials, P(X=n | p)=p**n. Solving
    # p**n=alpha gives the one-sided Clopper-Pearson lower bound.
    exact_lower = (
        math.pow(0.05, 1.0 / total) if total and wrong == total else None
    )
    return {
        "rejected_answer_groups": total,
        "wrong_answer_groups": wrong,
        "observed_wrong_precision": precision,
        "one_sided_95pct_exact_lower_if_zero_false_rejections": exact_lower,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="runs/model_z_robust_semantic/gate-precision-audit.json",
    )
    args = parser.parse_args()
    report = {
        "protocol_note": "train and validation references only; test sealed",
        "interpretation": (
            "The exact lower bound quantifies finite-sample uncertainty only "
            "when every flagged answer group was wrong."
        ),
        "splits": {},
    }
    for split in ("train", "validation"):
        bank, _ = load_inputs(split)
        labels = candidate_labels(split, bank).astype(bool)
        report["splits"][split] = {
            name: _audit_groups(bank, labels, gate(bank))
            for name, gate in GATES.items()
        }
    output = CODE / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
