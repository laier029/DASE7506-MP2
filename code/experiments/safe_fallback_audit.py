"""Audit natural support for hard-filter and boxed-recovery safe fallbacks."""

import json
from pathlib import Path
import sys


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))

from cascade_verifier import unanimous_broad_error_groups
from common import load_inputs, parse_answer
from declared_answer_consistency import boxed_answer, has_declared_answer_mismatch
from final_declaration_consistency import final_declaration_mismatch_groups
from numeric_plausibility import implausible_answer_groups


def groups(row):
    return {
        parse_answer(candidate["text"])
        for candidate in row["candidates"]
        if parse_answer(candidate["text"]) is not None
    }


def audit(split):
    bank, _ = load_inputs(split)
    arithmetic = unanimous_broad_error_groups(bank)
    numeric = implausible_answer_groups(bank)
    declaration = final_declaration_mismatch_groups(bank)
    all_removed = []
    boxed_missing = []
    boxed_conflicting = []
    for row_index, row in enumerate(bank):
        rejected = arithmetic[row_index] | numeric[row_index] | declaration[row_index]
        row_groups = groups(row)
        if row_groups and row_groups <= rejected:
            all_removed.append(row_index)
        targets = {
            boxed_answer(candidate["text"])
            for candidate in row["candidates"]
            if has_declared_answer_mismatch(candidate["text"])
        }
        targets.discard(None)
        if len(targets) > 1:
            boxed_conflicting.append(row_index)
        elif len(targets) == 1 and next(iter(targets)) not in row_groups:
            boxed_missing.append(row_index)
    return {
        "questions": len(bank),
        "hard_filter_all_groups_removed": all_removed,
        "boxed_conflicting_target_questions": boxed_conflicting,
        "boxed_protocol_invisible_target_questions": boxed_missing,
    }


def main():
    report = {
        "protocol_note": "train and validation inputs only; test sealed",
        "train": audit("train"),
        "validation": audit("validation"),
        "synthetic_fallback_regression": "covered by models/test_cascade_verifier.py",
    }
    output = CODE / "runs/model_z_robust_semantic/fallback-audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
