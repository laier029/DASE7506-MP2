"""Build a compact, validation-only catalog of Model Z errors."""

import argparse
import json
from pathlib import Path
import sys


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))

from answer_step_consistency import _reasoning_prefix
from common import DATA, load_bank, parse_answer, read_jsonl, validate_choices


def _tail(text, limit=360):
    compact = " ".join(_reasoning_prefix(text).split())
    return compact[-limit:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default=(
            "runs/model_z_robust_semantic/"
            "submission-validation-predictions.json"
        ),
    )
    parser.add_argument(
        "--output",
        default="runs/model_z_robust_semantic/validation-error-catalog.json",
    )
    args = parser.parse_args()
    bank = load_bank("validation")
    references = read_jsonl(DATA / "gsm8k_validation.jsonl")
    predictions = json.loads((CODE / args.predictions).read_text(encoding="utf-8"))
    validate_choices(bank, predictions["ids"], predictions["selected"])
    errors = []
    for row_index, (row, reference, selected) in enumerate(
        zip(bank, references, predictions["selected"])
    ):
        gold = parse_answer(reference["answer"])
        chosen = parse_answer(row["candidates"][selected]["text"])
        if chosen == gold:
            continue
        errors.append(
            {
                "row_index": row_index,
                "id": row["id"],
                "question": row["question"],
                "gold_answer": gold,
                "selected_candidate": selected,
                "selected_answer": chosen,
                "candidates": [
                    {
                        "index": candidate["index"],
                        "parsed_answer": parse_answer(candidate["text"]),
                        "reasoning_tail": _tail(candidate["text"]),
                    }
                    for candidate in row["candidates"]
                ],
            }
        )
    report = {
        "protocol_note": "validation errors only; test references sealed",
        "error_count": len(errors),
        "errors": errors,
    }
    output = CODE / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(errors)} errors to {output.relative_to(CODE)}")


if __name__ == "__main__":
    main()
