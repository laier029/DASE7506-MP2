"""Probe explicit final-unit mismatches as a high-precision hard gate."""

from pathlib import Path
import re
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from common import load_inputs, parse_answer
from dimension_consistency_probe import choices, correct
from group_ranker_probe import candidate_labels
from model_t_probe_utils import load_or_build_model_t_oof
from stable_signal_features import stable_signal_adjusted_scores
from stable_signal_verifier import stable_signal_scores


UNIT_PATTERNS = {
    "seconds": re.compile(r"\bseconds?\b", re.I),
    "minutes": re.compile(r"\bminutes?\b", re.I),
    "hours": re.compile(r"\bhours?\b", re.I),
    "days": re.compile(r"\bdays?\b", re.I),
    "weeks": re.compile(r"\bweeks?\b", re.I),
    "months": re.compile(r"\bmonths?\b", re.I),
    "years": re.compile(r"\byears?\b", re.I),
    "inches": re.compile(r"\binches?\b", re.I),
    "feet": re.compile(r"\b(?:feet|foot)\b", re.I),
    "yards": re.compile(r"\byards?\b", re.I),
    "meters": re.compile(r"\bmet(?:er|re)s?\b", re.I),
    "miles": re.compile(r"\bmiles?\b", re.I),
    "ounces": re.compile(r"\bounces?\b", re.I),
    "pounds": re.compile(r"\bpounds?\b", re.I),
    "grams": re.compile(r"\bgrams?\b", re.I),
    "kilograms": re.compile(r"\bkilograms?\b", re.I),
    "dollars": re.compile(r"\bdollars?\b", re.I),
    "cents": re.compile(r"\bcents?\b", re.I),
    "euros": re.compile(r"\beuros?\b", re.I),
    "pesetas": re.compile(r"\bpesetas?\b", re.I),
    "square_inches": re.compile(r"\bsquare\s+inches?\b", re.I),
    "cubic_inches": re.compile(r"\bcubic\s+inches?\b", re.I),
    "square_feet": re.compile(r"\bsquare\s+(?:feet|foot)\b", re.I),
    "cubic_feet": re.compile(r"\bcubic\s+(?:feet|foot)\b", re.I),
}


def last_unit(text):
    matches = []
    for unit, pattern in UNIT_PATTERNS.items():
        matches.extend((match.start(), unit) for match in pattern.finditer(text))
    return max(matches)[1] if matches else None


def expected_unit(question):
    """Use only the final question clause, where the requested unit is stated."""
    tail = re.split(r"[?.!]", question)[-2] if "?" in question else question
    unit = last_unit(tail)
    if unit in {"inches", "feet"}:
        if re.search(r"\bsquare\s+(?:inches?|feet|foot)\b", tail, re.I):
            unit = "square_inches" if unit == "inches" else "square_feet"
        elif re.search(r"\bcubic\s+(?:inches?|feet|foot)\b", tail, re.I):
            unit = "cubic_inches" if unit == "inches" else "cubic_feet"
    return unit


def candidate_final_unit(text):
    headings = list(
        re.finditer(r"(?im)^\s*(?:#{1,6}\s*)?(?:final\s+answer|answer)\s*:", text)
    )
    tail = text[headings[-1].end() :] if headings else text[-450:]
    return last_unit(tail)


def mismatch_groups(bank):
    output = []
    for row in bank:
        expected = expected_unit(row["question"])
        groups = {}
        for index, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate["text"])
            if answer is not None:
                groups.setdefault(answer, []).append(index)
        rejected = set()
        if expected is not None:
            for answer, indices in groups.items():
                actual = [
                    candidate_final_unit(row["candidates"][index]["text"])
                    for index in indices
                ]
                if all(unit is not None and unit != expected for unit in actual):
                    rejected.add(answer)
        output.append(rejected)
    return output


def precision_summary(name, bank, labels, rejected):
    total = wrong = 0
    for row_index, row_rejected in enumerate(rejected):
        for answer in row_rejected:
            indices = [
                index
                for index, candidate in enumerate(bank[row_index]["candidates"])
                if parse_answer(candidate["text"]) == answer
            ]
            total += 1
            wrong += int(not any(labels[row_index, indices]))
    print(
        f"{name}_groups={total} wrong={wrong} "
        f"wrong_precision={wrong / max(total, 1):.6f}"
    )


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    train_rejected = mismatch_groups(train_bank)
    validation_rejected = mismatch_groups(validation_bank)
    precision_summary("train", train_bank, train_labels, train_rejected)
    precision_summary("validation", validation_bank, validation_labels, validation_rejected)

    oof_scores = [
        stable_signal_adjusted_scores(train_bank, scores)
        for scores in load_or_build_model_t_oof(train_bank, train_arrays, train_labels)
    ]
    empty_train = [set() for _ in train_bank]
    deltas = []
    changes = []
    for scores in oof_scores:
        base = choices(train_bank, scores, empty_train)
        proposed = choices(train_bank, scores, train_rejected)
        deltas.append(correct(train_labels, proposed) - correct(train_labels, base))
        changes.append(int(np.sum(base != proposed)))

    validation_scores = stable_signal_scores(
        validation_bank, validation_arrays, CODE / "runs/model_x_full_cascade"
    )
    empty_validation = [set() for _ in validation_bank]
    base = choices(validation_bank, validation_scores, empty_validation)
    proposed = choices(validation_bank, validation_scores, validation_rejected)
    print(f"oof_deltas={deltas} oof_choice_changes={changes}")
    print(
        f"validation_delta={correct(validation_labels, proposed) - correct(validation_labels, base):+d} "
        f"validation_changes={np.flatnonzero(base != proposed).tolist()}"
    )


if __name__ == "__main__":
    main()
