"""Probe a precision-first target-dimension mismatch hard gate."""

from pathlib import Path
import re
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from common import load_inputs, parse_answer
from declared_answer_consistency import redirect_consistent_boxed_endorsement
from final_declaration_consistency import final_declaration_mismatch_groups
from group_ranker_probe import candidate_labels
from model_t_probe_utils import load_or_build_model_t_oof
from numeric_plausibility import implausible_answer_groups
from omitted_factor import select_with_omitted_factor_switch
from stable_signal_features import stable_signal_adjusted_scores
from stable_signal_verifier import stable_signal_scores


LINEAR_UNIT = (
    r"(?:inches|inch|feet|foot|meters|meter|metres|metre|yards|yard|"
    r"centimeters|centimeter|centimetres|centimetre)"
)
SQUARE_TARGET = re.compile(rf"\b(?:square\s+{LINEAR_UNIT}|area)\b", re.I)
CUBIC_TARGET = re.compile(rf"\b(?:cubic\s+{LINEAR_UNIT}|volume)\b", re.I)


def dimension_mismatch(text, question):
    """Whether a candidate changes an explicit area/volume target dimension."""
    question_square = bool(SQUARE_TARGET.search(question))
    question_cubic = bool(CUBIC_TARGET.search(question))
    candidate_square = bool(SQUARE_TARGET.search(text))
    candidate_cubic = bool(CUBIC_TARGET.search(text))
    if question_square and not question_cubic:
        return candidate_cubic and not candidate_square
    if question_cubic and not question_square:
        return candidate_square and not candidate_cubic
    return False


def mismatch_groups(bank):
    output = []
    for row in bank:
        groups = {}
        for index, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate["text"])
            if answer is not None:
                groups.setdefault(answer, []).append(index)
        output.append(
            {
                answer
                for answer, indices in groups.items()
                if all(
                    dimension_mismatch(
                        row["candidates"][index]["text"], row["question"]
                    )
                    for index in indices
                )
            }
        )
    return output


def precision_summary(name, bank, labels, rejected):
    total = wrong = 0
    for row_index, row in enumerate(bank):
        for answer in rejected[row_index]:
            indices = [
                index
                for index, candidate in enumerate(row["candidates"])
                if parse_answer(candidate["text"]) == answer
            ]
            total += 1
            wrong += int(not any(labels[row_index, indices]))
    print(
        f"{name}_rejected_groups={total} wrong={wrong} "
        f"wrong_precision={wrong / max(total, 1):.6f}"
    )


def choices(bank, scores, extra_rejected):
    numeric = implausible_answer_groups(bank)
    declaration = final_declaration_mismatch_groups(bank)
    rejected = [
        left | middle | right
        for left, middle, right in zip(numeric, declaration, extra_rejected)
    ]
    selected = np.asarray(
        select_with_omitted_factor_switch(bank, scores, 0.3, rejected)
    )
    return np.asarray(
        redirect_consistent_boxed_endorsement(bank, scores, selected)
    )


def correct(labels, selected):
    return int(labels[np.arange(len(labels)), selected].sum())


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    train_rejected = mismatch_groups(train_bank)
    validation_rejected = mismatch_groups(validation_bank)
    precision_summary("train", train_bank, train_labels, train_rejected)
    precision_summary(
        "validation", validation_bank, validation_labels, validation_rejected
    )

    oof_scores = [
        stable_signal_adjusted_scores(train_bank, scores)
        for scores in load_or_build_model_t_oof(
            train_bank, train_arrays, train_labels
        )
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
        validation_bank,
        validation_arrays,
        CODE / "runs/model_x_full_cascade",
    )
    empty_validation = [set() for _ in validation_bank]
    base = choices(validation_bank, validation_scores, empty_validation)
    proposed = choices(validation_bank, validation_scores, validation_rejected)
    changed = np.flatnonzero(base != proposed).tolist()
    print(f"oof_deltas={deltas} oof_choice_changes={changes}")
    print(
        f"validation_delta={correct(validation_labels, proposed) - correct(validation_labels, base):+d} "
        f"validation_changes={changed}"
    )


if __name__ == "__main__":
    main()
