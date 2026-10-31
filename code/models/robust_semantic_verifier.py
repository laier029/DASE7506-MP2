"""Model Z: full cascade plus two OOF-stable semantic composition signals."""

import numpy as np

from common import parse_answer
from declared_answer_consistency import redirect_consistent_boxed_endorsement
from final_declaration_consistency import final_declaration_mismatch_groups
from numeric_plausibility import implausible_answer_groups
from omitted_factor import select_with_omitted_factor_switch
from semantic_rank_features import semantic_rank_adjusted_scores
from stable_signal_verifier import stable_signal_scores
from terminal_dependency_features import terminal_dependency_adjusted_scores


def _predict_disagreements(bank, arrays, checkpoint):
    scores = stable_signal_scores(bank, arrays, checkpoint)
    scores = semantic_rank_adjusted_scores(bank, scores)
    scores = terminal_dependency_adjusted_scores(bank, scores)
    numeric_rejected = implausible_answer_groups(bank)
    declaration_rejected = final_declaration_mismatch_groups(bank)
    rejected = [
        numeric | declaration
        for numeric, declaration in zip(
            numeric_rejected, declaration_rejected
        )
    ]
    choices = select_with_omitted_factor_switch(bank, scores, 0.3, rejected)
    return redirect_consistent_boxed_endorsement(bank, scores, choices)


def _unanimous_parseable_choice(row):
    parsed = [parse_answer(candidate["text"]) for candidate in row["candidates"]]
    answers = {answer for answer in parsed if answer is not None}
    if len(answers) != 1:
        return None
    answer = next(iter(answers))
    return parsed.index(answer)


def predict(bank, arrays, checkpoint):
    """Short-circuit unanimous answers and verify only disagreement rows."""
    unanimous = [_unanimous_parseable_choice(row) for row in bank]
    active = np.asarray(
        [index for index, choice in enumerate(unanimous) if choice is None],
        dtype=np.int64,
    )
    output = [0 if choice is None else choice for choice in unanimous]
    if not len(active):
        return output
    active_bank = [bank[index] for index in active]
    active_arrays = {
        name: values[active]
        if (
            isinstance(values, np.ndarray)
            and values.ndim > 0
            and values.shape[0] == len(bank)
        )
        else values
        for name, values in arrays.items()
    }
    active_choices = _predict_disagreements(
        active_bank, active_arrays, checkpoint
    )
    for row_index, choice in zip(active, active_choices):
        output[int(row_index)] = int(choice)
    return output
