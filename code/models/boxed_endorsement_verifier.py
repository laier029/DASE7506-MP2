"""Model W: Model U plus consistent boxed-answer group endorsement."""

from declared_answer_consistency import redirect_consistent_boxed_endorsement
from numeric_plausibility import implausible_answer_groups
from omitted_factor import select_with_omitted_factor_switch
from stable_signal_verifier import stable_signal_scores


def predict(bank, arrays, checkpoint):
    scores = stable_signal_scores(bank, arrays, checkpoint)
    choices = select_with_omitted_factor_switch(
        bank,
        scores,
        0.3,
        implausible_answer_groups(bank),
    )
    return redirect_consistent_boxed_endorsement(bank, scores, choices)
