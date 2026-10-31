"""Model X: full negative-filter, survivor-score and recovery cascade."""

from declared_answer_consistency import redirect_consistent_boxed_endorsement
from final_declaration_consistency import final_declaration_mismatch_groups
from numeric_plausibility import implausible_answer_groups
from omitted_factor import select_with_omitted_factor_switch
from stable_signal_verifier import stable_signal_scores


def predict(bank, arrays, checkpoint):
    scores = stable_signal_scores(bank, arrays, checkpoint)
    numeric_rejected = implausible_answer_groups(bank)
    declaration_rejected = final_declaration_mismatch_groups(bank)
    rejected = [
        numeric | declaration
        for numeric, declaration in zip(
            numeric_rejected, declaration_rejected
        )
    ]
    choices = select_with_omitted_factor_switch(
        bank,
        scores,
        0.3,
        rejected,
    )
    return redirect_consistent_boxed_endorsement(bank, scores, choices)
