"""Conditionally switch answer groups when their ratio exposes an omitted factor."""

from collections import defaultdict
from fractions import Fraction

from calculation_coverage import (
    calculation_context,
    calculation_quantities,
    quantities,
)
from cascade_verifier import unanimous_broad_error_groups
from common import parse_answer


_MAX_GROUP_MARGIN = 1.25
_MINIMUM_SELECTED_GROUP_SIZE = 2


def _fraction_set(values):
    output = set()
    for value in values:
        try:
            output.add(Fraction(value))
        except (ValueError, ZeroDivisionError):
            continue
    return output


def _answer_groups(row):
    groups = defaultdict(list)
    for candidate_index, candidate in enumerate(row["candidates"]):
        answer = parse_answer(candidate["text"])
        if answer is not None:
            groups[answer].append(candidate_index)
    return groups


def select_with_omitted_factor_switch(
    bank, scores, group_mean_weight, additional_rejected
):
    """Run hard gates, group scoring, then a tightly gated factor switch."""
    arithmetic_rejected = unanimous_broad_error_groups(bank)
    if len(additional_rejected) != len(bank):
        raise ValueError("Additional rejection rows must align with the bank.")

    selected_choices = []
    for row_index, (row, row_scores) in enumerate(zip(bank, scores)):
        groups = _answer_groups(row)
        if not groups:
            raise ValueError("Question has no candidate with a parseable answer.")
        rejected = arithmetic_rejected[row_index] | set(
            additional_rejected[row_index]
        )
        survivors = {
            answer: indices
            for answer, indices in groups.items()
            if answer not in rejected
        }
        if not survivors:
            survivors = groups

        scored = []
        for answer, indices in survivors.items():
            values = row_scores[indices]
            candidate = indices[int(values.argmax())]
            group_score = float(
                values.max() + group_mean_weight * values.mean()
            )
            scored.append((group_score, answer, indices, candidate))
        scored.sort(reverse=True)
        selected = scored[0]

        if len(selected[2]) >= _MINIMUM_SELECTED_GROUP_SIZE:
            selected_value = Fraction(selected[1])
            if selected_value != 0:
                question_values = _fraction_set(
                    quantities(row["question"], include_words=True)
                )
                candidate_values = [
                    _fraction_set(
                        calculation_quantities(
                            calculation_context(candidate["text"]),
                            include_words=True,
                        )
                    )
                    for candidate in row["candidates"]
                ]
                selected_used = set().union(
                    *(candidate_values[index] for index in selected[2])
                )
                alternatives = []
                for item in scored[1:]:
                    alternative_value = Fraction(item[1])
                    if alternative_value == 0:
                        continue
                    ratio = abs(alternative_value / selected_value)
                    if ratio < 1:
                        ratio = 1 / ratio
                    if ratio <= 1 or ratio not in question_values:
                        continue
                    if ratio in selected_used:
                        continue
                    alternative_used = set().union(
                        *(candidate_values[index] for index in item[2])
                    )
                    if ratio not in alternative_used:
                        continue
                    if selected[0] - item[0] > _MAX_GROUP_MARGIN:
                        continue
                    alternatives.append(item)
                if alternatives:
                    selected = max(alternatives)

        selected_choices.append(selected[3])
    return selected_choices
