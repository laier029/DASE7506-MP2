"""Baseline B: select the most common parsed final answer."""

from collections import Counter

from common import parse_answer


def predict(bank, arrays, checkpoint):
    """Choose the earliest candidate in the largest valid-answer group.

    Ties are resolved by candidate order. If no final answer can be parsed,
    candidate 0 is selected.
    """
    choices = []

    for row in bank:
        answers = [parse_answer(candidate["text"]) for candidate in row["candidates"]]
        counts = Counter(answer for answer in answers if answer is not None)

        if not counts:
            choices.append(0)
            continue

        largest_group = max(counts.values())
        winning_answers = {
            answer for answer, count in counts.items() if count == largest_group
        }
        choice = next(
            index for index, answer in enumerate(answers) if answer in winning_answers
        )
        choices.append(choice)

    return choices
