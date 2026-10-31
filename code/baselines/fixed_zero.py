"""Baseline A: always select the first candidate."""


def predict(bank, arrays, checkpoint):
    """Return candidate 0 for every question."""
    return [0] * len(bank)
