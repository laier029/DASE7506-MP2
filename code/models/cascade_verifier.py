"""Model N: precision-first arithmetic group rejection, then Model M ranking."""

from collections import defaultdict
from pathlib import Path
import re

import numpy as np

from arithmetic_consistency import (
    arithmetic_counts,
    broad_arithmetic_mismatches,
)
from common import parse_answer
from structure_text_fusion_verifier import structure_text_scores


_CORRECTION_CUE = re.compile(
    r"(?i)\b(?:wait|re-?check|mistake|miscalculation|correction|"
    r"different approach|another approach|"
    r"let(?:'s| us) (?:recalculate|check|try)|"
    r"i (?:made an error|need to correct))\b"
)


def _unrecovered_broad_only_mismatches(text):
    """Return broad-only errors after the trace's last explicit repair cue."""
    matches = list(_CORRECTION_CUE.finditer(text))
    checked_text = text[matches[-1].end() :] if matches else text
    if arithmetic_counts(checked_text)[1] > 0:
        return set()
    return set(broad_arithmetic_mismatches(checked_text))


def unanimous_broad_error_groups(bank):
    """Identify answer groups whose traces unanimously repeat one error."""
    rejected = []
    for row in bank:
        groups = defaultdict(list)
        mismatch_sets = []
        for candidate_index, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate["text"])
            if answer is not None:
                groups[answer].append(candidate_index)
            mismatch_sets.append(
                _unrecovered_broad_only_mismatches(candidate["text"])
            )

        row_rejected = set()
        for answer, indices in groups.items():
            if len(indices) < 2:
                continue
            shared = set.intersection(*(mismatch_sets[index] for index in indices))
            if shared:
                row_rejected.add(answer)
        rejected.append(row_rejected)
    return rejected


def select_after_negative_filter(
    bank, scores, group_mean_weight, additional_rejected=None
):
    """Reject high-confidence bad groups, with a whole-question fallback."""
    rejected_rows = unanimous_broad_error_groups(bank)
    if additional_rejected is not None:
        if len(additional_rejected) != len(bank):
            raise ValueError("Additional rejection rows must align with the bank.")
        rejected_rows = [
            base | set(extra)
            for base, extra in zip(rejected_rows, additional_rejected)
        ]
    selected = []
    for row, row_scores, rejected in zip(bank, scores, rejected_rows):
        groups = defaultdict(list)
        for candidate_index, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate["text"])
            if answer is not None:
                groups[answer].append(candidate_index)
        if not groups:
            raise ValueError("Question has no candidate with a parseable answer.")

        survivors = {
            answer: indices
            for answer, indices in groups.items()
            if answer not in rejected
        }
        # The negative stage must never leave the scoring stage empty.
        if not survivors:
            survivors = groups

        best_group_score = None
        best_candidate = None
        for indices in survivors.values():
            values = row_scores[indices]
            candidate = indices[int(values.argmax())]
            group_score = float(
                values.max() + group_mean_weight * values.mean()
            )
            if best_group_score is None or group_score > best_group_score:
                best_group_score = group_score
                best_candidate = candidate
        selected.append(best_candidate)
    return selected


def predict(bank, arrays, checkpoint):
    """Run the safe negative group filter before Model M's score ranking."""
    checkpoint = Path(checkpoint)
    scores = structure_text_scores(bank, arrays, checkpoint)
    with np.load(checkpoint / "model.npz", allow_pickle=False) as model:
        group_mean_weight = float(model["group_mean_weight"].item())
    return select_after_negative_filter(bank, scores, group_mean_weight)
