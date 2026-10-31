"""Print selection changes caused by the broad-only arithmetic hard gate."""

from pathlib import Path
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from cascade_negative_probe import cascade_select, negative_signals
from cascade_stability_probe import correction_aware_gates
from common import load_inputs, parse_answer
from group_ranker_probe import candidate_labels
from parseable_group_verifier import select_parseable_groups
from structure_text_fusion_verifier import structure_text_scores


def compact(text, limit=220):
    return " ".join(text.split())[-limit:]


def main():
    bank, arrays = load_inputs("validation")
    labels = candidate_labels("validation", bank).astype(bool)
    _, signals = negative_signals(bank)
    _, gate = correction_aware_gates(bank)
    scores = structure_text_scores(
        bank, arrays, CODE / "runs/model_m_structure_text"
    )
    base = np.asarray(select_parseable_groups(bank, scores, 0.3))
    hard = cascade_select(bank, scores, gate, rescue="global")

    for row_index in np.flatnonzero(base != hard):
        row = bank[row_index]
        print(
            f"question={row_index} base={base[row_index]} "
            f"base_correct={bool(labels[row_index, base[row_index]])} "
            f"hard={hard[row_index]} "
            f"hard_correct={bool(labels[row_index, hard[row_index]])}"
        )
        for candidate_index, candidate in enumerate(row["candidates"]):
            print(
                f"  c={candidate_index} answer={parse_answer(candidate['text'])} "
                f"correct={bool(labels[row_index, candidate_index])} "
                f"gate={bool(gate[row_index, candidate_index])} "
                f"score={scores[row_index, candidate_index]:.4f} "
                f"tail={compact(candidate['text'])}"
            )

    print("correct candidates flagged by repaired broad-only gate")
    for row_index, candidate_index in np.argwhere(gate & labels):
        candidate = bank[row_index]["candidates"][candidate_index]
        killed = not bool((labels[row_index] & ~gate[row_index]).any())
        print(
            f"  question={row_index} candidate={candidate_index} killed={killed} "
            f"answer={parse_answer(candidate['text'])} "
            f"tail={compact(candidate['text'], 420)}"
        )


if __name__ == "__main__":
    main()
