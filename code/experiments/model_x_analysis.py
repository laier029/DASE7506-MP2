"""Produce validation-only ablation, structure and confidence diagnostics."""

from collections import defaultdict
import argparse
import json
from pathlib import Path
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from cascade_verifier import unanimous_broad_error_groups
from common import load_inputs, parse_answer
from final_declaration_consistency import final_declaration_mismatch_groups
from group_ranker_probe import candidate_labels
from numeric_plausibility import implausible_answer_groups
from stable_signal_verifier import stable_signal_scores
from structure_text_fusion_verifier import answer_structures


STAGES = (
    ("m", "runs/model_m_structure_text/validation-predictions-current.json"),
    ("n", "runs/model_n_cascade/validation-predictions-current.json"),
    ("o", "runs/model_o_coverage_cascade/validation-predictions-current.json"),
    ("p", "runs/model_p_plausibility_coverage/validation-predictions-current.json"),
    ("q", "runs/model_q_occurrence_plausibility/validation-predictions-current.json"),
    ("r", "runs/model_r_omitted_factor/validation-predictions-current.json"),
    ("s", "runs/model_s_target_alignment/validation-predictions-current.json"),
    ("t", "runs/model_t_final_cascade/validation-predictions-current.json"),
    ("u", "runs/model_u_stable_signal/validation-predictions-current.json"),
    ("x", "runs/model_x_full_cascade/submission-validation-predictions.json"),
)


def load_choices(path):
    return np.asarray(json.loads((CODE / path).read_text())["selected"])


def answer_groups(row):
    groups = defaultdict(list)
    for candidate_index, candidate in enumerate(row["candidates"]):
        answer = parse_answer(candidate["text"])
        if answer is not None:
            groups[answer].append(candidate_index)
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bank, arrays = load_inputs("validation")
    labels = candidate_labels("validation", bank).astype(bool)
    stages = {name: load_choices(path) for name, path in STAGES}
    structures = answer_structures(bank)
    final = stages["x"]
    report = {
        "protocol_note": "validation diagnostics only; test references sealed",
        "stage_correct": {
            name: int(labels[np.arange(len(labels)), choices].sum())
            for name, choices in stages.items()
        },
        "stage_transitions": {},
        "accuracy_by_structure": {},
    }
    for (left_name, _), (right_name, _) in zip(STAGES, STAGES[1:]):
        left = stages[left_name]
        right = stages[right_name]
        changed = left != right
        left_correct = labels[np.arange(len(labels)), left]
        right_correct = labels[np.arange(len(labels)), right]
        answer_changed = sum(
            parse_answer(row["candidates"][left_choice]["text"])
            != parse_answer(row["candidates"][right_choice]["text"])
            for row, left_choice, right_choice in zip(bank, left, right)
        )
        report["stage_transitions"][f"{left_name}_to_{right_name}"] = {
            "candidate_index_changes": int(changed.sum()),
            "answer_group_changes": int(answer_changed),
            "fixes": int((~left_correct & right_correct).sum()),
            "harms": int((left_correct & ~right_correct).sum()),
            "net": int(right_correct.sum() - left_correct.sum()),
        }
    for structure in sorted(set(structures)):
        rows = structures == structure
        correct = labels[np.arange(len(labels))[rows], final[rows]]
        report["accuracy_by_structure"][structure] = {
            "correct": int(correct.sum()),
            "n": int(rows.sum()),
            "accuracy": float(correct.mean()),
        }

    scores = stable_signal_scores(
        bank, arrays, CODE / "runs/model_x_full_cascade"
    )
    arithmetic = unanimous_broad_error_groups(bank)
    numeric = implausible_answer_groups(bank)
    declaration = final_declaration_mismatch_groups(bank)
    margins = []
    for row_index, (row, row_scores, choice) in enumerate(zip(bank, scores, final)):
        groups = answer_groups(row)
        rejected = arithmetic[row_index] | numeric[row_index] | declaration[row_index]
        survivors = {
            answer: indices
            for answer, indices in groups.items()
            if answer not in rejected
        }
        if not survivors:
            survivors = groups
        group_scores = {
            answer: float(
                row_scores[indices].max() + 0.3 * row_scores[indices].mean()
            )
            for answer, indices in survivors.items()
        }
        selected_answer = parse_answer(row["candidates"][choice]["text"])
        if selected_answer not in group_scores or len(group_scores) == 1:
            margins.append(float("inf"))
            continue
        alternatives = [
            value
            for answer, value in group_scores.items()
            if answer != selected_answer
        ]
        margins.append(group_scores[selected_answer] - max(alternatives))
    margins = np.asarray(margins)
    correct = labels[np.arange(len(labels)), final]
    bins = (
        ("negative", -np.inf, 0.0),
        ("0_to_0.25", 0.0, 0.25),
        ("0.25_to_0.5", 0.25, 0.5),
        ("0.5_to_1", 0.5, 1.0),
        ("1_to_2", 1.0, 2.0),
        ("2_plus", 2.0, np.inf),
    )
    report["confidence_by_final_group_margin"] = {}
    for name, lower, upper in bins:
        rows = (margins >= lower) & (margins < upper)
        report["confidence_by_final_group_margin"][name] = {
            "correct": int(correct[rows].sum()),
            "n": int(rows.sum()),
            "accuracy": float(correct[rows].mean()) if rows.any() else None,
        }
    report["filter_trigger_questions"] = {
        "arithmetic": int(sum(bool(values) for values in arithmetic)),
        "numeric": int(sum(bool(values) for values in numeric)),
        "final_declaration": int(sum(bool(values) for values in declaration)),
    }
    report["remaining_error_indices"] = np.flatnonzero(~correct).tolist()
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
