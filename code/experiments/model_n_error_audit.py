"""Summarize the errors that remain after Model N's negative filter."""

from collections import Counter, defaultdict
import argparse
from pathlib import Path
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from cascade_verifier import (
    select_after_negative_filter,
    unanimous_broad_error_groups,
)
from coverage_cascade_verifier import coverage_adjusted_scores
from numeric_plausibility import (
    fractional_answer_groups,
    implausible_answer_groups,
)
from occurrence_plausibility_verifier import occurrence_adjusted_scores
from omitted_factor import select_with_omitted_factor_switch
from common import load_inputs, parse_answer
from group_ranker_probe import candidate_labels
from structure_text_fusion_verifier import answer_structures, structure_text_scores
from target_omitted_factor_verifier import target_adjusted_scores


def answer_groups(row):
    groups = defaultdict(list)
    for candidate_index, candidate in enumerate(row["candidates"]):
        answer = parse_answer(candidate["text"])
        if answer is not None:
            groups[answer].append(candidate_index)
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--details", action="store_true")
    parser.add_argument(
        "--model", choices=("n", "o", "p", "q", "r", "s", "t"), default="t"
    )
    args = parser.parse_args()
    bank, arrays = load_inputs("validation")
    labels = candidate_labels("validation", bank).astype(bool)
    checkpoint_names = {
        "n": "runs/model_n_cascade",
        "o": "runs/model_o_coverage_cascade",
        "p": "runs/model_p_plausibility_coverage",
        "q": "runs/model_q_occurrence_plausibility",
        "r": "runs/model_r_omitted_factor",
        "s": "runs/model_s_target_alignment",
        "t": "runs/model_t_final_cascade",
    }
    checkpoint = CODE / checkpoint_names[args.model]
    scores = structure_text_scores(bank, arrays, checkpoint)
    if args.model in {"o", "p", "q", "r", "s", "t"}:
        scores = coverage_adjusted_scores(bank, scores)
    if args.model in {"q", "r", "s", "t"}:
        scores = occurrence_adjusted_scores(bank, scores)
    if args.model in {"s", "t"}:
        scores = target_adjusted_scores(bank, scores)
    additional_rejected = (
        implausible_answer_groups(bank)
        if args.model == "t"
        else fractional_answer_groups(bank)
        if args.model in {"p", "q", "r", "s"}
        else None
    )
    if args.model in {"r", "s", "t"}:
        selected = np.asarray(
            select_with_omitted_factor_switch(
                bank, scores, 0.3, additional_rejected
            )
        )
    else:
        selected = np.asarray(
            select_after_negative_filter(
                bank, scores, 0.3, additional_rejected=additional_rejected
            )
        )
    structures = answer_structures(bank)
    rejected = unanimous_broad_error_groups(bank)

    print("accuracy by answer structure")
    for structure in sorted(set(structures)):
        rows = structures == structure
        correct = labels[np.arange(len(labels))[rows], selected[rows]]
        print(
            f"  structure={structure} correct={int(correct.sum())} "
            f"n={int(rows.sum())} accuracy={100 * correct.mean():.2f}%"
        )

    transitions = Counter()
    correct_rank = Counter()
    margins = []
    details = []
    for row_index, (row, choice) in enumerate(zip(bank, selected)):
        if labels[row_index, choice]:
            continue
        groups = answer_groups(row)
        correct_answer = next(
            answer
            for answer, indices in groups.items()
            if labels[row_index, indices[0]]
        )
        selected_answer = parse_answer(row["candidates"][choice]["text"])
        transitions[(len(groups[correct_answer]), len(groups[selected_answer]))] += 1

        scored_groups = []
        for answer, indices in groups.items():
            if answer in rejected[row_index] and len(rejected[row_index]) < len(groups):
                continue
            values = scores[row_index, indices]
            scored_groups.append(
                (float(values.max() + 0.3 * values.mean()), answer)
            )
        scored_groups.sort(reverse=True)
        rank = next(
            rank
            for rank, (_, answer) in enumerate(scored_groups, start=1)
            if answer == correct_answer
        )
        correct_rank[rank] += 1
        selected_score = next(
            value for value, answer in scored_groups if answer == selected_answer
        )
        correct_score = next(
            value for value, answer in scored_groups if answer == correct_answer
        )
        margins.append(selected_score - correct_score)
        details.append(
            (
                row_index,
                structures[row_index],
                correct_answer,
                selected_answer,
                selected_score - correct_score,
                " ".join(row["question"].split()),
            )
        )

    print("remaining errors by correct-group-size -> selected-group-size")
    for transition, count in sorted(transitions.items()):
        print(f"  {transition[0]}->{transition[1]}: {count}")
    print("rank of correct answer group among Model N scores")
    for rank, count in sorted(correct_rank.items()):
        print(f"  rank={rank}: {count}")
    print(
        "wrong-selection margin quantiles "
        + " ".join(
            f"q{int(q * 100):02d}={np.quantile(margins, q):.4f}"
            for q in (0.0, 0.25, 0.5, 0.75, 1.0)
        )
    )
    if args.details:
        print("remaining error details")
        for row_index, structure, correct, chosen, margin, question in details:
            print(
                f"  q={row_index} structure={structure} gold={correct} "
                f"chosen={chosen} margin={margin:.4f} question={question}"
            )


if __name__ == "__main__":
    main()
