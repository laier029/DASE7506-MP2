"""Evaluate precision-first negative gates before answer-group ranking."""

from collections import defaultdict
from pathlib import Path
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from answer_step_consistency import answer_step_consistency
from arithmetic_consistency import arithmetic_counts, broad_arithmetic_counts
from common import load_inputs, parse_answer
from group_ranker_probe import candidate_labels, oof_model_k
from parseable_group_verifier import select_parseable_groups
from structure_text_fusion_verifier import structure_text_scores


def negative_signals(bank):
    names = ("unparseable", "narrow_arithmetic", "broad_only", "step_mismatch")
    signals = np.zeros((len(bank), 4, len(names)), dtype=bool)
    for row_index, row in enumerate(bank):
        for candidate_index, candidate in enumerate(row["candidates"]):
            text = candidate["text"]
            narrow_bad = arithmetic_counts(text)[1]
            broad_bad = broad_arithmetic_counts(text)[1]
            signals[row_index, candidate_index] = (
                parse_answer(text) is None,
                narrow_bad > 0,
                broad_bad > 0 and narrow_bad == 0,
                answer_step_consistency(text) is False,
            )
    return names, signals


def rule_statistics(name, gate, labels):
    flagged = int(gate.sum())
    wrong = ~labels
    caught_wrong = int((gate & wrong).sum())
    removed_correct = int((gate & labels).sum())
    precision = caught_wrong / flagged if flagged else 1.0
    wrong_recall = caught_wrong / wrong.sum()
    correct_removal = removed_correct / labels.sum()
    correct_question_kills = int(((labels & ~gate).sum(axis=1) == 0).sum())
    return (
        f"rule={name} flagged={flagged} precision={100 * precision:.2f}% "
        f"wrong_recall={100 * wrong_recall:.2f}% "
        f"correct_removal={100 * correct_removal:.3f}% "
        f"correct_question_kills={correct_question_kills}"
    )


def cascade_select(bank, scores, gate, *, rescue):
    selected = []
    for row, row_scores, row_gate in zip(bank, scores, gate):
        groups = defaultdict(list)
        for index, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate["text"])
            if answer is not None:
                groups[answer].append(index)
        survivors = {
            answer: [index for index in indices if not row_gate[index]]
            for answer, indices in groups.items()
        }
        if rescue == "group":
            for answer, indices in groups.items():
                if not survivors[answer]:
                    survivors[answer] = [
                        max(indices, key=lambda index: (row_scores[index], -index))
                    ]
        elif rescue == "global" and not any(survivors.values()):
            indices = [index for values in groups.values() for index in values]
            restored = max(indices, key=lambda index: (row_scores[index], -index))
            answer = parse_answer(row["candidates"][restored]["text"])
            survivors[answer] = [restored]
        elif rescue != "global":
            raise ValueError(f"Unknown rescue mode: {rescue}")

        best = None
        for indices in survivors.values():
            if not indices:
                continue
            values = row_scores[indices]
            candidate = indices[int(values.argmax())]
            group_score = float(values.max() + 0.3 * values.mean())
            item = (group_score, -candidate, candidate)
            if best is None or item > best:
                best = item
        if best is None:
            raise ValueError("Cascade removed every parseable candidate")
        selected.append(best[2])
    return np.asarray(selected)


def selection_accuracy(labels, selected):
    return labels[np.arange(len(labels)), selected].mean()


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    names, train_signals = negative_signals(train_bank)
    _, validation_signals = negative_signals(validation_bank)

    folds = np.arange(len(train_bank)) % 5
    train_scores = oof_model_k(
        train_bank, train_arrays, train_labels, folds
    )
    train_scores -= train_scores.mean(axis=1, keepdims=True)
    train_scores /= float(train_scores.std())
    validation_scores = structure_text_scores(
        validation_bank,
        validation_arrays,
        CODE / "runs/model_m_structure_text",
    )
    base_train = np.asarray(
        select_parseable_groups(train_bank, train_scores, 0.3)
    )
    base_validation = np.asarray(
        select_parseable_groups(validation_bank, validation_scores, 0.3)
    )
    print(
        f"base_oof={100 * selection_accuracy(train_labels, base_train):.4f} "
        f"base_validation={100 * selection_accuracy(validation_labels, base_validation):.4f}",
        flush=True,
    )

    rules = {
        "unparseable": lambda values: values[:, :, 0],
        "narrow": lambda values: values[:, :, 1],
        "broad_only": lambda values: values[:, :, 2],
        "step_mismatch": lambda values: values[:, :, 3],
        "narrow_or_step": lambda values: values[:, :, 1] | values[:, :, 3],
        "any_reasoning_flag": lambda values: values[:, :, 1:].any(axis=2),
        "at_least_two_reasoning_flags": lambda values: values[:, :, 1:].sum(axis=2) >= 2,
        "narrow_and_step": lambda values: values[:, :, 1] & values[:, :, 3],
    }
    for rule_name, rule in rules.items():
        train_gate = rule(train_signals)
        validation_gate = rule(validation_signals)
        print("train", rule_statistics(rule_name, train_gate, train_labels), flush=True)
        print(
            "validation",
            rule_statistics(rule_name, validation_gate, validation_labels),
            flush=True,
        )
        if rule_name == "unparseable":
            continue
        for rescue in ("global", "group"):
            train_selected = cascade_select(
                train_bank, train_scores, train_gate, rescue=rescue
            )
            validation_selected = cascade_select(
                validation_bank,
                validation_scores,
                validation_gate,
                rescue=rescue,
            )
            print(
                f"cascade={rule_name} rescue={rescue} "
                f"oof={100 * selection_accuracy(train_labels, train_selected):.4f} "
                f"validation={100 * selection_accuracy(validation_labels, validation_selected):.4f} "
                f"oof_changes={(train_selected != base_train).sum()} "
                f"validation_changes={(validation_selected != base_validation).sum()}",
                flush=True,
            )


if __name__ == "__main__":
    main()
