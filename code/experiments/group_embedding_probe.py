"""Probe learned answer-group representations from supplied embeddings."""

from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from sklearn.svm import LinearSVC


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from cascade_stability_probe import balanced_random_folds
from common import load_inputs, parse_answer
from dimension_consistency_probe import choices, correct
from group_ranker_probe import candidate_labels
from model_t_probe_utils import SEEDS, load_or_build_model_t_oof
from stable_signal_features import stable_signal_adjusted_scores
from stable_signal_verifier import stable_signal_scores


def group_data(bank, arrays, labels=None, representation="mean_std"):
    embeddings = arrays["embeddings"].astype(np.float32)
    embeddings -= embeddings.mean(axis=1, keepdims=True)
    features = []
    questions = []
    members = []
    targets = []
    for question_index, row in enumerate(bank):
        groups = defaultdict(list)
        for candidate_index, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate["text"])
            if answer is not None:
                groups[answer].append(candidate_index)
        for indices in groups.values():
            values = embeddings[question_index, indices]
            parts = [values.mean(axis=0)]
            if representation == "mean_std":
                parts.append(values.std(axis=0))
            features.append(np.concatenate(parts))
            questions.append(question_index)
            members.append(indices)
            if labels is not None:
                targets.append(float(labels[question_index, indices].any()))
    output = (
        np.asarray(features, dtype=np.float32),
        np.asarray(questions, dtype=np.int64),
        members,
    )
    if labels is not None:
        return (*output, np.asarray(targets, dtype=np.int64))
    return output


def candidate_adjustment(group_scores, questions, members, n_questions):
    output = np.zeros((n_questions, 4), dtype=np.float32)
    centered = group_scores.copy()
    for question in np.unique(questions):
        mask = questions == question
        centered[mask] -= centered[mask].mean()
    scale = float(centered.std()) or 1.0
    centered /= scale
    for score, question, indices in zip(centered, questions, members):
        output[question, indices] = score
    return output


def fit_predict(train_x, train_y, score_x, c_value):
    classifier = LinearSVC(C=c_value, dual="auto", random_state=7506)
    classifier.fit(train_x, train_y)
    return classifier.decision_function(score_x)


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    base_score_sets = [
        stable_signal_adjusted_scores(train_bank, scores)
        for scores in load_or_build_model_t_oof(train_bank, train_arrays, train_labels)
    ]
    validation_base_scores = stable_signal_scores(
        validation_bank, validation_arrays, CODE / "runs/model_x_full_cascade"
    )
    empty_train = [set() for _ in train_bank]
    empty_validation = [set() for _ in validation_bank]
    base_train_choices = [choices(train_bank, scores, empty_train) for scores in base_score_sets]
    base_train = [correct(train_labels, selected) for selected in base_train_choices]
    base_validation_choices = choices(
        validation_bank, validation_base_scores, empty_validation
    )
    base_validation = correct(validation_labels, base_validation_choices)

    for representation in ("mean", "mean_std"):
        train_x, train_questions, train_members, train_y = group_data(
            train_bank, train_arrays, train_labels, representation
        )
        validation_x, validation_questions, validation_members = group_data(
            validation_bank, validation_arrays, None, representation
        )
        for c_value in (0.001, 0.01):
            adjustment_sets = []
            validation_adjustment = None
            for seed_index, seed in enumerate(SEEDS):
                folds = (
                    np.arange(len(train_bank)) % 5
                    if seed == 7506
                    else balanced_random_folds(len(train_bank), seed)
                )
                decisions = np.zeros(len(train_x), dtype=np.float32)
                for fold in range(5):
                    fit = folds[train_questions] != fold
                    held = ~fit
                    decisions[held] = fit_predict(
                        train_x[fit], train_y[fit], train_x[held], c_value
                    )
                adjustment_sets.append(
                    candidate_adjustment(
                        decisions, train_questions, train_members, len(train_bank)
                    )
                )
                if seed_index == 0:
                    validation_decisions = fit_predict(
                        train_x, train_y, validation_x, c_value
                    )
                    validation_adjustment = candidate_adjustment(
                        validation_decisions,
                        validation_questions,
                        validation_members,
                        len(validation_bank),
                    )

            weights = (-0.20, -0.10, -0.05, -0.025, 0.0, 0.025, 0.05, 0.10, 0.20)
            trials = []
            for weight in weights:
                selected = choices(
                    train_bank,
                    base_score_sets[0] + weight * adjustment_sets[0],
                    empty_train,
                )
                trials.append(
                    (correct(train_labels, selected), -abs(weight), weight)
                )
            _, _, weight = max(trials)
            deltas = []
            changes = []
            for scores, adjustment, base, base_selected in zip(
                base_score_sets,
                adjustment_sets,
                base_train,
                base_train_choices,
            ):
                selected = choices(
                    train_bank, scores + weight * adjustment, empty_train
                )
                deltas.append(correct(train_labels, selected) - base)
                changes.append(int(np.sum(selected != base_selected)))
            proposed = choices(
                validation_bank,
                validation_base_scores + weight * validation_adjustment,
                empty_validation,
            )
            print(
                f"representation={representation} c={c_value:g} weight={weight:+.3f} "
                f"oof_deltas={deltas} oof_changes={changes} "
                f"validation_delta={correct(validation_labels, proposed) - base_validation:+d} "
                f"validation_changes={np.flatnonzero(proposed != base_validation_choices).tolist()}",
                flush=True,
            )


if __name__ == "__main__":
    main()
