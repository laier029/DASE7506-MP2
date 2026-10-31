"""Probe sparse question-word by candidate-state interaction features."""

from collections import Counter
from pathlib import Path
import re
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from calculation_coverage import calculation_occurrence_coverage, calculation_unique_coverage
from cascade_stability_probe import balanced_random_folds
from common import load_inputs, parse_answer
from coverage_cascade_verifier import coverage_adjusted_scores
from group_ranker_probe import candidate_labels
from model_t_probe_utils import FastFinalSelector, SEEDS, load_or_build_model_t_oof
from occurrence_plausibility_verifier import occurrence_adjusted_scores
from structure_text_fusion_verifier import answer_structures, structure_text_scores
from target_alignment import terminal_target_recall
from target_omitted_factor_verifier import target_adjusted_scores
from target_operation_probe import last_result_operation


WORD = re.compile(r"[a-z]+")


def state_documents(bank):
    unique = calculation_unique_coverage(bank)
    occurrence = calculation_occurrence_coverage(bank)
    target = terminal_target_recall(bank)
    documents = []
    for row_index, row in enumerate(bank):
        answers = [parse_answer(candidate["text"]) for candidate in row["candidates"]]
        counts = Counter(answer for answer in answers if answer is not None)
        question_words = WORD.findall(row["question"].lower())
        question_bigrams = [
            left + "_" + right
            for left, right in zip(question_words, question_words[1:])
        ]
        terms = question_words + question_bigrams
        for candidate_index, candidate in enumerate(row["candidates"]):
            add, subtract, multiply, divide, multi_add = last_result_operation(candidate["text"])
            operations = [
                name
                for name, active in (
                    ("add", add),
                    ("sub", subtract),
                    ("mul", multiply),
                    ("div", divide),
                    ("multiadd", multi_add),
                )
                if active
            ] or ["none"]
            group_size = counts.get(answers[candidate_index], 0)
            states = [
                *(f"op_{operation}" for operation in operations),
                f"group_{group_size}",
                "unique_high" if unique[row_index, candidate_index] >= unique[row_index].max() else "unique_low",
                "occ_high" if occurrence[row_index, candidate_index] >= occurrence[row_index].max() else "occ_low",
                "target_yes" if target[row_index, candidate_index] > 0 else "target_no",
            ]
            documents.append(
                " ".join(
                    f"{state}__{term}" for state in states for term in terms
                )
            )
    return documents


def standardize_rows(values):
    return (values - values.mean(axis=1, keepdims=True)) / (
        values.std(axis=1, keepdims=True) + 1e-6
    )


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    train_structures = answer_structures(train_bank)
    validation_structures = answer_structures(validation_bank)
    score_sets = load_or_build_model_t_oof(train_bank, train_arrays, train_labels)
    train_selector = FastFinalSelector(train_bank)
    validation_selector = FastFinalSelector(validation_bank)
    validation_scores = structure_text_scores(
        validation_bank, validation_arrays, CODE / "runs/model_t_final_cascade"
    )
    validation_scores = coverage_adjusted_scores(validation_bank, validation_scores)
    validation_scores = occurrence_adjusted_scores(validation_bank, validation_scores)
    validation_scores = target_adjusted_scores(validation_bank, validation_scores)
    train_base = [train_selector.correct_count(train_labels, scores) for scores in score_sets]
    validation_base = validation_selector.correct_count(validation_labels, validation_scores)

    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[^ ]+",
        lowercase=False,
        min_df=3,
        max_features=30000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    train_x = vectorizer.fit_transform(state_documents(train_bank))
    validation_x = vectorizer.transform(state_documents(validation_bank))
    active_structures = ("3-1", "2-2", "2-1-1", "1-1-1-1")
    weights = np.round(np.arange(-0.3, 0.301, 0.025), 3)
    for c_value in (0.01, 0.1):
        interaction_sets = []
        validation_interaction = np.zeros((len(validation_bank), 4), dtype=np.float32)
        scales = {}
        for seed_index, seed in enumerate(SEEDS):
            folds = (
                np.arange(len(train_bank)) % 5
                if seed == 7506
                else balanced_random_folds(len(train_bank), seed)
            )
            interaction = np.zeros((len(train_bank), 4), dtype=np.float32)
            for structure in active_structures:
                active = train_structures == structure
                for fold in range(5):
                    fit_rows = active & (folds != fold)
                    held_rows = active & (folds == fold)
                    fit = np.repeat(fit_rows, 4)
                    held = np.repeat(held_rows, 4)
                    classifier = LinearSVC(
                        C=c_value, dual="auto", random_state=7506
                    ).fit(train_x[fit], train_labels.reshape(-1)[fit])
                    interaction[held_rows] = classifier.decision_function(
                        train_x[held]
                    ).reshape(-1, 4)
                if seed_index == 0:
                    scale = float(interaction[active].std())
                    scales[structure] = scale if scale > 1e-6 else 1.0
                    fit = np.repeat(active, 4)
                    classifier = LinearSVC(
                        C=c_value, dual="auto", random_state=7506
                    ).fit(train_x[fit], train_labels.reshape(-1)[fit])
                    validation_rows = validation_structures == structure
                    held = np.repeat(validation_rows, 4)
                    validation_interaction[validation_rows] = classifier.decision_function(
                        validation_x[held]
                    ).reshape(-1, 4)
            for structure in active_structures:
                rows = train_structures == structure
                interaction[rows] /= scales.get(structure, float(interaction[rows].std()) or 1.0)
                if seed_index == 0:
                    validation_rows = validation_structures == structure
                    validation_interaction[validation_rows] /= scales[structure]
            interaction_sets.append(standardize_rows(interaction))
        validation_interaction = standardize_rows(validation_interaction)

        structure_weights = {}
        for structure in active_structures:
            rows = train_structures == structure
            trials = []
            for weight in weights:
                adjustment = np.zeros_like(interaction_sets[0])
                adjustment[rows] = weight * interaction_sets[0][rows]
                count = train_selector.correct_count(
                    train_labels, score_sets[0] + adjustment
                )
                trials.append((count, -abs(float(weight)), float(weight)))
            _, _, structure_weights[structure] = max(trials)
        train_deltas = []
        for scores, interaction, base in zip(score_sets, interaction_sets, train_base):
            adjustment = np.zeros_like(interaction)
            for structure, weight in structure_weights.items():
                rows = train_structures == structure
                adjustment[rows] = weight * interaction[rows]
            train_deltas.append(
                train_selector.correct_count(train_labels, scores + adjustment) - base
            )
        validation_adjustment = np.zeros_like(validation_interaction)
        for structure, weight in structure_weights.items():
            rows = validation_structures == structure
            validation_adjustment[rows] = weight * validation_interaction[rows]
        validation_delta = validation_selector.correct_count(
            validation_labels, validation_scores + validation_adjustment
        ) - validation_base
        print(
            f"c={c_value:g} weights={structure_weights} "
            f"train_deltas={train_deltas} validation_delta={validation_delta:+d}",
            flush=True,
        )


if __name__ == "__main__":
    main()
