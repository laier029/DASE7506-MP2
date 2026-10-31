"""Probe question-text priors for majority versus minority answer groups."""

from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from cascade_stability_probe import balanced_random_folds
from common import load_inputs, parse_answer
from coverage_cascade_verifier import coverage_adjusted_scores
from group_ranker_probe import candidate_labels
from model_t_probe_utils import FastFinalSelector, SEEDS, load_or_build_model_t_oof
from occurrence_plausibility_verifier import occurrence_adjusted_scores
from structure_text_fusion_verifier import answer_structures, structure_text_scores
from target_omitted_factor_verifier import target_adjusted_scores


def group_sizes_and_target(bank, labels):
    sizes = np.zeros((len(bank), 4), dtype=np.float32)
    majority_correct = np.zeros(len(bank), dtype=np.int64)
    for row_index, row in enumerate(bank):
        groups = defaultdict(list)
        for candidate_index, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate["text"])
            if answer is not None:
                groups[answer].append(candidate_index)
        majority = max(map(len, groups.values()))
        for indices in groups.values():
            for candidate_index in indices:
                sizes[row_index, candidate_index] = len(indices)
            if labels[row_index, indices[0]]:
                majority_correct[row_index] = int(len(indices) == majority)
    return sizes, majority_correct


def make_adjustment(decisions, sizes, structures):
    output = np.zeros_like(sizes, dtype=np.float32)
    for structure in ("3-1", "2-1-1"):
        rows = structures == structure
        if not rows.any():
            continue
        majority = sizes[rows].max(axis=1, keepdims=True)
        signs = np.where(sizes[rows] == majority, 1.0, -1.0)
        output[rows] = decisions[rows, None] * signs
    return output


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    train_structures = answer_structures(train_bank)
    validation_structures = answer_structures(validation_bank)
    train_sizes, train_target = group_sizes_and_target(train_bank, train_labels)
    validation_sizes, _ = group_sizes_and_target(validation_bank, validation_labels)
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
    train_questions = [row["question"] for row in train_bank]
    validation_questions = [row["question"] for row in validation_bank]
    weights = np.round(np.arange(0.0, 0.501, 0.025), 3)

    for analyzer, ngram, min_df in (("word", (1, 2), 3), ("char", (3, 5), 5)):
        vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram,
            min_df=min_df,
            max_features=25000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        train_x = vectorizer.fit_transform(train_questions)
        validation_x = vectorizer.transform(validation_questions)
        for c_value in (0.01, 0.1):
            for class_weight in (None, "balanced"):
                adjustment_sets = []
                full_validation_decisions = np.zeros(len(validation_bank), dtype=np.float32)
                fixed_scale = None
                for seed_index, seed in enumerate(SEEDS):
                    folds = (
                        np.arange(len(train_bank)) % 5
                        if seed == 7506
                        else balanced_random_folds(len(train_bank), seed)
                    )
                    decisions = np.zeros(len(train_bank), dtype=np.float32)
                    for structure in ("3-1", "2-1-1"):
                        active = train_structures == structure
                        for fold in range(5):
                            fit = active & (folds != fold)
                            held = active & (folds == fold)
                            classifier = LinearSVC(
                                C=c_value,
                                dual="auto",
                                class_weight=class_weight,
                                random_state=7506,
                            ).fit(train_x[fit], train_target[fit])
                            decisions[held] = classifier.decision_function(train_x[held])
                    scale = float(decisions[train_structures != "4"].std())
                    scale = scale if scale > 1e-6 else 1.0
                    if seed_index == 0:
                        fixed_scale = scale
                    adjustment_sets.append(
                        make_adjustment(decisions / scale, train_sizes, train_structures)
                    )

                for structure in ("3-1", "2-1-1"):
                    active = train_structures == structure
                    classifier = LinearSVC(
                        C=c_value,
                        dual="auto",
                        class_weight=class_weight,
                        random_state=7506,
                    ).fit(train_x[active], train_target[active])
                    validation_rows = validation_structures == structure
                    full_validation_decisions[validation_rows] = classifier.decision_function(
                        validation_x[validation_rows]
                    )
                validation_adjustment = make_adjustment(
                    full_validation_decisions / fixed_scale,
                    validation_sizes,
                    validation_structures,
                )
                trials = []
                for weight in weights:
                    count = train_selector.correct_count(
                        train_labels, score_sets[0] + weight * adjustment_sets[0]
                    )
                    trials.append((count, -float(weight), float(weight)))
                _, _, weight = max(trials)
                train_deltas = [
                    train_selector.correct_count(
                        train_labels, scores + weight * adjustment
                    )
                    - base
                    for scores, adjustment, base in zip(score_sets, adjustment_sets, train_base)
                ]
                validation_delta = validation_selector.correct_count(
                    validation_labels,
                    validation_scores + weight * validation_adjustment,
                ) - validation_base
                print(
                    f"analyzer={analyzer} c={c_value:g} class_weight={class_weight} "
                    f"weight={weight:.3f} train_deltas={train_deltas} "
                    f"validation_delta={validation_delta:+d}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
