"""Cross-fitted selective router over Model Z and two new verifier directions.

Model Z remains the default.  The semantic, ledger and combined answer-group
experts must produce a majority alternative before the router can switch.  The
router is trained only on train OOF expert outputs and is itself cross-fitted by
question.  Its configuration and probability threshold are selected on the
primary OOF scheme, checked on four additional schemes, and then evaluated once
on validation.  This script never opens test inputs or references.
"""

from collections import Counter, defaultdict
from fractions import Fraction
import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from common import load_inputs, parse_answer
from dimension_consistency_probe import choices
from group_ranker_probe import candidate_labels
from groupwise_question_ledger_probe import (
    VARIANTS,
    _folds,
    _model_z_oof_scores,
    _model_z_validation_scores,
    _target_sentence,
    build_group_dataset,
    group_adjustment,
    oof_group_scores,
    validation_group_scores,
)
from model_t_probe_utils import SEEDS
from quantity_state_ledger import quantity_state_ledger_features
from structure_text_fusion_verifier import answer_structures


EXPERT_ALPHA = {variant: 3e-4 for variant in VARIANTS}
ROUTER_CS = (0.01, 0.1, 1.0)
ROUTER_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 1.01)
CLASS_WEIGHTS = (None, "balanced")
FEATURE_SETS = ("compact", "with_target_text")
TARGET_HASH_DIM = 64


def _answer_groups(row):
    groups = defaultdict(list)
    for index, candidate in enumerate(row["candidates"]):
        answer = parse_answer(candidate["text"])
        if answer is not None:
            groups[answer].append(index)
    return groups


def _group_score(row, scores, answer):
    indices = _answer_groups(row).get(answer, ())
    if not indices:
        return -1e9
    values = scores[list(indices)]
    return float(values.max() + 0.3 * values.mean())


def _margin(row, scores):
    values = sorted(
        (
            _group_score(row, scores, answer)
            for answer in _answer_groups(row)
        ),
        reverse=True,
    )
    return values[0] - values[1] if len(values) > 1 else 0.0


def _fraction_features(base_answer, proposal_answer, groups):
    try:
        base = float(Fraction(base_answer))
        proposal = float(Fraction(proposal_answer))
        ratio = abs(proposal / base) if base else 0.0
        ratio = np.sign(ratio) * np.log1p(abs(ratio))
        difference = np.sign(proposal - base) * np.log1p(abs(proposal - base))
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        ratio = difference = 0.0
    numeric = []
    for answer in groups:
        try:
            numeric.append((float(Fraction(answer)), answer))
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            pass
    numeric.sort()
    denominator = max(len(numeric) - 1, 1)
    ranks = {answer: index / denominator for index, (_, answer) in enumerate(numeric)}
    return (
        float(ratio),
        float(difference),
        float(ranks.get(base_answer, 0.0)),
        float(ranks.get(proposal_answer, 0.0)),
    )


def _ledger_group(features, indices):
    values = features[list(indices)]
    return np.concatenate([values.mean(axis=0), values.max(axis=0)])


def _target_hash(bank):
    vectorizer = HashingVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        n_features=TARGET_HASH_DIM,
        alternate_sign=True,
        norm="l2",
        lowercase=True,
        dtype=np.float32,
    )
    return vectorizer.transform(
        [_target_sentence(row["question"]) for row in bank]
    ).toarray()


def _expert_choices(bank, adjustments):
    empty = [set() for _ in bank]
    return {
        variant: choices(bank, scores, empty)
        for variant, scores in adjustments.items()
    }


def build_router_dataset(bank, labels, base_scores, adjustments):
    """Build proposal-level features and signed accuracy gains."""
    empty = [set() for _ in bank]
    base_choices = choices(bank, base_scores, empty)
    expert_choices = _expert_choices(bank, adjustments)
    ledger = quantity_state_ledger_features(bank)
    target_hash = _target_hash(bank)
    structures = answer_structures(bank)
    structure_names = ("4", "3-1", "2-2", "2-1-1", "1-1-1-1")

    compact_rows = []
    full_rows = []
    questions = []
    proposals = []
    supports = []
    gains = []
    for question_index, row in enumerate(bank):
        groups = _answer_groups(row)
        base_choice = int(base_choices[question_index])
        base_answer = parse_answer(row["candidates"][base_choice]["text"])
        votes = []
        for variant in VARIANTS:
            choice = int(expert_choices[variant][question_index])
            answer = parse_answer(row["candidates"][choice]["text"])
            if answer is not None:
                votes.append((variant, answer, choice))
        counts = Counter(answer for _, answer, _ in votes)
        if not counts:
            continue
        proposal_answer, support = max(
            counts.items(), key=lambda item: (item[1], item[0])
        )
        if support < 2 or proposal_answer == base_answer:
            continue
        preferred = next(
            (
                choice
                for variant, answer, choice in votes
                if variant == "semantic_plus_ledger" and answer == proposal_answer
            ),
            None,
        )
        if preferred is None:
            preferred = next(
                choice for _, answer, choice in votes if answer == proposal_answer
            )
        proposal_choice = int(preferred)
        base_correct = bool(labels[question_index, base_choice])
        proposal_correct = bool(labels[question_index, proposal_choice])
        gain = int(proposal_correct) - int(base_correct)

        base_group = groups[base_answer]
        proposal_group = groups[proposal_answer]
        base_group_score = _group_score(
            row, base_scores[question_index], base_answer
        )
        proposal_group_score = _group_score(
            row, base_scores[question_index], proposal_answer
        )
        expert_values = []
        for variant in VARIANTS:
            row_scores = adjustments[variant][question_index]
            expert_values.extend(
                (
                    _margin(row, row_scores),
                    _group_score(row, row_scores, proposal_answer)
                    - _group_score(row, row_scores, base_answer),
                    float(
                        parse_answer(
                            row["candidates"][
                                int(expert_choices[variant][question_index])
                            ]["text"]
                        )
                        == proposal_answer
                    ),
                )
            )
        ledger_difference = _ledger_group(
            ledger[question_index], proposal_group
        ) - _ledger_group(ledger[question_index], base_group)
        numeric = _fraction_features(base_answer, proposal_answer, groups)
        structure_one_hot = [
            float(structures[question_index] == name) for name in structure_names
        ]
        compact = np.asarray(
            [
                support / len(VARIANTS),
                _margin(row, base_scores[question_index]),
                proposal_group_score - base_group_score,
                len(base_group) / 4,
                len(proposal_group) / 4,
                *expert_values,
                *numeric,
                *structure_one_hot,
                *ledger_difference.tolist(),
            ],
            dtype=np.float32,
        )
        compact_rows.append(compact)
        full_rows.append(
            np.concatenate([compact, target_hash[question_index]]).astype(
                np.float32
            )
        )
        questions.append(question_index)
        proposals.append(proposal_choice)
        supports.append(support)
        gains.append(gain)

    return {
        "compact": np.asarray(compact_rows, dtype=np.float32),
        "with_target_text": np.asarray(full_rows, dtype=np.float32),
        "questions": np.asarray(questions, dtype=np.int64),
        "proposals": np.asarray(proposals, dtype=np.int64),
        "supports": np.asarray(supports, dtype=np.int8),
        "gains": np.asarray(gains, dtype=np.int8),
        "proposal_structures": np.asarray(
            [structures[question] for question in questions]
        ),
        "base_choices": np.asarray(base_choices, dtype=np.int64),
        "n_questions": len(bank),
    }


def _fit_router(features, gains, fit, c_value, class_weight):
    mean = features[fit].mean(axis=0)
    scale = features[fit].std(axis=0)
    scale[scale < 1e-6] = 1.0
    labels = gains[fit] > 0
    if len(set(labels.tolist())) < 2:
        raise ValueError("Router fit split needs both beneficial and non-beneficial proposals")
    sample_weight = np.where(gains[fit] == 0, 0.25, 1.0)
    model = LogisticRegression(
        C=c_value,
        class_weight=class_weight,
        solver="liblinear",
        max_iter=2000,
        random_state=7506,
    )
    model.fit(
        (features[fit] - mean) / scale,
        labels,
        sample_weight=sample_weight,
    )
    return model, mean, scale


def cross_fitted_probabilities(dataset, seed, config):
    features = dataset[config["feature_set"]]
    question_folds = _folds(dataset["n_questions"], seed)
    probabilities = np.zeros(len(features), dtype=np.float32)
    for fold in range(5):
        held = question_folds[dataset["questions"]] == fold
        fit = ~held
        model, mean, scale = _fit_router(
            features,
            dataset["gains"],
            fit,
            config["c"],
            config["class_weight"],
        )
        probabilities[held] = model.predict_proba(
            (features[held] - mean) / scale
        )[:, 1]
    return probabilities


def fit_validation_probabilities(train, validation, config):
    features = train[config["feature_set"]]
    fit = np.ones(len(features), dtype=bool)
    model, mean, scale = _fit_router(
        features,
        train["gains"],
        fit,
        config["c"],
        config["class_weight"],
    )
    score_x = validation[config["feature_set"]]
    return model.predict_proba((score_x - mean) / scale)[:, 1]


def evaluate_router(
    bank,
    labels,
    dataset,
    probabilities,
    threshold,
    min_support,
    allowed_structures=None,
):
    selected = dataset["base_choices"].copy()
    active = (probabilities >= threshold) & (dataset["supports"] >= min_support)
    if allowed_structures is not None:
        active &= np.isin(dataset["proposal_structures"], allowed_structures)
    selected[dataset["questions"][active]] = dataset["proposals"][active]
    base_correct = labels[
        np.arange(len(labels)), dataset["base_choices"]
    ]
    selected_correct = labels[np.arange(len(labels)), selected]
    structures = answer_structures(bank)
    disagreement = np.asarray(
        [len(_answer_groups(row)) > 1 for row in bank], dtype=bool
    )
    switches = int(active.sum())
    fixes = int((~base_correct & selected_correct).sum())
    harms = int((base_correct & ~selected_correct).sum())
    return {
        "base_correct": int(base_correct.sum()),
        "proposed_correct": int(selected_correct.sum()),
        "delta": int(selected_correct.sum() - base_correct.sum()),
        "switches": switches,
        "fixes": fixes,
        "harms": harms,
        "switch_precision": fixes / max(fixes + harms, 1),
        "proposal_rows": int(len(dataset["questions"])),
        "beneficial_proposals": int((dataset["gains"] > 0).sum()),
        "harmful_proposals": int((dataset["gains"] < 0).sum()),
        "neutral_proposals": int((dataset["gains"] == 0).sum()),
        "disagreement_n": int(disagreement.sum()),
        "base_disagreement_correct": int(base_correct[disagreement].sum()),
        "proposed_disagreement_correct": int(selected_correct[disagreement].sum()),
        "structure_deltas": {
            structure: int(
                selected_correct[structures == structure].sum()
                - base_correct[structures == structure].sum()
            )
            for structure in sorted(set(structures))
        },
    }


def select_router_config(bank, labels, dataset, seed):
    trials = []
    probability_cache = {}
    for feature_set in FEATURE_SETS:
        for c_value in ROUTER_CS:
            for class_weight in CLASS_WEIGHTS:
                base_config = {
                    "feature_set": feature_set,
                    "c": c_value,
                    "class_weight": class_weight,
                }
                key = (feature_set, c_value, class_weight)
                probabilities = cross_fitted_probabilities(
                    dataset, seed, base_config
                )
                probability_cache[key] = probabilities
                for min_support in (2, 3):
                    for threshold in ROUTER_THRESHOLDS:
                        result = evaluate_router(
                            bank,
                            labels,
                            dataset,
                            probabilities,
                            threshold,
                            min_support,
                        )
                        trials.append(
                            {
                                **base_config,
                                "min_support": min_support,
                                "threshold": threshold,
                                **result,
                            }
                        )
    best = max(
        trials,
        key=lambda item: (
            item["delta"],
            -item["harms"],
            item["fixes"],
            -item["switches"],
            item["feature_set"] == "compact",
            -item["c"],
            item["threshold"],
        ),
    )
    config = {
        key: best[key]
        for key in (
            "feature_set",
            "c",
            "class_weight",
            "min_support",
            "threshold",
        )
    }
    probabilities = probability_cache[
        (config["feature_set"], config["c"], config["class_weight"])
    ]
    return config, best, trials, probabilities


def _expert_adjustments(train, validation):
    train_adjustments = {seed: {} for seed in SEEDS}
    validation_adjustments = {}
    for variant in VARIANTS:
        alpha = EXPERT_ALPHA[variant]
        for seed in SEEDS:
            group_scores, _ = oof_group_scores(train, variant, seed, alpha)
            train_adjustments[seed][variant] = group_adjustment(
                train, group_scores
            )
        validation_group, _ = validation_group_scores(
            train, validation, variant, alpha
        )
        validation_adjustments[variant] = group_adjustment(
            validation, validation_group
        )
        print(f"expert_ready={variant}", flush=True)
    return train_adjustments, validation_adjustments


def main():
    started = time.perf_counter()
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels(
        "validation", validation_bank
    ).astype(bool)
    train_groups = build_group_dataset(train_bank, train_arrays, train_labels)
    validation_groups = build_group_dataset(validation_bank, validation_arrays)
    train_base_scores = _model_z_oof_scores(
        train_bank, train_arrays, train_labels
    )
    validation_base_scores = _model_z_validation_scores(
        validation_bank, validation_arrays
    )
    train_adjustments, validation_adjustments = _expert_adjustments(
        train_groups, validation_groups
    )

    primary_dataset = build_router_dataset(
        train_bank,
        train_labels,
        train_base_scores[0],
        train_adjustments[SEEDS[0]],
    )
    config, primary_result, trials, primary_probabilities = select_router_config(
        train_bank, train_labels, primary_dataset, SEEDS[0]
    )
    oof_results = []
    for seed_index, seed in enumerate(SEEDS):
        if seed_index == 0:
            dataset = primary_dataset
            probabilities = primary_probabilities
        else:
            dataset = build_router_dataset(
                train_bank,
                train_labels,
                train_base_scores[seed_index],
                train_adjustments[seed],
            )
            probabilities = cross_fitted_probabilities(dataset, seed, config)
        result = evaluate_router(
            train_bank,
            train_labels,
            dataset,
            probabilities,
            config["threshold"],
            config["min_support"],
        )
        result["seed"] = seed
        oof_results.append(result)

    validation_dataset = build_router_dataset(
        validation_bank,
        validation_labels,
        validation_base_scores,
        validation_adjustments,
    )
    validation_probabilities = fit_validation_probabilities(
        primary_dataset, validation_dataset, config
    )
    validation_result = evaluate_router(
        validation_bank,
        validation_labels,
        validation_dataset,
        validation_probabilities,
        config["threshold"],
        config["min_support"],
    )
    if validation_result["base_correct"] != 768:
        raise AssertionError(
            f"Expected frozen validation baseline 768, got {validation_result['base_correct']}"
        )

    report = {
        "protocol_note": (
            "train OOF expert outputs plus a cross-fitted train router; one frozen "
            "validation evaluation; test inputs/references/scores were not read"
        ),
        "expert_alphas": EXPERT_ALPHA,
        "proposal_rule": (
            "at least two of semantic, ledger and combined experts must agree "
            "on an answer group different from Model Z"
        ),
        "router_selection": (
            "feature set, regularization, class weighting, support and probability "
            "threshold selected only on seed-7506 train OOF"
        ),
        "selected_config": config,
        "primary_selection_result": primary_result,
        "primary_trial_count": len(trials),
        "oof": oof_results,
        "validation": validation_result,
        "wall_seconds": time.perf_counter() - started,
    }
    output = CODE / "runs/diagnostics/oof-confidence-router.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"selected={config} "
        f"oof_deltas={[item['delta'] for item in oof_results]} "
        f"validation_delta={validation_result['delta']:+d} "
        f"validation_fixes={validation_result['fixes']} "
        f"validation_harms={validation_result['harms']}",
        flush=True,
    )
    print(f"saved={output}")


if __name__ == "__main__":
    main()
