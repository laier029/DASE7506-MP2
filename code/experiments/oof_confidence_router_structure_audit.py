"""Fresh-seed audit of a structure-restricted confidence router.

The restriction was derived from the primary seed only: the initial router was
positive on 2-1-1 and all-distinct questions and negative on stronger-consensus
structures.  Because the original robustness seeds were already observed, this
script uses new train OOF partitions as an independent audit before one frozen
validation evaluation.
"""

import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from calculation_coverage import calculation_occurrence_coverage
from cascade_stability_probe import model_m_oof
from common import load_inputs
from group_ranker_probe import candidate_labels
from groupwise_question_ledger_probe import (
    VARIANTS,
    _folds,
    _model_z_validation_scores,
    build_group_dataset,
    group_adjustment,
    oof_group_scores,
    validation_group_scores,
)
from model_t_probe_utils import standardize_rows
from numeric_filter_stability_probe import apply_model_o_coverage
from oof_confidence_router_probe import (
    EXPERT_ALPHA,
    build_router_dataset,
    cross_fitted_probabilities,
    evaluate_router,
    fit_validation_probabilities,
)
from semantic_rank_features import semantic_rank_adjusted_scores
from stable_signal_features import stable_signal_adjusted_scores
from structure_text_fusion_verifier import answer_structures
from target_alignment import terminal_target_recall
from terminal_dependency_features import terminal_dependency_adjusted_scores


PRIMARY_SEED = 7506
FRESH_SEEDS = (101, 211, 307, 401, 503)
ALLOWED_STRUCTURES = ("2-1-1", "1-1-1-1")
ROUTER_CONFIG = {
    "feature_set": "with_target_text",
    "c": 1.0,
    "class_weight": "balanced",
    "min_support": 3,
    "threshold": 0.9,
}


def model_z_scores_for_seed(bank, arrays, labels, seed, prepared):
    """Rebuild Model Z OOF scores for an arbitrary fresh partition seed."""
    text_features, structures, occurrence, target = prepared
    scores = model_m_oof(
        bank,
        arrays,
        labels,
        _folds(len(bank), seed),
        text_features,
        structures,
    )
    scores = apply_model_o_coverage(bank, scores, structures)
    rows = structures == "1-1-1-1"
    scores[rows] += 0.225 * occurrence[rows]
    scores[rows] += 0.125 * target[rows]
    scores = stable_signal_adjusted_scores(bank, scores)
    scores = semantic_rank_adjusted_scores(bank, scores)
    return terminal_dependency_adjusted_scores(bank, scores)


def prepared_model_t_inputs(bank):
    artifact = joblib.load(
        CODE / "runs/model_t_final_cascade/structure_text_model.joblib"
    )
    texts = [candidate["text"] for row in bank for candidate in row["candidates"]]
    text_features = artifact["vectorizer"].transform(texts)
    structures = answer_structures(bank)
    occurrence = standardize_rows(calculation_occurrence_coverage(bank))
    target = standardize_rows(terminal_target_recall(bank))
    return text_features, structures, occurrence, target


def expert_adjustments(dataset, seed):
    output = {}
    for variant in VARIANTS:
        scores, _ = oof_group_scores(
            dataset, variant, seed, EXPERT_ALPHA[variant]
        )
        output[variant] = group_adjustment(dataset, scores)
    return output


def validation_expert_adjustments(train, validation):
    output = {}
    for variant in VARIANTS:
        scores, _ = validation_group_scores(
            train, validation, variant, EXPERT_ALPHA[variant]
        )
        output[variant] = group_adjustment(validation, scores)
    return output


def main():
    started = time.perf_counter()
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    train_groups = build_group_dataset(train_bank, train_arrays, train_labels)
    validation_groups = build_group_dataset(validation_bank, validation_arrays)
    prepared = prepared_model_t_inputs(train_bank)

    primary_scores = model_z_scores_for_seed(
        train_bank, train_arrays, train_labels, PRIMARY_SEED, prepared
    )
    primary_dataset = build_router_dataset(
        train_bank,
        train_labels,
        primary_scores,
        expert_adjustments(train_groups, PRIMARY_SEED),
    )
    primary_probabilities = cross_fitted_probabilities(
        primary_dataset, PRIMARY_SEED, ROUTER_CONFIG
    )
    primary_result = evaluate_router(
        train_bank,
        train_labels,
        primary_dataset,
        primary_probabilities,
        ROUTER_CONFIG["threshold"],
        ROUTER_CONFIG["min_support"],
        ALLOWED_STRUCTURES,
    )

    fresh_results = []
    for seed in FRESH_SEEDS:
        base_scores = model_z_scores_for_seed(
            train_bank, train_arrays, train_labels, seed, prepared
        )
        dataset = build_router_dataset(
            train_bank,
            train_labels,
            base_scores,
            expert_adjustments(train_groups, seed),
        )
        probabilities = cross_fitted_probabilities(dataset, seed, ROUTER_CONFIG)
        result = evaluate_router(
            train_bank,
            train_labels,
            dataset,
            probabilities,
            ROUTER_CONFIG["threshold"],
            ROUTER_CONFIG["min_support"],
            ALLOWED_STRUCTURES,
        )
        result["seed"] = seed
        fresh_results.append(result)
        print(
            f"fresh_seed={seed} delta={result['delta']:+d} "
            f"fixes={result['fixes']} harms={result['harms']} "
            f"switches={result['switches']}",
            flush=True,
        )

    validation_base = _model_z_validation_scores(
        validation_bank, validation_arrays
    )
    validation_dataset = build_router_dataset(
        validation_bank,
        validation_labels,
        validation_base,
        validation_expert_adjustments(train_groups, validation_groups),
    )
    validation_probabilities = fit_validation_probabilities(
        primary_dataset, validation_dataset, ROUTER_CONFIG
    )
    validation_result = evaluate_router(
        validation_bank,
        validation_labels,
        validation_dataset,
        validation_probabilities,
        ROUTER_CONFIG["threshold"],
        ROUTER_CONFIG["min_support"],
        ALLOWED_STRUCTURES,
    )
    if validation_result["base_correct"] != 768:
        raise AssertionError("Frozen Model Z validation baseline changed")

    report = {
        "protocol_note": (
            "structure restriction derived from primary train OOF; five new "
            "question-level OOF seeds audited before one frozen validation run; "
            "test inputs/references/scores were not read"
        ),
        "router_config": ROUTER_CONFIG,
        "allowed_structures": ALLOWED_STRUCTURES,
        "primary": primary_result,
        "fresh_oof": fresh_results,
        "validation": validation_result,
        "wall_seconds": time.perf_counter() - started,
    }
    output = CODE / "runs/diagnostics/oof-confidence-router-structure-audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"fresh_deltas={[item['delta'] for item in fresh_results]} "
        f"validation_delta={validation_result['delta']:+d} "
        f"validation_fixes={validation_result['fixes']} "
        f"validation_harms={validation_result['harms']}",
        flush=True,
    )
    print(f"saved={output}")


if __name__ == "__main__":
    main()
