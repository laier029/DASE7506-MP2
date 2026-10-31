"""Final pooled-OOF router adjustment without test access.

This experiment changes exactly one part of the rejected Z2 candidate: the
six final routers learn from five OOF representations of train instead of the
single seed-7506 representation.  A question's total labelled weight is
normalised to one across repeated representations.  The experts, structures,
vote threshold, probability threshold and Model Z margin gate remain frozen.

Five new train OOF partitions audit the adjustment.  Validation is evaluated
once only if every audit delta is non-negative and the pooled precision lower
bound exceeds chance.  Test inputs, references and scores are never opened.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from common import load_inputs
from group_ranker_probe import candidate_labels
from groupwise_question_ledger_probe import (
    _folds,
    _model_z_validation_scores,
    build_group_dataset,
)
from oof_confidence_router_probe import build_router_dataset
from oof_confidence_router_structure_audit import (
    expert_adjustments,
    model_z_scores_for_seed,
    prepared_model_t_inputs,
    validation_expert_adjustments,
)
from oof_router_ensemble_z2 import (
    ROUTER_MEMBERS,
    active_switches,
    evaluate_active,
    summarize_results,
)


POOL_SEEDS = (7506, 11, 29, 47, 83)
FINAL_AUDIT_SEEDS = (1103, 1201, 1301, 1409, 1511)
FROZEN_GATE = {
    "member_threshold": 0.6,
    "min_member_votes": 5,
    "max_base_margin": 7.175619125366211,
}


def build_seed_dataset(bank, arrays, labels, group_dataset, prepared, seed):
    base_scores = model_z_scores_for_seed(
        bank, arrays, labels, seed, prepared
    )
    return build_router_dataset(
        bank,
        labels,
        base_scores,
        expert_adjustments(group_dataset, seed),
    )


def _pooled_arrays(datasets, feature_set):
    return (
        np.concatenate([dataset[feature_set] for dataset in datasets]),
        np.concatenate([dataset["gains"] for dataset in datasets]),
        np.concatenate([dataset["questions"] for dataset in datasets]),
    )


def _fit_pooled_member(features, gains, questions, fit, config):
    """Fit one router while giving every labelled question total weight one."""
    mean = features[fit].mean(axis=0)
    scale = features[fit].std(axis=0)
    scale[scale < 1e-6] = 1.0
    labelled = fit & (gains != 0)
    labels = gains[labelled] > 0
    if len(set(labels.tolist())) < 2:
        raise ValueError("Pooled gain router requires both fixes and harms")

    labelled_questions = questions[labelled]
    counts = np.bincount(labelled_questions, minlength=int(questions.max()) + 1)
    sample_weight = 1.0 / counts[labelled_questions]
    model = LogisticRegression(
        C=config["c"],
        class_weight=config["class_weight"],
        solver="liblinear",
        max_iter=2000,
        random_state=7506,
    )
    model.fit(
        (features[labelled] - mean) / scale,
        labels,
        sample_weight=sample_weight,
    )
    return model, mean, scale


def cross_fitted_pooled_probabilities(pool, score_dataset, audit_seed):
    """Score an audit dataset while excluding each held question fold."""
    question_folds = _folds(score_dataset["n_questions"], audit_seed)
    probabilities = np.zeros(
        (len(score_dataset["questions"]), len(ROUTER_MEMBERS)),
        dtype=np.float32,
    )
    for member_index, config in enumerate(ROUTER_MEMBERS):
        pool_x, pool_gains, pool_questions = _pooled_arrays(
            pool, config["feature_set"]
        )
        score_x = score_dataset[config["feature_set"]]
        for fold in range(5):
            held_questions = question_folds == fold
            fit = ~held_questions[pool_questions]
            held_rows = held_questions[score_dataset["questions"]]
            model, mean, scale = _fit_pooled_member(
                pool_x, pool_gains, pool_questions, fit, config
            )
            probabilities[held_rows, member_index] = model.predict_proba(
                (score_x[held_rows] - mean) / scale
            )[:, 1]
    return probabilities


def fit_validation_probabilities(pool, validation):
    probabilities = np.zeros(
        (len(validation["questions"]), len(ROUTER_MEMBERS)), dtype=np.float32
    )
    for member_index, config in enumerate(ROUTER_MEMBERS):
        pool_x, pool_gains, pool_questions = _pooled_arrays(
            pool, config["feature_set"]
        )
        fit = np.ones(len(pool_x), dtype=bool)
        model, mean, scale = _fit_pooled_member(
            pool_x, pool_gains, pool_questions, fit, config
        )
        score_x = validation[config["feature_set"]]
        probabilities[:, member_index] = model.predict_proba(
            (score_x - mean) / scale
        )[:, 1]
    return probabilities


def main():
    started = time.perf_counter()
    train_bank, train_arrays = load_inputs("train")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    train_groups = build_group_dataset(train_bank, train_arrays, train_labels)
    prepared = prepared_model_t_inputs(train_bank)

    pool = []
    for seed in POOL_SEEDS:
        pool.append(
            build_seed_dataset(
                train_bank,
                train_arrays,
                train_labels,
                train_groups,
                prepared,
                seed,
            )
        )
        print(f"pool_seed_ready={seed}", flush=True)

    audit_results = []
    for seed in FINAL_AUDIT_SEEDS:
        dataset = build_seed_dataset(
            train_bank,
            train_arrays,
            train_labels,
            train_groups,
            prepared,
            seed,
        )
        probabilities = cross_fitted_pooled_probabilities(pool, dataset, seed)
        result = evaluate_active(
            train_bank,
            train_labels,
            dataset,
            active_switches(dataset, probabilities, FROZEN_GATE),
        )
        result["seed"] = seed
        audit_results.append(result)
        print(
            f"audit_seed={seed} delta={result['delta']:+d} "
            f"fixes={result['fixes']} harms={result['harms']} "
            f"switches={result['switches']}",
            flush=True,
        )

    audit_summary = summarize_results(audit_results)
    audit_passed = (
        audit_summary["min_delta"] >= 0
        and audit_summary["positive_seeds"] >= 3
        and audit_summary["precision_wilson_lower_95_one_sided"] > 0.5
    )
    report = {
        "protocol_note": (
            "one frozen adjustment: pool five train OOF router datasets with "
            "per-question weight normalization; five new train OOF seeds audit "
            "the unchanged gate; validation is checked only after audit pass; "
            "test data are not opened"
        ),
        "only_adjustment": (
            "router fit source changed from one OOF dataset to five pooled OOF "
            "datasets; expert models and all decision gates are unchanged"
        ),
        "pool_seeds": POOL_SEEDS,
        "audit_seeds": FINAL_AUDIT_SEEDS,
        "frozen_gate": FROZEN_GATE,
        "router_members": [dict(config) for config in ROUTER_MEMBERS],
        "audit": {
            "passed": audit_passed,
            "summary": audit_summary,
            "per_seed": audit_results,
        },
        "validation": None,
        "accepted_as_final_candidate": False,
    }

    if audit_passed:
        validation_bank, validation_arrays = load_inputs("validation")
        validation_labels = candidate_labels(
            "validation", validation_bank
        ).astype(bool)
        validation_groups = build_group_dataset(
            validation_bank, validation_arrays
        )
        validation_dataset = build_router_dataset(
            validation_bank,
            validation_labels,
            _model_z_validation_scores(validation_bank, validation_arrays),
            validation_expert_adjustments(train_groups, validation_groups),
        )
        probabilities = fit_validation_probabilities(pool, validation_dataset)
        validation_result = evaluate_active(
            validation_bank,
            validation_labels,
            validation_dataset,
            active_switches(validation_dataset, probabilities, FROZEN_GATE),
        )
        if validation_result["base_correct"] != 768:
            raise AssertionError("Frozen Model Z validation baseline changed")
        report["validation"] = validation_result
        report["accepted_as_final_candidate"] = validation_result["delta"] >= 0
        report["decision"] = (
            "Accepted as the final research candidate; productionization and "
            "resource checks remain separate."
            if report["accepted_as_final_candidate"]
            else "Rejected because the final frozen validation check regressed."
        )
    else:
        report["decision"] = (
            "Rejected by the fresh train OOF audit; validation was not opened."
        )

    report["wall_seconds"] = time.perf_counter() - started
    output = CODE / "runs/diagnostics/oof-router-pooled-final.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"decision={report['decision']}", flush=True)
    print(f"saved={output}", flush=True)


if __name__ == "__main__":
    main()
