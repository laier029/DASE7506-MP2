"""Train-only Model Z2 selective-router experiment.

The experiment learns whether an agreed alternative answer group will fix or
harm Model Z.  Six deliberately small logistic routers must agree before a
switch is allowed, and switching is limited to low-margin Model Z cases in the
two structures that were positive in the earlier train-only audit.

Five development OOF partitions select one frozen gate.  Five previously
unused OOF partitions then audit that gate.  Validation is opened only if the
audit passes.  Test inputs, references and scores are never opened here.
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
from groupwise_question_ledger_probe import _folds, build_group_dataset
from oof_confidence_router_probe import build_router_dataset
from oof_confidence_router_structure_audit import (
    expert_adjustments,
    model_z_scores_for_seed,
    prepared_model_t_inputs,
    validation_expert_adjustments,
)
from groupwise_question_ledger_probe import _model_z_validation_scores
from structure_text_fusion_verifier import answer_structures


DEVELOPMENT_SEEDS = (7506, 11, 29, 47, 83)
AUDIT_SEEDS = (607, 701, 809, 907, 1009)
ALLOWED_STRUCTURES = ("2-1-1", "1-1-1-1")
MIN_EXPERT_SUPPORT = 3
MEMBER_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)
MIN_MEMBER_VOTES = (4, 5, 6)
MARGIN_QUANTILES = (0.25, 0.50, 0.75, 1.00)
MIN_EFFECTIVE_DEVELOPMENT_SWITCHES = 20

# Diversity comes only from two existing feature views, two regularization
# strengths and class balancing.  This small set is fixed before the audit.
ROUTER_MEMBERS = (
    {"feature_set": "compact", "c": 0.1, "class_weight": None},
    {"feature_set": "compact", "c": 1.0, "class_weight": "balanced"},
    {"feature_set": "with_target_text", "c": 0.1, "class_weight": None},
    {"feature_set": "with_target_text", "c": 0.1, "class_weight": "balanced"},
    {"feature_set": "with_target_text", "c": 1.0, "class_weight": None},
    {"feature_set": "with_target_text", "c": 1.0, "class_weight": "balanced"},
)


def _fit_gain_router(features, gains, fit, config):
    """Fit benefit-versus-harm directly; neutral proposals do not set the target."""
    mean = features[fit].mean(axis=0)
    scale = features[fit].std(axis=0)
    scale[scale < 1e-6] = 1.0
    labelled = fit & (gains != 0)
    labels = gains[labelled] > 0
    if len(set(labels.tolist())) < 2:
        raise ValueError("Gain router needs both fixes and harms in every fit split")
    model = LogisticRegression(
        C=config["c"],
        class_weight=config["class_weight"],
        solver="liblinear",
        max_iter=2000,
        random_state=7506,
    )
    model.fit((features[labelled] - mean) / scale, labels)
    return model, mean, scale


def cross_fitted_ensemble(dataset, seed):
    """Return one question-level cross-fitted probability per router member."""
    question_folds = _folds(dataset["n_questions"], seed)
    probabilities = np.zeros(
        (len(dataset["questions"]), len(ROUTER_MEMBERS)), dtype=np.float32
    )
    for member_index, config in enumerate(ROUTER_MEMBERS):
        features = dataset[config["feature_set"]]
        for fold in range(5):
            held = question_folds[dataset["questions"]] == fold
            fit = ~held
            model, mean, scale = _fit_gain_router(
                features, dataset["gains"], fit, config
            )
            probabilities[held, member_index] = model.predict_proba(
                (features[held] - mean) / scale
            )[:, 1]
    return probabilities


def fit_validation_ensemble(train, validation):
    """Fit the frozen ensemble on primary OOF proposals and score validation."""
    probabilities = np.zeros(
        (len(validation["questions"]), len(ROUTER_MEMBERS)), dtype=np.float32
    )
    fit = np.ones(len(train["questions"]), dtype=bool)
    for member_index, config in enumerate(ROUTER_MEMBERS):
        train_x = train[config["feature_set"]]
        score_x = validation[config["feature_set"]]
        model, mean, scale = _fit_gain_router(
            train_x, train["gains"], fit, config
        )
        probabilities[:, member_index] = model.predict_proba(
            (score_x - mean) / scale
        )[:, 1]
    return probabilities


def active_switches(dataset, probabilities, gate):
    votes = (probabilities >= gate["member_threshold"]).sum(axis=1)
    margins = dataset["compact"][:, 1]
    return (
        (votes >= gate["min_member_votes"])
        & (dataset["supports"] >= MIN_EXPERT_SUPPORT)
        & np.isin(dataset["proposal_structures"], ALLOWED_STRUCTURES)
        & (margins <= gate["max_base_margin"])
    )


def evaluate_active(bank, labels, dataset, active):
    selected = dataset["base_choices"].copy()
    selected[dataset["questions"][active]] = dataset["proposals"][active]
    rows = np.arange(len(labels))
    base_correct = labels[rows, dataset["base_choices"]]
    selected_correct = labels[rows, selected]
    structures = answer_structures(bank)
    fixes = int((~base_correct & selected_correct).sum())
    harms = int((base_correct & ~selected_correct).sum())
    return {
        "base_correct": int(base_correct.sum()),
        "proposed_correct": int(selected_correct.sum()),
        "delta": int(selected_correct.sum() - base_correct.sum()),
        "switches": int(active.sum()),
        "fixes": fixes,
        "harms": harms,
        "neutral_switches": int(active.sum()) - fixes - harms,
        "switch_precision": fixes / max(fixes + harms, 1),
        "structure_deltas": {
            structure: int(
                selected_correct[structures == structure].sum()
                - base_correct[structures == structure].sum()
            )
            for structure in sorted(set(structures))
        },
    }


def wilson_lower(fixes, harms, z=1.6448536269514722):
    """One-sided 95% lower bound for precision among effective switches."""
    n = fixes + harms
    if n == 0:
        return 0.0
    p = fixes / n
    denominator = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    radius = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return float((centre - radius) / denominator)


def summarize_results(results):
    fixes = sum(item["fixes"] for item in results)
    harms = sum(item["harms"] for item in results)
    deltas = [item["delta"] for item in results]
    return {
        "deltas": deltas,
        "min_delta": min(deltas),
        "positive_seeds": sum(delta > 0 for delta in deltas),
        "total_delta": sum(deltas),
        "total_switches": sum(item["switches"] for item in results),
        "total_fixes": fixes,
        "total_harms": harms,
        "effective_switches": fixes + harms,
        "pooled_switch_precision": fixes / max(fixes + harms, 1),
        "precision_wilson_lower_95_one_sided": wilson_lower(fixes, harms),
    }


def development_search(banks, labels, datasets, probabilities):
    pooled_margins = np.concatenate(
        [dataset["compact"][:, 1] for dataset in datasets]
    )
    margin_cutoffs = sorted(
        {float(np.quantile(pooled_margins, q)) for q in MARGIN_QUANTILES}
    )
    trials = []
    for threshold in MEMBER_THRESHOLDS:
        for votes in MIN_MEMBER_VOTES:
            for max_margin in margin_cutoffs:
                gate = {
                    "member_threshold": threshold,
                    "min_member_votes": votes,
                    "max_base_margin": max_margin,
                }
                results = [
                    evaluate_active(
                        bank,
                        labels,
                        dataset,
                        active_switches(dataset, member_probs, gate),
                    )
                    for bank, dataset, member_probs in zip(
                        banks, datasets, probabilities
                    )
                ]
                summary = summarize_results(results)
                eligible = (
                    summary["min_delta"] >= 0
                    and summary["positive_seeds"] >= 3
                    and summary["effective_switches"]
                    >= MIN_EFFECTIVE_DEVELOPMENT_SWITCHES
                    and summary["precision_wilson_lower_95_one_sided"] > 0.5
                )
                trials.append(
                    {
                        "gate": gate,
                        "eligible": eligible,
                        "summary": summary,
                        "per_seed": results,
                    }
                )
    eligible_trials = [trial for trial in trials if trial["eligible"]]
    if not eligible_trials:
        return None, trials
    best = max(
        eligible_trials,
        key=lambda trial: (
            trial["summary"]["min_delta"],
            trial["summary"]["precision_wilson_lower_95_one_sided"],
            trial["summary"]["total_delta"],
            -trial["summary"]["total_harms"],
            -trial["summary"]["total_switches"],
        ),
    )
    return best, trials


def build_seed_dataset(
    bank, arrays, labels, group_dataset, prepared, seed
):
    base_scores = model_z_scores_for_seed(
        bank, arrays, labels, seed, prepared
    )
    dataset = build_router_dataset(
        bank,
        labels,
        base_scores,
        expert_adjustments(group_dataset, seed),
    )
    probabilities = cross_fitted_ensemble(dataset, seed)
    return dataset, probabilities


def _json_ready_member_configs():
    return [dict(config) for config in ROUTER_MEMBERS]


def main():
    started = time.perf_counter()
    train_bank, train_arrays = load_inputs("train")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    train_groups = build_group_dataset(train_bank, train_arrays, train_labels)
    prepared = prepared_model_t_inputs(train_bank)

    development_datasets = []
    development_probabilities = []
    for seed in DEVELOPMENT_SEEDS:
        dataset, probabilities = build_seed_dataset(
            train_bank,
            train_arrays,
            train_labels,
            train_groups,
            prepared,
            seed,
        )
        development_datasets.append(dataset)
        development_probabilities.append(probabilities)
        print(f"development_seed_ready={seed}", flush=True)

    selected, trials = development_search(
        [train_bank] * len(DEVELOPMENT_SEEDS),
        train_labels,
        development_datasets,
        development_probabilities,
    )
    report = {
        "protocol_note": (
            "five development train OOF partitions select one ensemble gate; "
            "five previously unused train OOF partitions audit it; validation "
            "is opened only after the audit passes; test data are not opened"
        ),
        "router_target": (
            "beneficial switch versus harmful switch; neutral switches excluded "
            "from router labels"
        ),
        "router_members": _json_ready_member_configs(),
        "allowed_structures": ALLOWED_STRUCTURES,
        "minimum_expert_support": MIN_EXPERT_SUPPORT,
        "development_seeds": DEVELOPMENT_SEEDS,
        "audit_seeds": AUDIT_SEEDS,
        "development_trial_count": len(trials),
        "development": selected,
        "audit": None,
        "validation": None,
        "accepted_as_z2_candidate": False,
    }
    if selected is None:
        report["decision"] = (
            "No development gate met the predeclared stability and confidence "
            "requirements; audit and validation were not opened."
        )
    else:
        gate = selected["gate"]
        audit_results = []
        for seed in AUDIT_SEEDS:
            dataset, probabilities = build_seed_dataset(
                train_bank,
                train_arrays,
                train_labels,
                train_groups,
                prepared,
                seed,
            )
            result = evaluate_active(
                train_bank,
                train_labels,
                dataset,
                active_switches(dataset, probabilities, gate),
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
        report["audit"] = {
            "passed": audit_passed,
            "summary": audit_summary,
            "per_seed": audit_results,
        }

        if audit_passed:
            validation_bank, validation_arrays = load_inputs("validation")
            validation_labels = candidate_labels(
                "validation", validation_bank
            ).astype(bool)
            validation_groups = build_group_dataset(
                validation_bank, validation_arrays
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
            validation_probabilities = fit_validation_ensemble(
                development_datasets[0], validation_dataset
            )
            validation_result = evaluate_active(
                validation_bank,
                validation_labels,
                validation_dataset,
                active_switches(
                    validation_dataset, validation_probabilities, gate
                ),
            )
            if validation_result["base_correct"] != 768:
                raise AssertionError("Frozen Model Z validation baseline changed")
            report["validation"] = validation_result
            report["accepted_as_z2_candidate"] = (
                validation_result["delta"] >= 0
            )
            report["decision"] = (
                "Accepted as a train/validation candidate; productionization "
                "still requires resource and reproducibility checks."
                if report["accepted_as_z2_candidate"]
                else "Rejected because the frozen validation check regressed."
            )
        else:
            report["decision"] = (
                "Rejected by the fresh OOF audit; validation was not opened."
            )

    report["wall_seconds"] = time.perf_counter() - started
    output = CODE / "runs/diagnostics/oof-router-ensemble-z2.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"decision={report['decision']}", flush=True)
    print(f"saved={output}", flush=True)


if __name__ == "__main__":
    main()
