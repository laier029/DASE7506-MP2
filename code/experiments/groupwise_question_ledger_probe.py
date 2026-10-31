"""Evaluate two high-level verifier directions without touching test data.

Direction 1 is an answer-group ranker with explicit question/candidate text
interactions and a compact bilinear interaction between question words and the
supplied candidate embeddings.  Direction 2 adds a generic arithmetic state
ledger and learns when target wording should prefer particular dependency
shapes.  Hyperparameters are selected on the primary train OOF scheme; four
additional OOF schemes are robustness checks, and validation is evaluated only
after each variant is frozen.
"""

from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from cascade_stability_probe import balanced_random_folds
from common import load_inputs, parse_answer
from dimension_consistency_probe import choices
from group_ranker_probe import candidate_labels
from model_t_probe_utils import SEEDS, load_or_build_model_t_oof
from quantity_state_ledger import FEATURE_NAMES, quantity_state_ledger_features
from semantic_rank_features import semantic_rank_adjusted_scores
from stable_signal_features import stable_signal_adjusted_scores
from stable_signal_verifier import stable_signal_scores
from structure_text_fusion_verifier import answer_structures
from terminal_dependency_features import terminal_dependency_adjusted_scores


TEXT_DIM = 256
QUESTION_EMBED_DIM = 64
CANDIDATE_EMBED_DIM = 64
LEDGER_TARGET_DIM = 32
ALPHAS = (3e-5, 1e-4, 3e-4)
FUSION_WEIGHTS = (0.0, 0.025, 0.05, 0.10, 0.20, 0.30, 0.50)
VARIANTS = ("semantic_interaction", "ledger_only", "semantic_plus_ledger")


def _hashing_vectorizer(dim):
    return HashingVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        n_features=dim,
        alternate_sign=True,
        norm="l2",
        lowercase=True,
        dtype=np.float32,
    )


def _target_sentence(question):
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", question) if part.strip()]
    return parts[-1] if parts else question


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else vector


def _row_blocks(*blocks):
    return np.concatenate([_unit(block).ravel() for block in blocks]).astype(
        np.float32, copy=False
    )


def _folds(n_rows, seed):
    if seed == 7506:
        return np.arange(n_rows) % 5
    return balanced_random_folds(n_rows, seed)


def _answer_groups(row):
    groups = defaultdict(list)
    for index, candidate in enumerate(row["candidates"]):
        answer = parse_answer(candidate["text"])
        if answer is not None:
            groups[answer].append(index)
    return groups


def _structure_metadata(groups):
    structure = sorted((len(indices) for indices in groups.values()), reverse=True)
    structures = ([4], [3, 1], [2, 2], [2, 1, 1], [1, 1, 1, 1])
    return np.asarray(
        [len(groups) / 4, *(float(structure == item) for item in structures)],
        dtype=np.float32,
    )


def _candidate_projection(arrays):
    embeddings = arrays["embeddings"].astype(np.float32)
    embeddings = embeddings - embeddings.mean(axis=1, keepdims=True)
    rng = np.random.default_rng(7506)
    projection = rng.normal(
        0.0,
        1.0 / np.sqrt(CANDIDATE_EMBED_DIM),
        size=(embeddings.shape[-1], CANDIDATE_EMBED_DIM),
    ).astype(np.float32)
    projected = embeddings @ projection
    norms = np.linalg.norm(projected, axis=2, keepdims=True)
    return projected / np.maximum(norms, 1e-6)


def build_group_dataset(bank, arrays, labels=None):
    """Build aligned semantic and ledger features for parsed answer groups."""
    n_rows = len(bank)
    questions = [row["question"] for row in bank]
    targets = [_target_sentence(question) for question in questions]
    candidate_texts = [
        candidate["text"] for row in bank for candidate in row["candidates"]
    ]
    text_hasher = _hashing_vectorizer(TEXT_DIM)
    question_text = text_hasher.transform(questions).toarray()
    target_text = text_hasher.transform(targets).toarray()
    candidate_text = text_hasher.transform(candidate_texts).toarray().reshape(
        n_rows, 4, TEXT_DIM
    )
    question_embed = _hashing_vectorizer(QUESTION_EMBED_DIM).transform(
        questions
    ).toarray()
    ledger_target = _hashing_vectorizer(LEDGER_TARGET_DIM).transform(
        targets
    ).toarray()
    candidate_embed = _candidate_projection(arrays)
    ledger = quantity_state_ledger_features(bank)

    question_indices = []
    members = []
    group_targets = []
    semantic_rows = []
    ledger_raw_rows = []
    metadata_rows = []
    for question_index, row in enumerate(bank):
        groups = _answer_groups(row)
        metadata = _structure_metadata(groups)
        for indices in groups.values():
            candidate_word = candidate_text[question_index, indices].mean(axis=0)
            group_embedding = _unit(
                candidate_embed[question_index, indices].mean(axis=0)
            )
            bilinear = np.outer(
                question_embed[question_index], group_embedding
            ).ravel()
            semantic = np.concatenate(
                [
                    _row_blocks(
                        candidate_word,
                        question_text[question_index] * candidate_word,
                        target_text[question_index] * candidate_word,
                        np.abs(question_text[question_index] - candidate_word),
                        np.abs(target_text[question_index] - candidate_word),
                        bilinear,
                    ),
                    metadata,
                    np.asarray([len(indices) / 4], dtype=np.float32),
                ]
            )
            ledger_values = ledger[question_index, indices]
            ledger_raw = np.concatenate(
                [ledger_values.mean(axis=0), ledger_values.max(axis=0)]
            ).astype(np.float32)
            question_indices.append(question_index)
            members.append(tuple(indices))
            semantic_rows.append(semantic)
            ledger_raw_rows.append(ledger_raw)
            metadata_rows.append(metadata)
            if labels is not None:
                group_targets.append(bool(labels[question_index, indices].any()))

    question_indices = np.asarray(question_indices, dtype=np.int64)
    semantic_rows = np.asarray(semantic_rows, dtype=np.float32)
    ledger_raw_rows = np.asarray(ledger_raw_rows, dtype=np.float32)
    ledger_centered = np.zeros_like(ledger_raw_rows)
    for question_index in range(n_rows):
        active = question_indices == question_index
        values = ledger_raw_rows[active]
        if not len(values):
            continue
        ledger_centered[active] = (values - values.mean(axis=0)) / (
            values.std(axis=0) + 1e-6
        )
    ledger_rows = []
    for group_index, question_index in enumerate(question_indices):
        target_interaction = np.outer(
            ledger_target[question_index], ledger_centered[group_index]
        ).ravel()
        ledger_rows.append(
            np.concatenate(
                [
                    _row_blocks(
                        np.tanh(ledger_raw_rows[group_index]),
                        ledger_centered[group_index],
                        target_interaction,
                    ),
                    metadata_rows[group_index],
                ]
            )
        )
    ledger_rows = np.asarray(ledger_rows, dtype=np.float32)
    output = {
        "semantic_interaction": semantic_rows,
        "ledger_only": ledger_rows,
        "semantic_plus_ledger": np.concatenate(
            [semantic_rows, ledger_rows], axis=1
        ),
        "questions": question_indices,
        "members": members,
        "n_questions": n_rows,
        "ledger_support": {
            name: int(np.sum(np.ptp(ledger[:, :, index], axis=1) > 0))
            for index, name in enumerate(FEATURE_NAMES)
        },
    }
    if labels is not None:
        output["targets"] = np.asarray(group_targets, dtype=bool)
    return output


def _pairwise_examples(features, questions, targets, fit_questions):
    rows = []
    labels = []
    for question in np.flatnonzero(fit_questions):
        active = np.flatnonzero(questions == question)
        positive = active[targets[active]]
        negative = active[~targets[active]]
        for left in positive:
            for right in negative:
                difference = features[left] - features[right]
                rows.extend((difference, -difference))
                labels.extend((1, 0))
    if not rows:
        raise ValueError("No correct/incorrect answer-group pairs in fit split")
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int8)


def _fit_pairwise(features, questions, targets, fit_questions, alpha):
    train_x, train_y = _pairwise_examples(
        features, questions, targets, fit_questions
    )
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        fit_intercept=False,
        max_iter=3000,
        tol=1e-4,
        shuffle=True,
        random_state=7506,
        average=True,
    )
    model.fit(train_x, train_y)
    return model, len(train_y)


def oof_group_scores(dataset, variant, seed, alpha):
    features = dataset[variant]
    questions = dataset["questions"]
    targets = dataset["targets"]
    folds = _folds(dataset["n_questions"], seed)
    scores = np.zeros(len(features), dtype=np.float32)
    pair_count = 0
    for fold in range(5):
        fit_questions = folds != fold
        held = folds[questions] == fold
        model, pairs = _fit_pairwise(
            features, questions, targets, fit_questions, alpha
        )
        scores[held] = model.decision_function(features[held])
        pair_count += pairs
    return scores, pair_count


def validation_group_scores(train, validation, variant, alpha):
    fit_questions = np.ones(train["n_questions"], dtype=bool)
    model, pair_count = _fit_pairwise(
        train[variant],
        train["questions"],
        train["targets"],
        fit_questions,
        alpha,
    )
    return model.decision_function(validation[variant]), pair_count


def group_adjustment(dataset, group_scores):
    output = np.zeros((dataset["n_questions"], 4), dtype=np.float32)
    questions = dataset["questions"]
    for question in range(dataset["n_questions"]):
        active = np.flatnonzero(questions == question)
        if len(active) <= 1:
            continue
        values = group_scores[active]
        normalized = (values - values.mean()) / (values.std() + 1e-6)
        for group_index, score in zip(active, normalized):
            for candidate in dataset["members"][group_index]:
                output[question, candidate] = score
    return output


def _model_z_oof_scores(bank, arrays, labels):
    output = []
    for scores in load_or_build_model_t_oof(bank, arrays, labels):
        scores = stable_signal_adjusted_scores(bank, scores)
        scores = semantic_rank_adjusted_scores(bank, scores)
        scores = terminal_dependency_adjusted_scores(bank, scores)
        output.append(scores)
    return output


def _model_z_validation_scores(bank, arrays):
    scores = stable_signal_scores(
        bank, arrays, CODE / "runs/model_z_robust_semantic"
    )
    scores = semantic_rank_adjusted_scores(bank, scores)
    return terminal_dependency_adjusted_scores(bank, scores)


def _correctness(labels, selected):
    return labels[np.arange(len(labels)), selected]


def _answer_group_changes(bank, before, after):
    return sum(
        parse_answer(row["candidates"][left]["text"])
        != parse_answer(row["candidates"][right]["text"])
        for row, left, right in zip(bank, before, after)
    )


def evaluate_adjustment(bank, labels, base_scores, adjustment, weight):
    empty = [set() for _ in bank]
    before = choices(bank, base_scores, empty)
    after = choices(bank, base_scores + weight * adjustment, empty)
    before_correct = _correctness(labels, before)
    after_correct = _correctness(labels, after)
    structures = answer_structures(bank)
    disagreement = np.asarray(
        [
            len(
                {
                    parse_answer(candidate["text"])
                    for candidate in row["candidates"]
                    if parse_answer(candidate["text"]) is not None
                }
            )
            > 1
            for row in bank
        ]
    )
    return {
        "base_correct": int(before_correct.sum()),
        "proposed_correct": int(after_correct.sum()),
        "delta": int(after_correct.sum() - before_correct.sum()),
        "fixes": int((~before_correct & after_correct).sum()),
        "harms": int((before_correct & ~after_correct).sum()),
        "answer_group_changes": int(_answer_group_changes(bank, before, after)),
        "disagreement_n": int(disagreement.sum()),
        "base_disagreement_correct": int(before_correct[disagreement].sum()),
        "proposed_disagreement_correct": int(after_correct[disagreement].sum()),
        "structure_deltas": {
            structure: int(
                after_correct[structures == structure].sum()
                - before_correct[structures == structure].sum()
            )
            for structure in sorted(set(structures))
        },
    }


def select_config(
    variant, train, train_bank, train_labels, primary_scores
):
    empty = [set() for _ in train_bank]
    base = int(_correctness(
        train_labels, choices(train_bank, primary_scores, empty)
    ).sum())
    trials = []
    trial_records = []
    cached = {}
    for alpha in ALPHAS:
        group_scores, pair_count = oof_group_scores(
            train, variant, SEEDS[0], alpha
        )
        adjustment = group_adjustment(train, group_scores)
        cached[alpha] = (group_scores, adjustment, pair_count)
        standalone = evaluate_adjustment(
            train_bank, train_labels, np.zeros_like(primary_scores), adjustment, 1.0
        )["proposed_correct"]
        for weight in FUSION_WEIGHTS:
            result = evaluate_adjustment(
                train_bank, train_labels, primary_scores, adjustment, weight
            )
            trial_records.append(
                {
                    "alpha": alpha,
                    "weight": weight,
                    "proposed_correct": result["proposed_correct"],
                    "delta": result["delta"],
                    "fixes": result["fixes"],
                    "harms": result["harms"],
                    "answer_group_changes": result["answer_group_changes"],
                    "standalone_correct": standalone,
                }
            )
            trials.append(
                (
                    result["proposed_correct"],
                    standalone,
                    -weight,
                    -alpha,
                    alpha,
                    weight,
                    pair_count,
                )
            )
    best = max(trials)
    best_nonzero = max(
        (record for record in trial_records if record["weight"] > 0),
        key=lambda record: (
            record["proposed_correct"],
            -record["weight"],
            -record["alpha"],
        ),
    )
    return {
        "alpha": best[4],
        "weight": best[5],
        "primary_base_correct": base,
        "primary_selected_correct": best[0],
        "primary_standalone_correct": best[1],
        "pair_examples_across_folds": best[6],
        "best_nonzero_trial": best_nonzero,
        "primary_trials": trial_records,
    }, cached[best[4]][1]


def main():
    started = time.perf_counter()
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels(
        "validation", validation_bank
    ).astype(bool)
    train = build_group_dataset(train_bank, train_arrays, train_labels)
    validation = build_group_dataset(validation_bank, validation_arrays)
    train_score_sets = _model_z_oof_scores(
        train_bank, train_arrays, train_labels
    )
    validation_scores = _model_z_validation_scores(
        validation_bank, validation_arrays
    )
    empty_validation = [set() for _ in validation_bank]
    validation_base_choices = choices(
        validation_bank, validation_scores, empty_validation
    )
    validation_base_correct = int(
        _correctness(validation_labels, validation_base_choices).sum()
    )
    if validation_base_correct != 768:
        raise AssertionError(
            f"Expected frozen Model Z validation baseline 768, got {validation_base_correct}"
        )

    results = {}
    for variant in VARIANTS:
        config, primary_adjustment = select_config(
            variant,
            train,
            train_bank,
            train_labels,
            train_score_sets[0],
        )
        oof_results = []
        for seed_index, (seed, base_scores) in enumerate(
            zip(SEEDS, train_score_sets)
        ):
            if seed_index == 0:
                adjustment = primary_adjustment
                pair_count = config["pair_examples_across_folds"]
            else:
                group_scores, pair_count = oof_group_scores(
                    train, variant, seed, config["alpha"]
                )
                adjustment = group_adjustment(train, group_scores)
            result = evaluate_adjustment(
                train_bank,
                train_labels,
                base_scores,
                adjustment,
                config["weight"],
            )
            result["seed"] = seed
            result["pair_examples_across_folds"] = pair_count
            oof_results.append(result)

        validation_group, validation_pairs = validation_group_scores(
            train, validation, variant, config["alpha"]
        )
        validation_adjustment = group_adjustment(
            validation, validation_group
        )
        validation_result = evaluate_adjustment(
            validation_bank,
            validation_labels,
            validation_scores,
            validation_adjustment,
            config["weight"],
        )
        validation_result["full_train_pair_examples"] = validation_pairs
        results[variant] = {
            "feature_dimension": int(train[variant].shape[1]),
            "selected_on_primary_oof": config,
            "oof": oof_results,
            "validation": validation_result,
        }
        print(
            f"variant={variant} alpha={config['alpha']:.1e} "
            f"weight={config['weight']:.3f} "
            f"oof_deltas={[item['delta'] for item in oof_results]} "
            f"validation_delta={validation_result['delta']:+d} "
            f"validation_fixes={validation_result['fixes']} "
            f"validation_harms={validation_result['harms']}",
            flush=True,
        )

    report = {
        "protocol_note": (
            "train references for question-level OOF and validation references "
            "for one frozen evaluation; test inputs/references/scores were not read"
        ),
        "frozen_baseline": "Model Z",
        "validation_baseline_correct": validation_base_correct,
        "validation_n": len(validation_bank),
        "config_selection": (
            "alpha and nonnegative fusion weight selected on seed-7506 train OOF; "
            "other four OOF schemes are robustness checks"
        ),
        "feature_design": {
            "semantic_text_hash_dimension": TEXT_DIM,
            "question_embedding_hash_dimension": QUESTION_EMBED_DIM,
            "candidate_embedding_projection_dimension": CANDIDATE_EMBED_DIM,
            "ledger_target_hash_dimension": LEDGER_TARGET_DIM,
            "ledger_feature_names": FEATURE_NAMES,
            "train_ledger_support": train["ledger_support"],
            "validation_ledger_support": validation["ledger_support"],
        },
        "results": results,
        "wall_seconds": time.perf_counter() - started,
    }
    output = CODE / "runs/diagnostics/groupwise-question-ledger.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"saved={output}")


if __name__ == "__main__":
    main()
