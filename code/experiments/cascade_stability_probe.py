"""Check whether a precision-first negative gate survives random OOF folds."""

from collections import Counter
from collections import defaultdict
from fractions import Fraction
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

from answer_step_consistency import _reasoning_prefix, answer_step_consistency
from arithmetic_consistency import (
    arithmetic_counts,
    broad_arithmetic_counts,
    broad_arithmetic_mismatches,
)
from cascade_negative_probe import (
    cascade_select,
    negative_signals,
    rule_statistics,
    selection_accuracy,
)
from common import load_inputs, parse_answer
from group_ranker_probe import candidate_labels, oof_model_k
from parseable_group_verifier import select_parseable_groups
from structure_text_fusion_verifier import structure_text_scores


STRUCTURE_WEIGHTS = {
    "4": 0.0,
    "3-1": 0.025,
    "2-2": -0.225,
    "2-1-1": -0.10,
    "1-1-1-1": -0.10,
}

CORRECTION_CUE = re.compile(
    r"(?i)\b(?:wait|re-?check|mistake|miscalculation|correction|"
    r"different approach|another approach|"
    r"let(?:'s| us) (?:recalculate|check|try)|"
    r"i (?:made an error|need to correct))\b"
)
NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


def answer_structures(bank):
    return np.asarray(
        [
            "-".join(
                str(count)
                for count in sorted(
                    Counter(
                        parse_answer(candidate["text"])
                        for candidate in row["candidates"]
                    ).values(),
                    reverse=True,
                )
            )
            for row in bank
        ]
    )


def balanced_random_folds(size, seed, fold_count=5):
    permutation = np.random.default_rng(seed).permutation(size)
    folds = np.empty(size, dtype=np.int8)
    folds[permutation] = np.arange(size) % fold_count
    return folds


def model_m_oof(bank, arrays, labels, folds, text_features, structures):
    scores = oof_model_k(bank, arrays, labels, folds)
    scores -= scores.mean(axis=1, keepdims=True)
    scores /= float(scores.std())

    text_oof = np.zeros(labels.shape, dtype=np.float32)
    flat_labels = labels.reshape(-1)
    for structure, weight in STRUCTURE_WEIGHTS.items():
        if weight == 0.0:
            continue
        for fold in range(folds.max() + 1):
            fit_rows = (folds != fold) & (structures == structure)
            held_rows = (folds == fold) & (structures == structure)
            if not held_rows.any():
                continue
            fit = np.repeat(fit_rows, 4)
            held = np.repeat(held_rows, 4)
            classifier = LinearSVC(
                C=0.1, dual="auto", random_state=7506
            ).fit(text_features[fit], flat_labels[fit])
            text_oof[held_rows] = classifier.decision_function(
                text_features[held]
            ).reshape(-1, 4)

    text_oof -= text_oof.mean(axis=1, keepdims=True)
    for structure, weight in STRUCTURE_WEIGHTS.items():
        rows = structures == structure
        if weight == 0.0 or not rows.any():
            continue
        scale = float(text_oof[rows].std())
        scores[rows] += weight * text_oof[rows] / max(scale, 1e-6)
    return scores


def soft_select(bank, scores, gate, penalty):
    adjusted = scores.copy()
    adjusted[gate] -= penalty
    return np.asarray(select_parseable_groups(bank, adjusted, 0.3))


def unanimous_answer_group_gate(bank, candidate_gate):
    """Reject an answer only when all candidates supporting it are flagged."""
    group_gate = np.zeros_like(candidate_gate)
    for row_index, row in enumerate(bank):
        groups = defaultdict(list)
        for candidate_index, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate["text"])
            if answer is not None:
                groups[answer].append(candidate_index)
        for indices in groups.values():
            if candidate_gate[row_index, indices].all():
                group_gate[row_index, indices] = True
    return group_gate


def correction_aware_gates(bank):
    """Return narrow and broad-only errors remaining after the last repair cue."""
    narrow_gate = np.zeros((len(bank), 4), dtype=bool)
    broad_only_gate = np.zeros((len(bank), 4), dtype=bool)
    for row_index, row in enumerate(bank):
        for candidate_index, candidate in enumerate(row["candidates"]):
            text = candidate["text"]
            matches = list(CORRECTION_CUE.finditer(text))
            checked_text = text[matches[-1].end() :] if matches else text
            narrow_bad = arithmetic_counts(checked_text)[1]
            broad_bad = broad_arithmetic_counts(checked_text)[1]
            narrow_gate[row_index, candidate_index] = narrow_bad > 0
            broad_only_gate[row_index, candidate_index] = (
                broad_bad > 0 and narrow_bad == 0
            )
    return narrow_gate, broad_only_gate


def unsupported_step_mismatch_gate(bank):
    """Keep only step mismatches whose answer is not stated after the equation."""
    gate = np.zeros((len(bank), 4), dtype=bool)
    for row_index, row in enumerate(bank):
        for candidate_index, candidate in enumerate(row["candidates"]):
            text = candidate["text"]
            if answer_step_consistency(text) is not False:
                continue
            answer = parse_answer(text)
            reasoning = _reasoning_prefix(text)
            suffix = reasoning[reasoning.rfind("=") + 1 :]
            mentioned = False
            for match in NUMBER.finditer(suffix):
                try:
                    mentioned = Fraction(match.group().replace(",", "")) == Fraction(
                        answer
                    )
                except (ValueError, ZeroDivisionError):
                    continue
                if mentioned:
                    break
            gate[row_index, candidate_index] = not mentioned
    return gate


def repeated_mismatch_group_gate(bank, *, broad_only=False):
    """Reject an answer group when two traces repeat one numeric mismatch."""
    gate = np.zeros((len(bank), 4), dtype=bool)
    for row_index, row in enumerate(bank):
        groups = defaultdict(list)
        mismatch_sets = []
        for candidate_index, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate["text"])
            if answer is not None:
                groups[answer].append(candidate_index)
            text = candidate["text"]
            matches = list(CORRECTION_CUE.finditer(text))
            checked_text = text[matches[-1].end() :] if matches else text
            if broad_only and arithmetic_counts(checked_text)[1] > 0:
                mismatch_sets.append(set())
            else:
                mismatch_sets.append(
                    set(broad_arithmetic_mismatches(checked_text))
                )
        for indices in groups.values():
            repeated = set()
            for position, left_index in enumerate(indices):
                for right_index in indices[position + 1 :]:
                    repeated |= mismatch_sets[left_index] & mismatch_sets[right_index]
            if repeated:
                gate[row_index, indices] = True
    return gate


def unanimous_mismatch_group_gate(bank, *, broad_only=False):
    """Reject only when every trace for an answer repeats one mismatch."""
    gate = np.zeros((len(bank), 4), dtype=bool)
    for row_index, row in enumerate(bank):
        groups = defaultdict(list)
        mismatch_sets = []
        for candidate_index, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate["text"])
            if answer is not None:
                groups[answer].append(candidate_index)
            text = candidate["text"]
            matches = list(CORRECTION_CUE.finditer(text))
            checked_text = text[matches[-1].end() :] if matches else text
            if broad_only and arithmetic_counts(checked_text)[1] > 0:
                mismatch_sets.append(set())
            else:
                mismatch_sets.append(
                    set(broad_arithmetic_mismatches(checked_text))
                )
        for indices in groups.values():
            if len(indices) < 2:
                continue
            shared = set.intersection(*(mismatch_sets[index] for index in indices))
            if shared:
                gate[row_index, indices] = True
    return gate


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
    _, train_signals = negative_signals(train_bank)
    _, validation_signals = negative_signals(validation_bank)
    train_gate = train_signals[:, :, 2]
    validation_gate = validation_signals[:, :, 2]
    train_repaired_narrow, train_repaired_broad = correction_aware_gates(train_bank)
    validation_repaired_narrow, validation_repaired_broad = correction_aware_gates(
        validation_bank
    )
    train_step_unsupported = unsupported_step_mismatch_gate(train_bank)
    validation_step_unsupported = unsupported_step_mismatch_gate(validation_bank)
    train_repeated_mismatch = repeated_mismatch_group_gate(train_bank)
    validation_repeated_mismatch = repeated_mismatch_group_gate(validation_bank)
    train_repeated_broad = repeated_mismatch_group_gate(
        train_bank, broad_only=True
    )
    validation_repeated_broad = repeated_mismatch_group_gate(
        validation_bank, broad_only=True
    )
    train_unanimous_mismatch = unanimous_mismatch_group_gate(train_bank)
    validation_unanimous_mismatch = unanimous_mismatch_group_gate(
        validation_bank
    )
    train_unanimous_broad = unanimous_mismatch_group_gate(
        train_bank, broad_only=True
    )
    validation_unanimous_broad = unanimous_mismatch_group_gate(
        validation_bank, broad_only=True
    )
    train_group_gate = unanimous_answer_group_gate(train_bank, train_gate)
    validation_group_gate = unanimous_answer_group_gate(
        validation_bank, validation_gate
    )
    print(
        "train repaired_broad",
        rule_statistics("repaired_broad", train_repaired_broad, train_labels),
        flush=True,
    )
    print(
        "validation repaired_broad",
        rule_statistics(
            "repaired_broad", validation_repaired_broad, validation_labels
        ),
        flush=True,
    )
    print(
        "train repaired_narrow",
        rule_statistics("repaired_narrow", train_repaired_narrow, train_labels),
        flush=True,
    )
    print(
        "validation repaired_narrow",
        rule_statistics(
            "repaired_narrow", validation_repaired_narrow, validation_labels
        ),
        flush=True,
    )
    print(
        "train unsupported_step",
        rule_statistics("unsupported_step", train_step_unsupported, train_labels),
        flush=True,
    )
    print(
        "validation unsupported_step",
        rule_statistics(
            "unsupported_step", validation_step_unsupported, validation_labels
        ),
        flush=True,
    )
    print(
        "train repeated_mismatch",
        rule_statistics(
            "repeated_mismatch", train_repeated_mismatch, train_labels
        ),
        flush=True,
    )
    print(
        "validation repeated_mismatch",
        rule_statistics(
            "repeated_mismatch", validation_repeated_mismatch, validation_labels
        ),
        flush=True,
    )
    print(
        "train repeated_broad",
        rule_statistics("repeated_broad", train_repeated_broad, train_labels),
        flush=True,
    )
    print(
        "validation repeated_broad",
        rule_statistics(
            "repeated_broad", validation_repeated_broad, validation_labels
        ),
        flush=True,
    )
    for label, train_values, validation_values in (
        (
            "unanimous_mismatch",
            train_unanimous_mismatch,
            validation_unanimous_mismatch,
        ),
        ("unanimous_broad", train_unanimous_broad, validation_unanimous_broad),
    ):
        print(
            "train", label,
            rule_statistics(label, train_values, train_labels),
            flush=True,
        )
        print(
            "validation", label,
            rule_statistics(label, validation_values, validation_labels),
            flush=True,
        )

    # Match the frozen Model M artifact, which was trained on full candidate
    # text (the separate reasoning-only probe was not promoted).
    texts = [
        candidate["text"]
        for row in train_bank
        for candidate in row["candidates"]
    ]
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=5,
        max_features=25000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    text_features = vectorizer.fit_transform(texts)
    structures = answer_structures(train_bank)

    penalties = (0.025, 0.05, 0.10, 0.20, 0.40, 0.80)
    for seed in (7506, 11, 29, 47, 83):
        folds = (
            np.arange(len(train_bank)) % 5
            if seed == 7506
            else balanced_random_folds(len(train_bank), seed)
        )
        scores = model_m_oof(
            train_bank,
            train_arrays,
            train_labels,
            folds,
            text_features,
            structures,
        )
        base = np.asarray(select_parseable_groups(train_bank, scores, 0.3))
        fields = [
            f"seed={seed}",
            f"base={100 * selection_accuracy(train_labels, base):.4f}",
        ]
        for penalty in penalties:
            selected = soft_select(train_bank, scores, train_gate, penalty)
            delta = int(train_labels[np.arange(len(train_labels)), selected].sum()) - int(
                train_labels[np.arange(len(train_labels)), base].sum()
            )
            fields.append(f"p{penalty:g}={delta:+d}")
        hard = cascade_select(train_bank, scores, train_gate, rescue="global")
        hard_delta = int(train_labels[np.arange(len(train_labels)), hard].sum()) - int(
            train_labels[np.arange(len(train_labels)), base].sum()
        )
        fields.append(f"hard={hard_delta:+d}")
        group_hard = cascade_select(
            train_bank, scores, train_group_gate, rescue="global"
        )
        group_hard_delta = int(
            train_labels[np.arange(len(train_labels)), group_hard].sum()
        ) - int(train_labels[np.arange(len(train_labels)), base].sum())
        fields.append(f"group_hard={group_hard_delta:+d}")
        repaired_hard = cascade_select(
            train_bank, scores, train_repaired_broad, rescue="global"
        )
        repaired_hard_delta = int(
            train_labels[np.arange(len(train_labels)), repaired_hard].sum()
        ) - int(train_labels[np.arange(len(train_labels)), base].sum())
        fields.append(f"repaired_hard={repaired_hard_delta:+d}")
        repaired_union = cascade_select(
            train_bank,
            scores,
            train_repaired_broad | train_repaired_narrow,
            rescue="global",
        )
        repaired_union_delta = int(
            train_labels[np.arange(len(train_labels)), repaired_union].sum()
        ) - int(train_labels[np.arange(len(train_labels)), base].sum())
        fields.append(f"repaired_union={repaired_union_delta:+d}")
        step_hard = cascade_select(
            train_bank, scores, train_step_unsupported, rescue="global"
        )
        step_hard_delta = int(
            train_labels[np.arange(len(train_labels)), step_hard].sum()
        ) - int(train_labels[np.arange(len(train_labels)), base].sum())
        fields.append(f"step_hard={step_hard_delta:+d}")
        repeated_hard = cascade_select(
            train_bank, scores, train_repeated_mismatch, rescue="global"
        )
        repeated_hard_delta = int(
            train_labels[np.arange(len(train_labels)), repeated_hard].sum()
        ) - int(train_labels[np.arange(len(train_labels)), base].sum())
        fields.append(f"repeated_hard={repeated_hard_delta:+d}")
        repeated_broad_hard = cascade_select(
            train_bank, scores, train_repeated_broad, rescue="global"
        )
        repeated_broad_delta = int(
            train_labels[
                np.arange(len(train_labels)), repeated_broad_hard
            ].sum()
        ) - int(train_labels[np.arange(len(train_labels)), base].sum())
        fields.append(f"repeated_broad={repeated_broad_delta:+d}")
        cascade_union = cascade_select(
            train_bank,
            scores,
            train_repaired_broad | train_repeated_broad,
            rescue="global",
        )
        cascade_union_delta = int(
            train_labels[np.arange(len(train_labels)), cascade_union].sum()
        ) - int(train_labels[np.arange(len(train_labels)), base].sum())
        fields.append(f"cascade_union={cascade_union_delta:+d}")
        unanimous = cascade_select(
            train_bank, scores, train_unanimous_mismatch, rescue="global"
        )
        unanimous_delta = int(
            train_labels[np.arange(len(train_labels)), unanimous].sum()
        ) - int(train_labels[np.arange(len(train_labels)), base].sum())
        fields.append(f"unanimous={unanimous_delta:+d}")
        unanimous_broad = cascade_select(
            train_bank, scores, train_unanimous_broad, rescue="global"
        )
        unanimous_broad_delta = int(
            train_labels[np.arange(len(train_labels)), unanimous_broad].sum()
        ) - int(train_labels[np.arange(len(train_labels)), base].sum())
        fields.append(f"unanimous_broad={unanimous_broad_delta:+d}")
        print(" ".join(fields), flush=True)

    validation_scores = structure_text_scores(
        validation_bank,
        validation_arrays,
        CODE / "runs/model_m_structure_text",
    )
    validation_base = np.asarray(
        select_parseable_groups(validation_bank, validation_scores, 0.3)
    )
    fields = [
        "validation",
        f"base={100 * selection_accuracy(validation_labels, validation_base):.4f}",
    ]
    for penalty in penalties:
        selected = soft_select(
            validation_bank, validation_scores, validation_gate, penalty
        )
        delta = int(
            validation_labels[np.arange(len(validation_labels)), selected].sum()
        ) - int(
            validation_labels[np.arange(len(validation_labels)), validation_base].sum()
        )
        fields.append(f"p{penalty:g}={delta:+d}")
    hard = cascade_select(
        validation_bank, validation_scores, validation_gate, rescue="global"
    )
    hard_delta = int(
        validation_labels[np.arange(len(validation_labels)), hard].sum()
    ) - int(
        validation_labels[np.arange(len(validation_labels)), validation_base].sum()
    )
    fields.append(f"hard={hard_delta:+d}")
    group_hard = cascade_select(
        validation_bank,
        validation_scores,
        validation_group_gate,
        rescue="global",
    )
    group_hard_delta = int(
        validation_labels[np.arange(len(validation_labels)), group_hard].sum()
    ) - int(
        validation_labels[np.arange(len(validation_labels)), validation_base].sum()
    )
    fields.append(f"group_hard={group_hard_delta:+d}")
    repaired_hard = cascade_select(
        validation_bank,
        validation_scores,
        validation_repaired_broad,
        rescue="global",
    )
    repaired_hard_delta = int(
        validation_labels[np.arange(len(validation_labels)), repaired_hard].sum()
    ) - int(
        validation_labels[np.arange(len(validation_labels)), validation_base].sum()
    )
    fields.append(f"repaired_hard={repaired_hard_delta:+d}")
    repaired_union = cascade_select(
        validation_bank,
        validation_scores,
        validation_repaired_broad | validation_repaired_narrow,
        rescue="global",
    )
    repaired_union_delta = int(
        validation_labels[np.arange(len(validation_labels)), repaired_union].sum()
    ) - int(
        validation_labels[np.arange(len(validation_labels)), validation_base].sum()
    )
    fields.append(f"repaired_union={repaired_union_delta:+d}")
    step_hard = cascade_select(
        validation_bank,
        validation_scores,
        validation_step_unsupported,
        rescue="global",
    )
    step_hard_delta = int(
        validation_labels[np.arange(len(validation_labels)), step_hard].sum()
    ) - int(
        validation_labels[np.arange(len(validation_labels)), validation_base].sum()
    )
    fields.append(f"step_hard={step_hard_delta:+d}")
    repeated_hard = cascade_select(
        validation_bank,
        validation_scores,
        validation_repeated_mismatch,
        rescue="global",
    )
    repeated_hard_delta = int(
        validation_labels[np.arange(len(validation_labels)), repeated_hard].sum()
    ) - int(
        validation_labels[np.arange(len(validation_labels)), validation_base].sum()
    )
    fields.append(f"repeated_hard={repeated_hard_delta:+d}")
    repeated_broad_hard = cascade_select(
        validation_bank,
        validation_scores,
        validation_repeated_broad,
        rescue="global",
    )
    repeated_broad_delta = int(
        validation_labels[
            np.arange(len(validation_labels)), repeated_broad_hard
        ].sum()
    ) - int(
        validation_labels[np.arange(len(validation_labels)), validation_base].sum()
    )
    fields.append(f"repeated_broad={repeated_broad_delta:+d}")
    cascade_union = cascade_select(
        validation_bank,
        validation_scores,
        validation_repaired_broad | validation_repeated_broad,
        rescue="global",
    )
    cascade_union_delta = int(
        validation_labels[np.arange(len(validation_labels)), cascade_union].sum()
    ) - int(
        validation_labels[np.arange(len(validation_labels)), validation_base].sum()
    )
    fields.append(f"cascade_union={cascade_union_delta:+d}")
    unanimous = cascade_select(
        validation_bank,
        validation_scores,
        validation_unanimous_mismatch,
        rescue="global",
    )
    unanimous_delta = int(
        validation_labels[np.arange(len(validation_labels)), unanimous].sum()
    ) - int(
        validation_labels[np.arange(len(validation_labels)), validation_base].sum()
    )
    fields.append(f"unanimous={unanimous_delta:+d}")
    unanimous_broad = cascade_select(
        validation_bank,
        validation_scores,
        validation_unanimous_broad,
        rescue="global",
    )
    unanimous_broad_delta = int(
        validation_labels[
            np.arange(len(validation_labels)), unanimous_broad
        ].sum()
    ) - int(
        validation_labels[np.arange(len(validation_labels)), validation_base].sum()
    )
    fields.append(f"unanimous_broad={unanimous_broad_delta:+d}")
    print(" ".join(fields), flush=True)


if __name__ == "__main__":
    main()
