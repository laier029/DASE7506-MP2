"""Probe cross-candidate agreement among intermediate calculation results."""

from fractions import Fraction
from pathlib import Path
import math
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from answer_step_consistency import calculation_values
from common import load_inputs, parse_answer
from group_ranker_probe import candidate_labels, oof_model_k
from parseable_group_verifier import select_parseable_groups
from structure_text_fusion_verifier import structure_text_scores


def canonical(value):
    if not math.isfinite(value):
        return None
    if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-6):
        return float(round(value))
    return round(float(value), 4)


def agreement_features(bank):
    output = np.zeros((len(bank), 4, 8), dtype=np.float32)
    for row_index, row in enumerate(bank):
        value_sets = []
        answers = []
        for candidate in row["candidates"]:
            values = {
                normalized
                for value in calculation_values(candidate["text"])
                if (normalized := canonical(value)) is not None
            }
            value_sets.append(values)
            answer = parse_answer(candidate["text"])
            answers.append(
                canonical(float(Fraction(answer))) if answer is not None else None
            )
        for candidate_index in range(4):
            own = value_sets[candidate_index]
            peers = [
                value_sets[index] for index in range(4) if index != candidate_index
            ]
            peer_union = set().union(*peers)
            supported = own & peer_union
            peer_counts = [
                sum(value in peer for peer in peers) for value in own
            ]
            jaccards = [
                len(own & peer) / max(len(own | peer), 1) for peer in peers
            ]
            answer = answers[candidate_index]
            cross_answer_support = (
                sum(answer in peer for peer in peers) if answer is not None else 0
            )
            nonanswer = own - ({answer} if answer is not None else set())
            nonanswer_supported = nonanswer & peer_union
            output[row_index, candidate_index] = (
                len(supported) / max(len(own), 1),
                np.log1p(len(supported)),
                sum(peer_counts) / max(len(own) * 3, 1),
                max(jaccards),
                sum(jaccards) / 3,
                cross_answer_support / 3,
                float(cross_answer_support > 0),
                len(nonanswer_supported) / max(len(nonanswer), 1),
            )
    return output


def standardize_rows(values):
    return (values - values.mean(axis=1, keepdims=True)) / (
        values.std(axis=1, keepdims=True) + 1e-6
    )


def accuracy(bank, labels, scores):
    selected = np.asarray(select_parseable_groups(bank, scores, 0.3))
    return labels[np.arange(len(labels)), selected].mean()


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank).astype(bool)
    validation_labels = candidate_labels("validation", validation_bank).astype(bool)
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
    train_features = agreement_features(train_bank)
    validation_features = agreement_features(validation_bank)
    names = (
        "supported_result_fraction",
        "supported_result_count",
        "mean_peer_support",
        "maximum_peer_jaccard",
        "mean_peer_jaccard",
        "final_answer_peer_support_count",
        "final_answer_any_peer_support",
        "nonfinal_supported_fraction",
    )
    print(
        f"base_oof={100 * accuracy(train_bank, train_labels, train_scores):.4f} "
        f"base_validation={100 * accuracy(validation_bank, validation_labels, validation_scores):.4f}",
        flush=True,
    )
    for feature_index, name in enumerate(names):
        train_feature = standardize_rows(train_features[:, :, feature_index])
        validation_feature = standardize_rows(
            validation_features[:, :, feature_index]
        )
        best = (-1.0, 0.0)
        for weight in np.arange(-0.20, 0.201, 0.01):
            score = accuracy(
                train_bank,
                train_labels,
                train_scores + weight * train_feature,
            )
            if score > best[0]:
                best = (score, float(weight))
        validation_accuracy = accuracy(
            validation_bank,
            validation_labels,
            validation_scores + best[1] * validation_feature,
        )
        print(
            f"feature={name} weight={best[1]:.3f} "
            f"oof={100 * best[0]:.4f} "
            f"validation={100 * validation_accuracy:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
