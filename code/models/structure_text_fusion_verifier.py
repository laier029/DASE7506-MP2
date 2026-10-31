"""Model M: Model K plus answer-structure-specific text classifiers."""

from collections import Counter
from pathlib import Path

import joblib
import numpy as np

from answer_step_consistency import _reasoning_prefix
from common import parse_answer
from parseable_group_verifier import select_parseable_groups
from reasoning_density_verifier import reasoning_density_scores


def answer_structures(bank):
    return np.asarray(
        [
            "-".join(
                str(value)
                for value in sorted(
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


def structure_text_scores(bank, arrays, checkpoint):
    """Return Model M scores before answer-group aggregation."""
    checkpoint = Path(checkpoint)
    artifact = joblib.load(checkpoint / "structure_text_model.joblib")
    scores = reasoning_density_scores(bank, arrays, checkpoint)
    scores -= scores.mean(axis=1, keepdims=True)
    scores /= artifact["embedding_score_scale"]

    texts = [candidate["text"] for row in bank for candidate in row["candidates"]]
    if artifact.get("reasoning_only", False):
        texts = [_reasoning_prefix(text) for text in texts]
    text_features = artifact["vectorizer"].transform(texts)
    structures = answer_structures(bank)
    for structure, classifier in artifact["classifiers"].items():
        rows = structures == structure
        if not rows.any():
            continue
        candidates = np.repeat(rows, 4)
        text_scores = classifier.decision_function(
            text_features[candidates]
        ).reshape(-1, 4)
        text_scores -= text_scores.mean(axis=1, keepdims=True)
        text_scores /= artifact["text_score_scales"][structure]
        scores[rows] += artifact["structure_weights"][structure] * text_scores
    return scores


def predict(bank, arrays, checkpoint):
    scores = structure_text_scores(bank, arrays, checkpoint)
    checkpoint = Path(checkpoint)
    with np.load(checkpoint / "model.npz", allow_pickle=False) as model:
        group_mean_weight = float(model["group_mean_weight"].item())
    return select_parseable_groups(bank, scores, group_mean_weight)
