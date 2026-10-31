"""Model L: Model K fused with a compact character TF-IDF signal."""

from pathlib import Path
from collections import Counter

import joblib
import numpy as np

from common import parse_answer
from parseable_group_verifier import select_parseable_groups
from reasoning_density_verifier import reasoning_density_scores


def text_fusion_scores(bank, arrays, checkpoint):
    """Return Model L scores before answer-group aggregation."""
    checkpoint = Path(checkpoint)
    model_k_scores = reasoning_density_scores(bank, arrays, checkpoint)
    artifact = joblib.load(checkpoint / "text_model.joblib")
    texts = [candidate["text"] for row in bank for candidate in row["candidates"]]
    text_features = artifact["vectorizer"].transform(texts)
    text_scores = artifact["classifier"].decision_function(text_features)
    text_scores = text_scores.reshape(len(bank), 4)

    model_k_scores -= model_k_scores.mean(axis=1, keepdims=True)
    text_scores -= text_scores.mean(axis=1, keepdims=True)
    scores = model_k_scores / artifact["embedding_score_scale"]
    normalized_text = text_scores / artifact["text_score_scale"]
    structure_weights = artifact.get("structure_weights")
    if structure_weights:
        for row_index, row in enumerate(bank):
            structure = tuple(
                sorted(
                    Counter(
                        parse_answer(candidate["text"])
                        for candidate in row["candidates"]
                    ).values(),
                    reverse=True,
                )
            )
            key = "-".join(str(value) for value in structure)
            weight = structure_weights.get(key, artifact["text_weight"])
            scores[row_index] += weight * normalized_text[row_index]
    else:
        scores += artifact["text_weight"] * normalized_text

    return scores


def predict(bank, arrays, checkpoint):
    scores = text_fusion_scores(bank, arrays, checkpoint)
    checkpoint = Path(checkpoint)
    with np.load(checkpoint / "model.npz", allow_pickle=False) as model:
        group_mean_weight = float(model["group_mean_weight"].item())
    return select_parseable_groups(bank, scores, group_mean_weight)
