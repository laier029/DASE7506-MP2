"""Validate the direction of the character TF-IDF fusion with OOF scores."""

from collections import defaultdict
from collections import Counter
from fractions import Fraction
import math
from pathlib import Path
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))

from common import DATA, load_inputs, parse_answer, read_jsonl
from answer_step_consistency import last_step_value
from linear_verifier import linear_scores
from parseable_group_verifier import select_parseable_groups
from reasoning_density_verifier import reasoning_density_scores


def select_with_cross_last(bank, scores, cross_weight):
    selected = []
    for row, row_scores in zip(bank, scores):
        answers = [parse_answer(candidate["text"]) for candidate in row["candidates"]]
        last_values = [last_step_value(candidate["text"]) for candidate in row["candidates"]]
        groups = defaultdict(list)
        for index, answer in enumerate(answers):
            if answer is not None:
                groups[answer].append(index)
        ranked = []
        for answer, indices in groups.items():
            answer_value = float(Fraction(answer))
            cross_support = sum(
                answers[index] != answer
                and last_values[index] is not None
                and math.isclose(
                    answer_value, last_values[index], rel_tol=1e-6, abs_tol=1e-6
                )
                for index in range(4)
            ) / 4
            values = row_scores[indices]
            candidate = indices[int(values.argmax())]
            group_score = (
                float(values.max() + 0.3 * values.mean())
                + cross_weight * cross_support
            )
            ranked.append((group_score, -candidate, candidate))
        selected.append(max(ranked)[2])
    return np.asarray(selected)


def main():
    bank, arrays = load_inputs("train")
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    labels = np.asarray(
        [
            [
                parse_answer(candidate["text"]) == parse_answer(reference["answer"])
                for candidate in row["candidates"]
            ]
            for row, reference in zip(bank, references)
        ]
    )
    texts = [candidate["text"] for row in bank for candidate in row["candidates"]]
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=5,
        max_features=50000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    text_features = vectorizer.fit_transform(texts)
    fold_ids = np.arange(len(bank)) % 5
    text_oof = np.empty(len(texts), dtype=np.float32)
    embedding_oof = np.empty((len(bank), 4), dtype=np.float32)
    embeddings = arrays["embeddings"].astype(np.float32)
    embeddings -= embeddings.mean(axis=1, keepdims=True)

    for fold in range(5):
        row_train = fold_ids != fold
        row_val = ~row_train
        candidate_train = np.repeat(row_train, 4)
        candidate_val = np.repeat(row_val, 4)
        classifier = LinearSVC(C=0.1, dual="auto", random_state=7506).fit(
            text_features[candidate_train], labels.reshape(-1)[candidate_train]
        )
        text_oof[candidate_val] = classifier.decision_function(
            text_features[candidate_val]
        )

        fit_x = embeddings[row_train].reshape(-1, 1024)
        mean = fit_x.mean(axis=0)
        scale = fit_x.std(axis=0)
        scale[scale < 1e-6] = 1.0
        fit_x = (fit_x - mean) / scale
        fit_y = labels[row_train].reshape(-1).astype(np.float32)
        gram = fit_x.T @ fit_x / len(fit_x)
        target = fit_x.T @ fit_y / len(fit_x)
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        weights = eigenvectors @ (
            (eigenvectors.T @ target) / (eigenvalues + 0.3)
        )
        embedding_oof[row_val] = (
            (embeddings[row_val] - mean) / scale
        ) @ weights

    full_embedding = linear_scores(
        bank, arrays, CODE / "runs/model_k_reasoning_density", relative=True
    )
    model_k = reasoning_density_scores(
        bank, arrays, CODE / "runs/model_k_reasoning_density"
    )
    embedding_oof += model_k - full_embedding
    text_oof = text_oof.reshape(-1, 4)
    embedding_oof -= embedding_oof.mean(axis=1, keepdims=True)
    text_oof -= text_oof.mean(axis=1, keepdims=True)
    embedding_oof /= embedding_oof.std()
    text_oof /= text_oof.std()

    for weight in np.arange(-0.2, 0.201, 0.025):
        selected = np.asarray(
            select_parseable_groups(
                bank, embedding_oof + weight * text_oof, 0.3
            )
        )
        accuracy = labels[np.arange(len(labels)), selected].mean()
        print(f"weight={weight:.3f} oof={100 * accuracy:.4f}", flush=True)

    fused = embedding_oof - 0.05 * text_oof
    structures = [
        tuple(
            sorted(
                Counter(
                    parse_answer(candidate["text"])
                    for candidate in row["candidates"]
                ).values(),
                reverse=True,
            )
        )
        for row in bank
    ]
    base_selected = np.asarray(
        select_parseable_groups(bank, embedding_oof, 0.3)
    )
    fused_selected = np.asarray(select_parseable_groups(bank, fused, 0.3))
    for structure in sorted(set(structures), reverse=True):
        mask = np.asarray([value == structure for value in structures])
        base_correct = labels[mask][
            np.arange(mask.sum()), base_selected[mask]
        ]
        fused_correct = labels[mask][
            np.arange(mask.sum()), fused_selected[mask]
        ]
        changed = base_selected[mask] != fused_selected[mask]
        wrong_right = ((~base_correct) & fused_correct & changed).sum()
        right_wrong = (base_correct & (~fused_correct) & changed).sum()
        print(
            f"structure={structure} n={mask.sum()} "
            f"base={100 * base_correct.mean():.4f} "
            f"fused={100 * fused_correct.mean():.4f} "
            f"changed={changed.sum()} wr={wrong_right} rw={right_wrong}",
            flush=True,
        )
        best_structure = (base_correct.mean(), 0.0)
        structure_bank = [row for row, keep in zip(bank, mask) if keep]
        for weight in np.arange(-0.2, 0.201, 0.025):
            structure_selected = np.asarray(
                select_parseable_groups(
                    structure_bank,
                    embedding_oof[mask] + weight * text_oof[mask],
                    0.3,
                )
            )
            structure_accuracy = labels[mask][
                np.arange(mask.sum()), structure_selected
            ].mean()
            if structure_accuracy > best_structure[0]:
                best_structure = (structure_accuracy, weight)
        print(
            f"structure_best={structure} "
            f"oof={100 * best_structure[0]:.4f} weight={best_structure[1]:.3f}",
            flush=True,
        )
    for cross_weight in np.arange(0.0, 0.601, 0.05):
        selected = select_with_cross_last(bank, fused, cross_weight)
        accuracy = labels[np.arange(len(labels)), selected].mean()
        print(
            f"cross_weight={cross_weight:.3f} oof={100 * accuracy:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
