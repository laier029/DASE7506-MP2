"""Probe lexical and TF-IDF overlap between each question and candidate."""

from collections import Counter
from pathlib import Path
import re
import sys

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))
sys.path.insert(0, str(CODE / "experiments"))

from common import load_inputs
from parseable_group_verifier import select_parseable_groups
from text_fusion_verifier import text_fusion_scores
from question_number_coverage_probe import candidate_labels, oof_l_scores


TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|(?:\d+(?:\.\d+)?|\.\d+)")
STOP = set(ENGLISH_STOP_WORDS) | {
    "calculate",
    "determine",
    "find",
    "given",
    "step",
    "answer",
    "final",
}


def tokens(text, keep_stopwords=False):
    values = TOKEN.findall(text.lower())
    if keep_stopwords:
        return values
    return [value for value in values if value not in STOP and len(value) > 1]


def recall(question_items, candidate_items):
    if not question_items:
        return 0.0
    question_counts = Counter(question_items)
    candidate_counts = Counter(candidate_items)
    return sum(
        min(count, candidate_counts[value])
        for value, count in question_counts.items()
    ) / sum(question_counts.values())


def overlap_features(bank):
    output = np.zeros((len(bank), 4, 5), dtype=np.float32)
    for row_index, row in enumerate(bank):
        q_content = tokens(row["question"])
        q_all = tokens(row["question"], keep_stopwords=True)
        q_bigrams = list(zip(q_content, q_content[1:]))
        q_unique = set(q_content)
        for candidate_index, candidate in enumerate(row["candidates"]):
            c_content = tokens(candidate["text"])
            c_all = tokens(candidate["text"], keep_stopwords=True)
            c_bigrams = list(zip(c_content, c_content[1:]))
            c_unique = set(c_content)
            union = q_unique | c_unique
            output[row_index, candidate_index] = (
                recall(q_content, c_content),
                recall(q_all, c_all),
                recall(q_bigrams, c_bigrams),
                len(q_unique & c_unique) / max(len(q_unique), 1),
                len(q_unique & c_unique) / max(len(union), 1),
            )
    return output


def tfidf_similarity(train_bank, validation_bank):
    train_questions = [row["question"] for row in train_bank]
    validation_questions = [row["question"] for row in validation_bank]
    train_candidates = [
        candidate["text"]
        for row in train_bank
        for candidate in row["candidates"]
    ]
    validation_candidates = [
        candidate["text"]
        for row in validation_bank
        for candidate in row["candidates"]
    ]
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=3,
        max_features=25000,
        stop_words="english",
        sublinear_tf=True,
        dtype=np.float32,
    )
    vectorizer.fit(train_questions + train_candidates)

    def similarities(questions, candidates):
        question_x = vectorizer.transform(questions)
        candidate_x = vectorizer.transform(candidates)
        output = np.empty((len(questions), 4), dtype=np.float32)
        for candidate_index in range(4):
            output[:, candidate_index] = np.asarray(
                question_x.multiply(candidate_x[candidate_index::4]).sum(axis=1)
            ).ravel()
        return output

    return similarities(train_questions, train_candidates), similarities(
        validation_questions, validation_candidates
    )


def standardize_rows(values):
    return (values - values.mean(axis=1, keepdims=True)) / (
        values.std(axis=1, keepdims=True) + 1e-6
    )


def main():
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_labels = candidate_labels("train", train_bank)
    validation_labels = candidate_labels("validation", validation_bank)
    folds = np.arange(len(train_bank)) % 5
    train_scores, _ = oof_l_scores(
        train_bank, train_arrays, train_labels, folds
    )
    validation_scores = text_fusion_scores(
        validation_bank,
        validation_arrays,
        CODE / "runs/model_l_text_fusion",
    )
    train_overlap = overlap_features(train_bank)
    validation_overlap = overlap_features(validation_bank)
    train_similarity, validation_similarity = tfidf_similarity(
        train_bank, validation_bank
    )
    train_overlap = np.concatenate(
        [train_overlap, train_similarity[:, :, None]], axis=2
    )
    validation_overlap = np.concatenate(
        [validation_overlap, validation_similarity[:, :, None]], axis=2
    )
    names = (
        "content_multiset_recall",
        "all_token_multiset_recall",
        "content_bigram_recall",
        "content_unique_recall",
        "content_jaccard",
        "question_candidate_tfidf_cosine",
    )
    for feature_index, name in enumerate(names):
        train_feature = standardize_rows(train_overlap[:, :, feature_index])
        validation_feature = standardize_rows(
            validation_overlap[:, :, feature_index]
        )
        best = (-1.0, 0.0)
        for weight in np.arange(-0.20, 0.201, 0.01):
            selected = np.asarray(
                select_parseable_groups(
                    train_bank,
                    train_scores + weight * train_feature,
                    0.3,
                )
            )
            accuracy = train_labels[
                np.arange(len(train_labels)), selected
            ].mean()
            if accuracy > best[0]:
                best = (accuracy, float(weight))
        selected = np.asarray(
            select_parseable_groups(
                validation_bank,
                validation_scores + best[1] * validation_feature,
                0.3,
            )
        )
        validation_accuracy = validation_labels[
            np.arange(len(validation_labels)), selected
        ].mean()
        print(
            f"feature={name} oof={100 * best[0]:.4f} "
            f"weight={best[1]:.3f} validation={100 * validation_accuracy:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
