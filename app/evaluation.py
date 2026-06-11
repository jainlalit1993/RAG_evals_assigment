from __future__ import annotations

from difflib import SequenceMatcher
from math import log2
from typing import List


def _tokenize(text: str) -> List[str]:
    return [token.strip().lower() for token in text.split() if token.strip()]


def rouge_n(reference: str, prediction: str, n: int) -> float:
    ref_tokens = _tokenize(reference)
    pred_tokens = _tokenize(prediction)
    if not ref_tokens or not pred_tokens or len(pred_tokens) < n:
        return 0.0

    ref_ngrams = {}
    for i in range(len(ref_tokens) - n + 1):
        ngram = tuple(ref_tokens[i:i+n])
        ref_ngrams[ngram] = ref_ngrams.get(ngram, 0) + 1

    match_count = 0
    for j in range(len(pred_tokens) - n + 1):
        ngram = tuple(pred_tokens[j:j+n])
        if ref_ngrams.get(ngram, 0) > 0:
            match_count += 1
            ref_ngrams[ngram] -= 1

    possible_matches = max(len(pred_tokens) - n + 1, 1)
    return round(match_count / possible_matches, 4)


def rouge_l(reference: str, prediction: str) -> float:
    ref_tokens = _tokenize(reference)
    pred_tokens = _tokenize(prediction)
    if not ref_tokens or not pred_tokens:
        return 0.0

    # Compute longest common subsequence (LCS)
    len_ref = len(ref_tokens)
    len_pred = len(pred_tokens)
    dp = [[0] * (len_pred + 1) for _ in range(len_ref + 1)]

    for i in range(len_ref - 1, -1, -1):
        for j in range(len_pred - 1, -1, -1):
            if ref_tokens[i] == pred_tokens[j]:
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])

    lcs_length = dp[0][0]
    if lcs_length == 0:
        return 0.0

    precision = lcs_length / len_pred
    recall = lcs_length / len_ref
    if precision + recall == 0:
        return 0.0
    return round((2 * precision * recall) / (precision + recall), 4)


def fuzzy_similarity(reference: str, prediction: str) -> float:
    if not reference or not prediction:
        return 0.0
    matcher = SequenceMatcher(None, reference.lower(), prediction.lower())
    return round(matcher.ratio() * 100, 2)


def normalize_source(source: str) -> str:
    return source.strip().lower()


def compute_search_metrics(retrieved_sources: List[str], expected_sources: List[str]) -> dict:
    expected_set = {normalize_source(src) for src in expected_sources}
    retrieved_norm = [normalize_source(src) for src in retrieved_sources]

    relevance = [1 if src in expected_set else 0 for src in retrieved_norm]
    total_relevant = len(expected_set)

    # MRR
    first_relevant = next((i + 1 for i, rel in enumerate(relevance) if rel == 1), None)
    mrr = 1 / first_relevant if first_relevant else 0.0

    # Precision / Recall / F1
    retrieved_positive = len(relevance)
    true_positives = sum(relevance)
    precision = true_positives / retrieved_positive if retrieved_positive else 0.0
    recall = true_positives / total_relevant if total_relevant else 0.0
    f1 = 0.0
    if precision + recall:
        f1 = 2 * precision * recall / (precision + recall)

    # NDCG
    dcg = 0.0
    idcg = 0.0
    for i, rel in enumerate(relevance):
        dcg += rel / log2(i + 2)
    for i in range(min(total_relevant, len(relevance))):
        idcg += 1 / log2(i + 2)

    ndcg = round(dcg / idcg, 4) if idcg else 0.0

    return {
        "mrr": round(mrr, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "ndcg": ndcg,
        "relevance": relevance,
    }


def compute_answer_metrics(answer: str, expected_answer: str, generated_sources: List[str], expected_sources: List[str]) -> dict:
    rouge_1 = rouge_n(expected_answer, answer, 1)
    rouge_2 = rouge_n(expected_answer, answer, 2)
    rouge_l_score = rouge_l(expected_answer, answer)
    fuzzy = fuzzy_similarity(expected_answer, answer)

    expected_set = {normalize_source(src) for src in expected_sources}
    generated_set = {normalize_source(src) for src in generated_sources}
    source_overlap = generated_set.intersection(expected_set)

    grounding = 0.0
    if generated_sources:
        grounding = round(len(source_overlap) / len(generated_set) * 100, 2)

    return {
        "rouge_1": rouge_1,
        "rouge_2": rouge_2,
        "rouge_l": rouge_l_score,
        "fuzzy": fuzzy,
        "grounding": grounding,
        "supported_sources": sorted(source_overlap),
    }


def evaluate_retrieval_quality(retrieved_sources: List[str], expected_sources: List[str]) -> dict:
    if not expected_sources:
        return {
            "mrr": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "ndcg": 0.0,
        }
    return compute_search_metrics(retrieved_sources, expected_sources)


def evaluate_answer_quality(answer: str, expected_answer: str, generated_sources: List[str], expected_sources: List[str]) -> dict:
    if not expected_answer:
        return {
            "rouge_1": 0.0,
            "rouge_2": 0.0,
            "rouge_l": 0.0,
            "fuzzy": 0.0,
            "grounding": 0.0,
        }
    return compute_answer_metrics(answer, expected_answer, generated_sources, expected_sources)
