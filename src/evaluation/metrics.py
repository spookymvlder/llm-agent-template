"""Retrieval metrics from binary relevance (is this retrieved chunk one of the expected sources?)."""
from __future__ import annotations

import math
from typing import Iterable, Mapping


def precision_at_k(relevant: list[bool], k: int) -> float:
    """Fraction of the top-k results that are relevant. `relevant` is in rank order (index 0 = rank 1)."""
    top = relevant[:k]
    return sum(top) / len(top) if top else 0.0


def recall_at_k(matched_expected: int, total_expected: int) -> float:
    """Fraction of the expected sources that appeared in the top k."""
    return matched_expected / total_expected if total_expected else 0.0


def hit_rate_at_k(relevant: list[bool], k: int) -> float:
    """1.0 if any of the top-k results is relevant, else 0.0."""
    return 1.0 if any(relevant[:k]) else 0.0


def mrr(relevant: list[bool]) -> float:
    """Reciprocal rank of the first relevant result (0.0 if none)."""
    for i, is_relevant in enumerate(relevant):
        if is_relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(relevant: list[bool], k: int, total_expected: int) -> float:
    """Normalised discounted cumulative gain: rewards relevant results ranked higher.

    The ideal ranking puts min(total_expected, k) relevant results first.
    """
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(relevant[:k]) if r)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(total_expected, k)))
    return dcg / ideal if ideal else 0.0


def retrieval_metrics(relevant: list[bool], matched_expected: int, total_expected: int, k: int) -> dict[str, float]:
    """All retrieval metrics for one query."""
    return {
        f"hit_rate@{k}": hit_rate_at_k(relevant, k),
        f"precision@{k}": precision_at_k(relevant, k),
        f"recall@{k}": recall_at_k(matched_expected, total_expected),
        "mrr": mrr(relevant),
        f"ndcg@{k}": ndcg_at_k(relevant, k, total_expected),
    }


def mean_metrics(rows: Iterable[Mapping[str, float | None]]) -> dict[str, float]:
    """Average each key across rows, ignoring missing/None values (keys absent everywhere are omitted)."""
    totals: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, bool):
                value = float(value)
            if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
                totals.setdefault(key, []).append(float(value))
    return {key: round(sum(vals) / len(vals), 4) for key, vals in totals.items()}
