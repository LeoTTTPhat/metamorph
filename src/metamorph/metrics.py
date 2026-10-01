"""Retrieval metrics: recall@k vs a reference list, NDCG@k / MRR@k vs qrels."""
from __future__ import annotations

import numpy as np


def recall_at_k(served: np.ndarray, reference: np.ndarray, k: int) -> float:
    """|served[:k] ∩ reference[:k]| / |reference[:k]|."""
    ref = np.asarray(reference)[:k]
    if ref.size == 0:
        return 1.0
    return float(np.intersect1d(np.asarray(served)[:k], ref).size) / ref.size


def ndcg_at_k(served: np.ndarray, rels: dict[int, float], k: int) -> float:
    """NDCG@k with graded relevance from a {doc_id: gain} dict."""
    served = np.asarray(served)[:k]
    gains = np.array([rels.get(int(d), 0.0) for d in served], dtype=np.float64)
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg = float((gains * discounts).sum())
    ideal = np.sort(np.array(list(rels.values()), dtype=np.float64))[::-1][:k]
    idcg = float((ideal / np.log2(np.arange(2, ideal.size + 2))).sum())
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(served: np.ndarray, rels: dict[int, float], k: int) -> float:
    for rank, d in enumerate(np.asarray(served)[:k], start=1):
        if rels.get(int(d), 0.0) > 0:
            return 1.0 / rank
    return 0.0


def eval_query_set(serve_fn, q_ids: list[str], qrels: dict[str, dict[int, float]],
                   reference_topk: dict[str, np.ndarray], k: int) -> dict[str, float]:
    """Evaluate a serving function over a query set.

    serve_fn(query_id) -> ranked doc-id array.
    reference_topk: exact new-space top-k per query (agreement metric).
    """
    recs, ndcgs, mrrs = [], [], []
    for qid in q_ids:
        served = serve_fn(qid)
        if qid in reference_topk:
            recs.append(recall_at_k(served, reference_topk[qid], k))
        rel = qrels.get(qid)
        if rel:
            ndcgs.append(ndcg_at_k(served, rel, k))
            mrrs.append(mrr_at_k(served, rel, k))
    out = {}
    if recs:
        out[f"recall@{k}_vs_dest"] = float(np.mean(recs))
    if ndcgs:
        out[f"ndcg@{k}"] = float(np.mean(ndcgs))
        out[f"mrr@{k}"] = float(np.mean(mrrs))
    return out
