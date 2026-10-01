"""ANN index wrappers: static old-space HNSW, incremental new-space HNSW, brute force."""
from __future__ import annotations

import numpy as np

from .utils import l2_normalize, l2_normalize_inplace, topk_indices


class BruteForceIndex:
    """Exact cosine search; the evaluation oracle and small-scale fallback."""

    def __init__(self, X: np.ndarray):
        self.X = l2_normalize_inplace(X)

    def search(self, q: np.ndarray, k: int, mask: np.ndarray | None = None):
        scores = self.X @ l2_normalize(q)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        ids = topk_indices(scores, k)
        ids = ids[np.isfinite(scores[ids])]
        return ids, scores[ids]


class HnswIndex:
    """hnswlib wrapper (cosine). Supports incremental adds and masked search
    via overfetch + post-filter with escalation."""

    def __init__(self, dim: int, max_elements: int, M: int = 16,
                 ef_construction: int = 200, ef_search: int = 128, seed: int = 42):
        import hnswlib

        self.dim = dim
        self.index = hnswlib.Index(space="cosine", dim=dim)
        self.index.init_index(max_elements=max_elements, M=M,
                              ef_construction=ef_construction, random_seed=seed)
        self.base_ef = ef_search
        self.index.set_ef(ef_search)
        self.count = 0

    def add(self, X: np.ndarray, ids: np.ndarray) -> None:
        X = l2_normalize(np.atleast_2d(X))
        ids = np.asarray(ids, dtype=np.int64)
        if X.shape[0] == 0:
            return
        self.index.add_items(X, ids)
        self.count += X.shape[0]

    def search(self, q: np.ndarray, k: int, mask: np.ndarray | None = None,
               max_escalation: int = 3):
        """Top-k by cosine similarity; if `mask` given, only ids with mask[id]=True.

        Post-filter strategy: fetch k, drop masked-out ids, escalate the fetch
        size (x4 each round) until k survivors or the index is exhausted.
        """
        if self.count == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        q = l2_normalize(np.asarray(q, dtype=np.float32))
        fetch = min(k, self.count)
        for _ in range(max_escalation + 1):
            # ef sized to this round's fetch (NOT ratcheted across queries:
            # a single escalated query must not inflate every later search)
            self.index.set_ef(max(self.base_ef, fetch))
            ids, dist = self.index.knn_query(q.reshape(1, -1), k=fetch)
            ids, dist = ids[0].astype(np.int64), dist[0]
            if mask is not None:
                keep = mask[ids]
                ids, dist = ids[keep], dist[keep]
            if ids.shape[0] >= k or fetch >= self.count:
                break
            fetch = min(self.count, fetch * 4)
        return ids[:k], (1.0 - dist[:k]).astype(np.float32)  # cosine similarity
