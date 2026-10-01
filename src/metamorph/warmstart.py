"""Warm-start construction of the new-space index from old-space structure.

Two mechanisms, both implemented from scratch so work can be counted honestly:

1. warm k-means (IVF/partitions): initialize new-space Lloyd iterations with
   the new-space means of the OLD cluster assignments, vs k-means++ init.
   Metric: iterations to converge, final inertia.

2. warm NN-descent (graph indexes): build the new-space kNN graph by
   NN-descent initialized from the OLD-space kNN graph, vs random init.
   Metric: distance computations vs graph recall; the finished graph is
   servable via beam search (query recall@k measured in exp4).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import l2_normalize


# --------------------------------------------------------------- warm k-means
@dataclass
class KMeansResult:
    centroids: np.ndarray
    assignments: np.ndarray
    inertia: float
    n_iter: int


def lloyd_kmeans(X: np.ndarray, init_centroids: np.ndarray, max_iter: int = 100,
                 tol: float = 1e-4) -> KMeansResult:
    """Plain Lloyd iterations from a given init (cosine-normalized data, L2 objective)."""
    X = np.asarray(X, dtype=np.float32)
    C = np.array(init_centroids, dtype=np.float32, copy=True)
    k = C.shape[0]
    assign = np.zeros(X.shape[0], dtype=np.int32)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        d2 = ((X ** 2).sum(1, keepdims=True) - 2 * X @ C.T + (C ** 2).sum(1))
        assign = np.argmin(d2, axis=1).astype(np.int32)
        newC = np.zeros_like(C)
        counts = np.bincount(assign, minlength=k).astype(np.float32)
        np.add.at(newC, assign, X)
        nonempty = counts > 0
        newC[nonempty] /= counts[nonempty, None]
        newC[~nonempty] = C[~nonempty]
        shift = float(np.linalg.norm(newC - C, axis=1).max())
        C = newC
        if shift < tol:
            break
    d2 = ((X ** 2).sum(1, keepdims=True) - 2 * X @ C.T + (C ** 2).sum(1))
    inertia = float(d2[np.arange(X.shape[0]), assign].sum())
    return KMeansResult(C, assign, inertia, n_iter)


def warm_kmeans_init(X_new: np.ndarray, old_assignments: np.ndarray, k: int) -> np.ndarray:
    """Centroids = new-space means of the old clusters."""
    C = np.zeros((k, X_new.shape[1]), dtype=np.float32)
    counts = np.bincount(old_assignments, minlength=k).astype(np.float32)
    np.add.at(C, old_assignments, X_new.astype(np.float32))
    nz = counts > 0
    C[nz] /= counts[nz, None]
    if (~nz).any():  # empty old clusters: random points
        rand = np.random.default_rng(0).choice(X_new.shape[0], size=int((~nz).sum()))
        C[~nz] = X_new[rand]
    return C


# ------------------------------------------------------------- warm NN-descent
class KnnGraph:
    """Fixed-degree kNN graph with similarity-ordered neighbor lists."""

    def __init__(self, neighbors: np.ndarray, sims: np.ndarray):
        self.neighbors = neighbors            # (N, K) int64
        self.sims = sims                      # (N, K) float32
        self.n, self.k = neighbors.shape

    @staticmethod
    def random_init(X: np.ndarray, k: int, rng: np.random.Generator,
                    counter: list[int] | None = None) -> "KnnGraph":
        n = X.shape[0]
        nbrs = np.empty((n, k), dtype=np.int64)
        sims = np.empty((n, k), dtype=np.float32)
        for i in range(n):
            cand = rng.choice(n - 1, size=k, replace=False)
            cand[cand >= i] += 1
            s = X[cand] @ X[i]
            if counter is not None:
                counter[0] += k
            order = np.argsort(-s)
            nbrs[i], sims[i] = cand[order], s[order]
        return KnnGraph(nbrs, sims)

    @staticmethod
    def from_old_graph(X_new: np.ndarray, old_neighbors: np.ndarray,
                       counter: list[int] | None = None) -> "KnnGraph":
        """Rescore old-space neighbor lists with new-space similarities."""
        n, k = old_neighbors.shape
        sims = np.empty((n, k), dtype=np.float32)
        nbrs = np.empty((n, k), dtype=np.int64)
        for i in range(n):
            cand = old_neighbors[i]
            s = X_new[cand] @ X_new[i]
            if counter is not None:
                counter[0] += k
            order = np.argsort(-s)
            nbrs[i], sims[i] = cand[order], s[order]
        return KnnGraph(nbrs, sims)

    def recall_vs_truth(self, true_neighbors: np.ndarray) -> float:
        """Fraction of true kNN edges present, on the rows of true_neighbors."""
        hits, total = 0, 0
        for i in range(true_neighbors.shape[0]):
            hits += np.intersect1d(self.neighbors[i], true_neighbors[i]).size
            total += true_neighbors.shape[1]
        return hits / max(total, 1)


def nn_descent(X: np.ndarray, graph: KnnGraph, n_iters: int = 10,
               sample: int = 20, rng: np.random.Generator | None = None,
               counter: list[int] | None = None,
               callback=None, random_inject: int = 2) -> KnnGraph:
    """NN-descent refinement (Dong et al. 2011, simplified local join).

    counter: single-element list accumulating #distance computations.
    callback(iter, graph): invoked after each iteration (for recall curves).
    random_inject: random candidates added per node per iteration — supplies
    the exploration diversity a correlated (warm) init lacks, preventing
    premature plateaus (costs are counted like any other candidate).
    """
    X = l2_normalize(X)
    rng = rng or np.random.default_rng(0)
    n, k = graph.n, graph.k
    for it in range(1, n_iters + 1):
        # reverse neighbor lists (sampled)
        rev: list[list[int]] = [[] for _ in range(n)]
        for i in range(n):
            for j in graph.neighbors[i]:
                if len(rev[j]) < sample:
                    rev[j].append(i)
        updated = 0
        for i in range(n):
            fwd = graph.neighbors[i]
            pool = set(fwd.tolist())
            for j in fwd[:sample]:
                pool.update(graph.neighbors[j][:sample].tolist())
                pool.update(rev[j][:5])
            pool.update(rev[i])
            if random_inject > 0:
                pool.update(rng.integers(0, n, size=random_inject).tolist())
            pool.discard(i)
            pool.difference_update(fwd.tolist())
            if not pool:
                continue
            cand = np.fromiter(pool, dtype=np.int64)
            s = X[cand] @ X[i]
            if counter is not None:
                counter[0] += cand.size
            all_ids = np.concatenate([graph.neighbors[i], cand])
            all_s = np.concatenate([graph.sims[i], s])
            order = np.argsort(-all_s)[:k]
            new_nbrs = all_ids[order]
            updated += int((new_nbrs != graph.neighbors[i]).any())
            graph.neighbors[i], graph.sims[i] = new_nbrs, all_s[order]
        if callback is not None:
            callback(it, graph)
        if updated == 0:
            break
    return graph


def beam_search(X: np.ndarray, graph: KnnGraph, q: np.ndarray, k: int,
                ef: int = 64, n_entries: int = 4,
                rng: np.random.Generator | None = None) -> np.ndarray:
    """Greedy best-first search over the kNN graph (returns top-k ids)."""
    X = l2_normalize(X)
    q = l2_normalize(np.asarray(q, dtype=np.float32))
    rng = rng or np.random.default_rng(0)
    entries = rng.choice(graph.n, size=min(n_entries, graph.n), replace=False)
    visited = set(entries.tolist())
    scores = {int(e): float(X[e] @ q) for e in entries}
    frontier = sorted(scores, key=scores.get, reverse=True)
    best: list[int] = list(frontier)
    while frontier:
        cur = frontier.pop(0)
        # expand
        for nb in graph.neighbors[cur]:
            nb = int(nb)
            if nb in visited:
                continue
            visited.add(nb)
            scores[nb] = float(X[nb] @ q)
            frontier.append(nb)
        best = sorted(visited, key=lambda i: scores[i], reverse=True)[:ef]
        frontier = [f for f in frontier if scores[f] >= scores[best[-1]]]
        frontier.sort(key=lambda i: scores[i], reverse=True)
        if len(frontier) > ef:
            frontier = frontier[:ef]
        if not frontier:
            break
    return np.array(best[:k], dtype=np.int64)


def exact_knn(X: np.ndarray, k: int, rows: np.ndarray | None = None,
              block: int = 512) -> np.ndarray:
    """Exact kNN (excluding self) for the given rows (default: all), blocked matmul."""
    X = l2_normalize(X)
    rows = np.arange(X.shape[0]) if rows is None else np.asarray(rows, dtype=np.int64)
    out = np.empty((rows.shape[0], k), dtype=np.int64)
    for s0 in range(0, rows.shape[0], block):
        rb = rows[s0:s0 + block]
        S = X[rb] @ X.T                       # (b, N)
        S[np.arange(rb.shape[0]), rb] = -np.inf
        part = np.argpartition(-S, k, axis=1)[:, :k]
        ord_ = np.take_along_axis(S, part, axis=1)
        idx = np.argsort(-ord_, axis=1)
        out[s0:s0 + rb.shape[0]] = np.take_along_axis(part, idx, axis=1)
    return out
