"""Synthetic migration data for tests and dry runs (no downloads, no models).

Construction: old embeddings are a mixture of Gaussians on the sphere; the new
space is a random rotation of the old space plus per-cluster distortion and
noise (so the old->new map is learnable but imperfect — the regime real model
upgrades live in). Queries are noisy copies of docs; each query's source doc
is its single relevant document.
"""
from __future__ import annotations

import numpy as np

from .migrate import MigrationData
from .utils import l2_normalize


def random_semi_orthogonal(d_in: int, d_out: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(d_in, d_out))
    U, _, Vt = np.linalg.svd(A, full_matrices=False)
    return (U @ Vt).astype(np.float32)


def make_synthetic(n_docs: int = 600, n_queries: int = 80, d_old: int = 32,
                   d_new: int = 48, n_clusters: int = 6, noise: float = 0.15,
                   cluster_distortion: float = 0.25, seed: int = 0) -> MigrationData:
    rng = np.random.default_rng(seed)
    centers = l2_normalize(rng.normal(size=(n_clusters, d_old)))
    assign = rng.integers(0, n_clusters, size=n_docs)
    old = l2_normalize(centers[assign] + 0.35 * rng.normal(size=(n_docs, d_old)))

    W = random_semi_orthogonal(d_old, d_new, rng)
    # per-cluster distortion: some regions of the space shift more than others
    cluster_shift = rng.normal(size=(n_clusters, d_new)) * cluster_distortion
    new = old @ W + cluster_shift[assign] + noise * rng.normal(size=(n_docs, d_new))
    new = l2_normalize(new)

    q_src = rng.choice(n_docs, size=n_queries, replace=False)
    q_old = l2_normalize(old[q_src] + 0.2 * rng.normal(size=(n_queries, d_old)))
    q_new = l2_normalize(new[q_src] + 0.2 * rng.normal(size=(n_queries, d_new)))

    q_ids = [f"q{i}" for i in range(n_queries)]
    qrels = {f"q{i}": {int(q_src[i]): 1.0} for i in range(n_queries)}
    return MigrationData(
        doc_ids=[f"d{i}" for i in range(n_docs)], old=old, new=new,
        q_ids=q_ids, q_old=q_old, q_new=q_new, qrels=qrels,
        meta={"synthetic": True, "seed": seed, "n_docs": n_docs})
