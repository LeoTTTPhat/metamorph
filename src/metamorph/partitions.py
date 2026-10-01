"""Partitioner: k-means over old-space embeddings; the unit of cutover."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


class Partitioner:
    def __init__(self, n_partitions: int, seed: int = 42):
        self.n_partitions = n_partitions
        self.seed = seed
        self.assignments: np.ndarray | None = None
        self.centroids: np.ndarray | None = None

    # above this corpus size full-batch Lloyd is replaced by mini-batch k-means:
    # partitions are a bookkeeping unit (cutover granularity), not a retrieval
    # structure, so the cheaper clustering is more than accurate enough.
    MINIBATCH_THRESHOLD = 200_000

    def fit(self, X_old: np.ndarray) -> "Partitioner":
        k = min(self.n_partitions, X_old.shape[0])
        if X_old.shape[0] > self.MINIBATCH_THRESHOLD:
            from sklearn.cluster import MiniBatchKMeans
            km = MiniBatchKMeans(n_clusters=k, random_state=self.seed, n_init=3,
                                 batch_size=8192, max_iter=100,
                                 max_no_improvement=20)
        else:
            km = KMeans(n_clusters=k, random_state=self.seed, n_init=3)
        self.assignments = km.fit_predict(X_old).astype(np.int32)
        self.centroids = km.cluster_centers_.astype(np.float32)
        self.n_partitions = k
        return self

    def members(self, p: int) -> np.ndarray:
        return np.where(self.assignments == p)[0]

    def sizes(self) -> np.ndarray:
        return np.bincount(self.assignments, minlength=self.n_partitions)
