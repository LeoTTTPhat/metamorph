"""Query workload: split queries into workload/eval streams; Zipf-skewed sampling."""
from __future__ import annotations

import numpy as np


def split_queries(query_ids: list[str], seed: int, workload_frac: float = 0.5
                  ) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    ids = list(query_ids)
    rng.shuffle(ids)
    n_w = max(1, int(len(ids) * workload_frac))
    return ids[:n_w], ids[n_w:] or ids[:n_w]


class ZipfStream:
    """Infinite stream over a fixed query set with Zipf(alpha) popularity skew."""

    def __init__(self, query_ids: list[str], alpha: float, seed: int):
        self.ids = list(query_ids)
        rng = np.random.default_rng(seed)
        ranks = np.arange(1, len(self.ids) + 1, dtype=np.float64)
        p = ranks ** (-alpha)
        self.p = p / p.sum()
        # randomize which query gets which popularity rank
        self.perm = rng.permutation(len(self.ids))
        self.rng = rng

    def next_batch(self, n: int) -> list[str]:
        picks = self.rng.choice(len(self.ids), size=n, p=self.p)
        return [self.ids[self.perm[i]] for i in picks]
