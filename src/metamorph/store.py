"""EmbeddingStore: old embeddings plus a budget-gated oracle for new embeddings.

The NEW-model embeddings are precomputed once per dataset, but the migration
engine may only read them through `reveal(ids, reason)`, which charges the
CostMeter (1 unit per newly revealed doc). This gives exact cost accounting
while letting a full migration timeline run in minutes.
"""
from __future__ import annotations

import numpy as np

from .utils import CostMeter, l2_normalize_inplace


class EmbeddingStore:
    def __init__(self, old: np.ndarray, new_oracle: np.ndarray, meter: CostMeter | None = None):
        if old.shape[0] != new_oracle.shape[0]:
            raise ValueError("old/new row mismatch")
        self.old = l2_normalize_inplace(old)
        self._new = l2_normalize_inplace(new_oracle)
        self.n = old.shape[0]
        self.d_old = old.shape[1]
        self.d_new = new_oracle.shape[1]
        self.revealed = np.zeros(self.n, dtype=bool)
        self.meter = meter if meter is not None else CostMeter()

    def reveal(self, ids: np.ndarray, reason: str) -> np.ndarray:
        """Charge budget for not-yet-revealed ids and return their new embeddings."""
        ids = np.asarray(ids, dtype=np.int64)
        fresh = ids[~self.revealed[ids]]
        # ids may contain duplicates; charge unique fresh ids only
        fresh = np.unique(fresh)
        if fresh.size:
            self.meter.charge(int(fresh.size), reason)
            self.revealed[fresh] = True
        return self._new[ids]

    def peek_revealed(self, ids: np.ndarray) -> np.ndarray:
        """Read new embeddings for ids that MUST already be revealed (free)."""
        ids = np.asarray(ids, dtype=np.int64)
        if not np.all(self.revealed[ids]):
            raise RuntimeError("peek_revealed on unrevealed ids — budget violation")
        return self._new[ids]

    def oracle_full(self) -> np.ndarray:
        """Ground-truth access for EVALUATION ONLY (never inside the serving path)."""
        return self._new

    @property
    def revealed_fraction(self) -> float:
        return float(self.revealed.mean())
