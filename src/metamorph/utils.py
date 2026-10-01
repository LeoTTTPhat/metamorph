"""Small shared utilities: seeding, normalization, timers, cost accounting."""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    """Return a fresh Generator; also seeds the legacy global RNG for any stray users."""
    np.random.seed(seed)
    return np.random.default_rng(seed)


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def l2_normalize_inplace(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize rows in place when already float32/C-contiguous, else copy.

    At 10^7 x 384 a defensive copy costs ~15 GB; the encoders already emit
    unit-norm rows, so the common case is a no-op scan.
    """
    x = np.asarray(x)
    if x.dtype != np.float32 or not x.flags["C_CONTIGUOUS"] or not x.flags["WRITEABLE"]:
        return l2_normalize(x)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.all(np.abs(n - 1.0) < 1e-4):
        return x
    np.divide(x, np.maximum(n, eps), out=x)
    return x


def cosine_scores(q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Cosine of a single (normalized) query vs rows of (normalized) X."""
    return X @ q


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest scores, sorted descending by score."""
    k = min(k, scores.shape[0])
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part], kind="stable")]


def rank_normalize(x: np.ndarray) -> np.ndarray:
    """Map values to [0, 1] by rank (ties broken by order); constant arrays -> 0."""
    n = x.shape[0]
    if n <= 1:
        return np.zeros_like(x, dtype=np.float32)
    order = np.argsort(x, kind="stable")
    out = np.empty(n, dtype=np.float32)
    out[order] = np.arange(n, dtype=np.float32) / (n - 1)
    if np.all(x == x[0]):
        return np.zeros(n, dtype=np.float32)
    return out


@dataclass
class CostMeter:
    """Budget accounting: 1 unit = 1 document re-embedded with the NEW model."""

    by_reason: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def charge(self, n: int, reason: str) -> None:
        if n < 0:
            raise ValueError("negative charge")
        self.by_reason[reason] += int(n)

    @property
    def total(self) -> int:
        return int(sum(self.by_reason.values()))

    def snapshot(self) -> dict[str, int]:
        d = dict(self.by_reason)
        d["total"] = self.total
        return d


class Timer:
    """Accumulating wall-clock timer keyed by stage name."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    class _Ctx:
        def __init__(self, timer: "Timer", key: str) -> None:
            self.timer, self.key = timer, key

        def __enter__(self):
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, *exc):
            self.timer.totals[self.key] += time.perf_counter() - self.t0
            self.timer.counts[self.key] += 1
            return False

    def track(self, key: str) -> "Timer._Ctx":
        return Timer._Ctx(self, key)

    def mean_ms(self, key: str) -> float:
        c = self.counts[key]
        return 1000.0 * self.totals[key] / c if c else 0.0
