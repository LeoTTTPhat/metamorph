"""Confidence bounds for the online recall certifier.

Two families, both for means of [0, 1]-valued samples:

* :func:`empirical_bernstein_lb` — the fixed-sample bound used in the first
  version of the system (Maurer & Pontil 2009, combined with Hoeffding).  It is
  valid for a *pre-specified* sample size only: applying it after every
  certification round, and stopping the first time it clears the target, is a
  multiple-testing violation, and its 1/m small-sample term makes it unusable
  below a few hundred samples.

* :func:`betting_lb` — an *anytime-valid* lower confidence bound from a
  capital process (Waudby-Smith & Ramdas 2024).  ``P(exists t: LB_t > mu) <=
  delta`` holds uniformly over time, so the certifier may look at the bound
  after every round and stop the moment it clears the target, and every sample
  ever drawn can be pooled without a window.  For low-variance samples near 1 —
  exactly the regime of a migration approaching its destination — it is far
  tighter than the empirical-Bernstein bound.

Both return a lower bound; :func:`upper_bound_from_lb` mirrors a bound to the
upper side by working with ``1 - X``.
"""
from __future__ import annotations

import numpy as np


def empirical_bernstein_lb(samples: np.ndarray, delta: float) -> float:
    """Fixed-sample 1-delta lower confidence bound for the mean of [0,1] samples.

    Combines the empirical-Bernstein bound (Maurer & Pontil 2009; tight for low
    variance but with a heavy O(1/m) term) with the one-sided Hoeffding bound,
    each at level delta/2; the max of the two is a valid 1-delta bound by the
    union bound.
    """
    samples = np.asarray(samples, dtype=np.float64)
    m = samples.shape[0]
    if m == 0:
        return 0.0
    mean = float(samples.mean())
    d2 = delta / 2.0
    hoeffding = mean - float(np.sqrt(np.log(1.0 / d2) / (2.0 * m)))
    if m < 2:
        return max(0.0, hoeffding)
    var = float(samples.var(ddof=1))
    ln = float(np.log(2.0 / d2))
    bernstein = mean - float(np.sqrt(2.0 * var * ln / m)) - 7.0 * ln / (3.0 * (m - 1))
    return max(0.0, hoeffding, bernstein)


def _log_capital(samples: np.ndarray, m: float, delta: float, c: float = 0.75) -> float:
    """log of the one-sided capital process for the null ``mu <= m``.

    Bets are predictable: the stake used on sample ``i`` depends only on
    ``samples[:i]``.  The predictable plug-in of Waudby-Smith & Ramdas is used,
    truncated to ``[0, c/m]`` so the capital stays positive.
    """
    x = np.asarray(samples, dtype=np.float64)
    t = x.shape[0]
    if t == 0:
        return 0.0
    # running mean / variance of the PREFIX (shifted by one, with the standard
    # 1/4 prior variance and 1/2 prior mean so the first bet is well defined)
    idx = np.arange(1, t + 1, dtype=np.float64)
    csum = np.cumsum(x)
    mu_hat = np.concatenate([[0.5], (0.5 + csum[:-1]) / (1.0 + idx[:-1])])
    csq = np.cumsum((x - mu_hat) ** 2)
    sigma2 = np.concatenate([[0.25], (0.25 + csq[:-1]) / (1.0 + idx[:-1])])
    lam = np.sqrt(2.0 * np.log(1.0 / delta) / np.maximum(sigma2 * idx * np.log(1.0 + idx), 1e-12))
    lam = np.minimum(lam, c / max(m, 1e-9))
    terms = 1.0 + lam * (x - m)
    if np.any(terms <= 0):
        return np.inf  # certain rejection of this m
    return float(np.sum(np.log(terms)))


def betting_lb(samples: np.ndarray, delta: float, grid: int = 256) -> float:
    """Anytime-valid 1-delta lower confidence bound for the mean of [0,1] samples.

    The bound is the largest ``m`` whose one-sided capital process has already
    exceeded ``1/delta``; validity is uniform over the sample stream, so this
    may be recomputed and acted on after every certification round.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.shape[0] == 0:
        return 0.0
    x = np.clip(x, 0.0, 1.0)
    thresh = np.log(1.0 / delta)
    lo, hi = 0.0, float(x.mean())
    if _log_capital(x, hi, delta) >= thresh:      # even the sample mean is rejected
        return hi
    if _log_capital(x, lo, delta) < thresh:       # no evidence that mu > 0
        return 0.0
    # capital is decreasing in m: bisect for the crossing point
    for _ in range(int(np.log2(grid)) + 8):
        mid = 0.5 * (lo + hi)
        if _log_capital(x, mid, delta) >= thresh:
            lo = mid
        else:
            hi = mid
    return float(lo)


def upper_bound_from_lb(samples: np.ndarray, delta: float, kind: str = "betting") -> float:
    """1-delta UPPER confidence bound, by bounding the mean of ``1 - X``."""
    x = np.asarray(samples, dtype=np.float64)
    if x.shape[0] == 0:
        return 1.0
    lb = lower_bound(1.0 - x, delta, kind)
    return float(min(1.0, 1.0 - lb))


def suffix_lower_bound(chunks: list[np.ndarray], delta: float, kind: str = "betting",
                       max_rounds: int = 64) -> tuple[float, int]:
    """Tightest valid lower bound over the whole stream and over its suffixes.

    Pooling every sample ever drawn is licensed by an anytime-valid bound, but
    it bounds the mean of the *whole stream*, which---when quality is rising---
    is dragged down by the earliest and weakest rounds.  Dropping old rounds is
    tighter but is a data-dependent choice, so it has to be paid for.  The error
    budget is split: ``delta/2`` on the full stream, and ``delta/2`` shared over
    at most ``max_rounds - 1`` suffix start points.  The result is a valid
    ``1 - delta`` lower bound by a union bound; each component is itself
    anytime-valid in the number of rounds, so the whole construction may be
    recomputed and acted on after every round.  Because the full stream keeps
    half the budget, this is never much worse than pooling and is far better
    once recent rounds carry enough samples to stand alone.

    Returns (bound, number of rounds in the winning suffix).
    """
    if not chunks:
        return 0.0, 0
    n = len(chunks)
    best = lower_bound(np.concatenate(chunks), delta / 2.0, kind)
    best_len = n
    if n > 1:
        starts = range(max(1, n - (max_rounds - 1)), n)
        d = delta / (2.0 * (max_rounds - 1))
        for j in starts:
            pooled = np.concatenate(chunks[j:])
            if pooled.size == 0:
                continue
            lb = lower_bound(pooled, d, kind)
            if lb > best:
                best, best_len = lb, n - j
    return best, best_len


def lower_bound(samples: np.ndarray, delta: float, kind: str = "betting") -> float:
    if kind == "eb":
        return empirical_bernstein_lb(samples, delta)
    if kind == "betting":
        return betting_lb(samples, delta)
    raise ValueError(f"unknown bound kind {kind!r}")
