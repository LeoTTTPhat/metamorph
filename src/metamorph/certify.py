"""Online quality certification via disagreement sampling.

Estimates the quality of the CURRENT serving configuration against the
fully-migrated destination system, by sampling queries, revealing (paying for)
true new embeddings of a small candidate pool, and comparing the served top-k
with the exact pool-restricted top-k.  Revealed docs are recycled as migrated,
so certification budget doubles as migration progress.

Three things distinguish this from a naive sampler.

1. **Anytime-valid bounds.**  The decision bound is a betting confidence
   sequence (:mod:`metamorph.bounds`), valid uniformly over rounds, so the
   certifier may look after every round and stop the first time it clears the
   target.  Every sample ever drawn is pooled; no window is needed.  The
   fixed-sample empirical-Bernstein bound is retained for ablation.

2. **Pool-truncation debiasing.**  The pool is finite, so the pool-restricted
   top-k can miss a true top-k document, which biases the estimate *upward* —
   the unsafe direction.  A random subsample of certification queries is
   re-scored against a doubled pool; the mean drop is an estimate of the
   truncation bias and its one-sided upper confidence bound is subtracted from
   the decision bound, making the guarantee hold for global recall rather than
   pool-restricted recall.  The same probe yields a selection rule for the pool
   depth C: grow C until the measured bias upper bound falls under tolerance.

3. **A quality target, not only a set-agreement target.**  Strict top-k
   agreement is a far stricter goal than end-task quality: swapping two
   near-tied documents costs a full agreement point and almost no utility.
   The certifier therefore also estimates *score mass* --- the fraction of the
   destination top-k's exact new-space score mass that the served list
   captures --- which needs no relevance judgements and is computed from
   exactly the same reveals.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bounds import (empirical_bernstein_lb, lower_bound, suffix_lower_bound,
                     upper_bound_from_lb)
from .config import CertifierConfig
from .serve import QueryRouter
from .state import MigrationState
from .store import EmbeddingStore
from .utils import topk_indices

__all__ = ["CertReport", "Certifier", "empirical_bernstein_lb"]


@dataclass
class CertReport:
    recall_hat: float               # pool-restricted top-k agreement estimate
    lower_bound: float              # this round's samples only
    pooled_lb: float                # the decision bound (bias-corrected)
    n_queries: int
    n_pooled: int
    certified: bool
    cost_units: int
    miss_mass_by_partition: np.ndarray = field(default_factory=lambda: np.empty(0))
    # --- estimator extensions ------------------------------------------
    score_mass_hat: float = float("nan")   # fraction of destination score mass served
    score_mass_lb: float = float("nan")    # anytime LB on score mass
    raw_pooled_lb: float = float("nan")    # decision bound BEFORE bias correction
    bias_hat: float = float("nan")         # estimated pool-truncation bias
    bias_ub: float = float("nan")          # one-sided UCB on that bias
    n_probe: int = 0                       # queries used for the doubled-pool probe
    eb_pooled_lb: float = float("nan")     # empirical-Bernstein bound, for ablation
    metric: str = "recall"
    suffix_rounds: int = 0                 # rounds in the winning suffix window


class Certifier:
    def __init__(self, store: EmbeddingStore, state: MigrationState,
                 router: QueryRouter, cfg: CertifierConfig, rng: np.random.Generator):
        self.store, self.state, self.router, self.cfg = store, state, router, cfg
        self.rng = rng
        self.history: list[CertReport] = []
        # per-round samples.  With an anytime-valid bound the whole stream may
        # be pooled; `history_window` still truncates it when the ablation asks
        # for the windowed (fixed-sample) behaviour.
        self._samples: list[np.ndarray] = []
        self._mass_samples: list[np.ndarray] = []
        self._bias_samples: list[np.ndarray] = []        # agreement truncation bias
        self._bias_mass_samples: list[np.ndarray] = []   # score-mass truncation bias

    # ------------------------------------------------------------------ pools
    def _pool_for(self, res_ids: np.ndarray, q_old: np.ndarray, q_new: np.ndarray,
                  depth: int) -> np.ndarray:
        """Served list widened with old-space AND new-space plausible candidates.

        Migrated documents are free to score (their vectors are already paid
        for), so including the new index's top-C costs nothing and closes the
        commonest source of pool-truncation bias: a document whose true
        new-space score is high but whose old-space score is not.
        """
        parts = [res_ids]
        if depth > 0:
            extra_ids, _ = self.router.old_index.search(
                q_old if q_old is not None else self.router.bank.g(q_new), depth)
            parts.append(extra_ids)
            if self.router.new_index.count:
                new_ids, _ = self.router.new_index.search(q_new, depth)
                parts.append(new_ids)
        return np.unique(np.concatenate(parts))

    @staticmethod
    def _quality(served: np.ndarray, pool: np.ndarray, exact: np.ndarray,
                 k: int) -> tuple[float, float, np.ndarray]:
        """(agreement, score-mass ratio, missed ids) of `served` inside `pool`."""
        top_idx = topk_indices(exact, min(k, pool.size))
        true_top = pool[top_idx]
        inter = np.intersect1d(served, true_top, assume_unique=False)
        agree = inter.size / max(true_top.size, 1)
        # score mass: served list's exact new-space scores vs the pool's best k.
        # Scores are cosines in [-1, 1]; shift to [0, 2] so the ratio is a
        # well-defined [0, 1] quantity that a bounded-mean bound applies to.
        pos = {int(d): j for j, d in enumerate(pool)}
        served_scores = np.array([exact[pos[int(d)]] for d in served if int(d) in pos],
                                 dtype=np.float64)
        best = exact[top_idx].astype(np.float64)
        num = float(np.sum(np.sort(served_scores)[::-1][:k] + 1.0)) if served_scores.size else 0.0
        den = float(np.sum(best + 1.0))
        mass = min(1.0, num / den) if den > 0 else 1.0
        missed = np.setdiff1d(true_top, served, assume_unique=False)
        return float(agree), float(mass), missed

    # ------------------------------------------------------------------ round
    def certify(self, q_old_pool: np.ndarray, q_new_pool: np.ndarray) -> CertReport:
        """Run one certification round over a sample of the recent query stream."""
        cfg = self.cfg
        k = self.router.cfg.k
        Q = q_new_pool.shape[0]
        m = cfg.sample_queries
        pick = self.rng.choice(Q, size=m, replace=m > Q)
        cost_before = self.store.meter.by_reason.get("certify", 0)
        # cost-capped round: queries are processed in random order until the
        # round's reveal budget is exhausted, so stopping early keeps the
        # processed prefix an unbiased sample.
        cap_units = max(1, int(round(cfg.max_round_cost_frac * self.store.n)))
        agrees = np.zeros(m, dtype=np.float32)
        masses = np.zeros(m, dtype=np.float32)
        biases: list[float] = []
        biases_mass: list[float] = []
        miss_mass = np.zeros(self.state.n_partitions, dtype=np.float64)
        m_done = 0
        probe_every = (int(round(1.0 / cfg.probe_frac))
                       if cfg.probe_frac > 0 else 0)

        for j, qi in enumerate(pick):
            if (j > 0 and self.store.meter.by_reason.get("certify", 0)
                    - cost_before >= cap_units):
                break
            m_done = j + 1
            q_new = q_new_pool[qi]
            q_old = q_old_pool[qi] if q_old_pool is not None else None
            res = self.router.serve(q_old, q_new)
            served = res.ids
            pool = self._pool_for(res.ids, q_old, q_new, cfg.pool_extra)
            if pool.size == 0:
                agrees[j] = masses[j] = 1.0
                continue
            true_new = self.store.reveal(pool, reason="certify")
            exact = true_new @ q_new
            agrees[j], masses[j], missed = self._quality(served, pool, exact, k)
            if missed.size:
                for p in self.state.part_of[missed]:
                    miss_mass[p] += 1.0
            # ---- pool-truncation probe: same query, doubled pool ------------
            if probe_every and (j % probe_every == 0) and cfg.pool_extra > 0:
                deep = self._pool_for(res.ids, q_old, q_new,
                                      cfg.pool_extra * cfg.probe_multiplier)
                if deep.size > pool.size:
                    deep_new = self.store.reveal(deep, reason="certify")
                    deep_exact = deep_new @ q_new
                    a_deep, m_deep, _ = self._quality(served, deep, deep_exact, k)
                    biases.append(max(0.0, float(agrees[j] - a_deep)))
                    biases_mass.append(max(0.0, float(masses[j] - m_deep)))

        if cfg.recycle:
            self._recycled = np.where(self.store.revealed & ~self.state.migrated)[0]
        else:
            self._recycled = np.empty(0, dtype=np.int64)

        agrees, masses = agrees[:m_done], masses[:m_done]
        miss_mass /= max(m_done, 1)
        self._samples.append(agrees)
        self._mass_samples.append(masses)
        if biases:
            self._bias_samples.append(np.asarray(biases, dtype=np.float32))
            self._bias_mass_samples.append(np.asarray(biases_mass, dtype=np.float32))

        def _window(chunks: list[np.ndarray]) -> list[np.ndarray]:
            return chunks[-cfg.history_window:] if cfg.history_window > 0 else chunks

        def _flat(chunks: list[np.ndarray]) -> np.ndarray:
            return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)

        # the decision bound spends delta/2 on the quality samples and delta/2
        # on the truncation-bias samples, so their sum is a 1-delta bound on
        # deep-pool quality by the union bound
        d_half = cfg.delta / 2.0 if cfg.probe_frac > 0 else cfg.delta
        if cfg.suffix_bound and cfg.history_window <= 0:
            raw_lb, suffix_rounds = suffix_lower_bound(self._samples, d_half, cfg.bound)
            mass_lb, _ = suffix_lower_bound(self._mass_samples, d_half, cfg.bound)
        else:
            raw_lb = lower_bound(_flat(_window(self._samples)), d_half, cfg.bound)
            mass_lb = lower_bound(_flat(_window(self._mass_samples)), d_half, cfg.bound)
            suffix_rounds = len(_window(self._samples))
        pooled = _flat(_window(self._samples))
        eb_lb = empirical_bernstein_lb(pooled, cfg.delta)

        # ---- pool-truncation bias correction (per target metric) ------------
        src = (self._bias_mass_samples if cfg.metric == "score_mass"
               else self._bias_samples)
        if src:
            bias_all = np.concatenate(src)
            bias_hat = float(bias_all.mean())
            bias_ub = float(upper_bound_from_lb(bias_all, d_half, cfg.bound))
        else:
            bias_all = np.empty(0, dtype=np.float32)
            bias_hat = bias_ub = 0.0 if cfg.probe_frac <= 0 else float("nan")

        corr = bias_ub if np.isfinite(bias_ub) else 0.0
        decision_lb = max(0.0, (mass_lb if cfg.metric == "score_mass" else raw_lb) - corr)
        # a decision needs enough probe samples for the truncation correction to
        # mean something; below that the certifier abstains rather than certify
        # against an uninformative bias bound
        can_decide = (cfg.probe_frac <= 0) or (bias_all.size >= cfg.min_probe)

        report = CertReport(
            recall_hat=float(agrees.mean()) if m_done else 0.0,
            lower_bound=lower_bound(agrees, cfg.delta, cfg.bound),
            pooled_lb=decision_lb,
            n_queries=m_done, n_pooled=int(pooled.size),
            certified=bool(can_decide and decision_lb >= cfg.target_recall),
            cost_units=self.store.meter.by_reason.get("certify", 0) - cost_before,
            miss_mass_by_partition=miss_mass,
            score_mass_hat=float(masses.mean()) if m_done else 0.0,
            score_mass_lb=mass_lb, raw_pooled_lb=raw_lb,
            bias_hat=bias_hat, bias_ub=bias_ub, n_probe=int(bias_all.size),
            eb_pooled_lb=eb_lb, metric=cfg.metric, suffix_rounds=suffix_rounds)
        self.history.append(report)
        return report

    def pop_recycled(self) -> np.ndarray:
        out = getattr(self, "_recycled", np.empty(0, dtype=np.int64))
        self._recycled = np.empty(0, dtype=np.int64)
        return out
