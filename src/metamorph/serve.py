"""Dual-space query router with verification reranking.

Candidate generation is per-space (old HNSW over unmigrated docs, new HNSW
over migrated docs); the FINAL ranking is always computed in the new space:
exact embeddings for migrated docs, adapter-mapped embeddings for unmigrated
docs, optional query-time verification (true re-embeds of the top unmigrated
candidates, budget-accounted and recycled into migration progress).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .adapters import AdapterBank
from .config import ServeConfig
from .index import HnswIndex
from .state import MigrationState
from .store import EmbeddingStore
from .utils import Timer, l2_normalize, topk_indices


@dataclass
class ServedResult:
    ids: np.ndarray                 # final top-k doc ids (desc score)
    scores: np.ndarray              # final new-space scores
    candidates: np.ndarray          # union candidate set (for scheduler signals)
    margin: float                   # score gap around rank k (hardness signal)
    n_from_old: int
    n_from_new: int
    verified_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    # --- operational form of the margin bound (Prop. 2) -------------------
    # eps_hat: a query-time ESTIMATE of the fusion error, the high quantile of
    #   the calibration residuals on this query's migrated candidates (whose
    #   exact new-space scores are known for free).  No oracle access.
    # margin_k: the served score gap at rank k.
    # certified_topk: margin_k > 2*eps_hat, i.e. the bound's precondition holds
    #   with the measurable surrogate in place of the unobservable eps_q.
    eps_hat: float = float("nan")
    margin_k: float = float("nan")
    certified_topk: bool = False
    # cert_recall_lb: graded per-query certificate. The fraction of served
    #   top-k documents that provably belong to the pool's true new-space
    #   top-k, given the per-candidate fusion-error bound. 1.0 recovers the
    #   all-or-nothing margin condition; unlike it, this degrades gracefully.
    cert_recall_lb: float = float("nan")
    cand_scores: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))


class QueryRouter:
    def __init__(self, store: EmbeddingStore, state: MigrationState,
                 old_index: HnswIndex, new_index: HnswIndex,
                 bank: AdapterBank, cfg: ServeConfig, timer: Timer | None = None):
        self.store, self.state = store, state
        self.old_index, self.new_index = old_index, new_index
        self.bank, self.cfg = bank, cfg
        self.timer = timer if timer is not None else Timer()
        # leak-free adapted embeddings for the adapter's TRAINING docs (their
        # regular adapted scores are leak-inflated): per-query fusion refits
        # use these cross-fitted vectors instead. Set via set_cv_adapted.
        self._cv_pos: np.ndarray | None = None    # doc id -> row in _cv_vecs (-1: none)
        self._cv_vecs: np.ndarray | None = None

    def set_cv_adapted(self, ids: np.ndarray, vecs: np.ndarray) -> None:
        pos = np.full(self.store.n, -1, dtype=np.int64)
        pos[np.asarray(ids, dtype=np.int64)] = np.arange(len(ids))
        self._cv_pos, self._cv_vecs = pos, np.asarray(vecs, dtype=np.float32)

    def _fusion_theta(self, ids_new: np.ndarray, exact_scores: np.ndarray,
                      q_old: np.ndarray, q_new: np.ndarray,
                      min_fit: int = 10, lam: float = 1.0) -> np.ndarray:
        """Per-query score-fusion weights, refined on dual-scored migrated docs.

        Migrated candidates carry BOTH evidence scores (their old embeddings
        are retained) and an exact new-space target, so the global fusion
        theta from the seed sample can be re-fitted per query. Ridge shrinkage
        toward the global theta keeps the fit robust to range restriction
        (top candidates have compressed score ranges) and small samples.
        """
        theta0 = self.bank.calib_theta
        n = ids_new.shape[0]
        self._last_eps_hat = float("nan")
        if n < min_fit:
            return theta0
        u1 = self.store.old[ids_new] @ q_old
        u2 = self.bank.f(self.store.old[ids_new]) @ q_new
        if self._cv_pos is not None:
            pos = self._cv_pos[ids_new]
            has = pos >= 0
            if has.any():  # adapter-training docs: swap in cross-fitted scores
                u2[has] = self._cv_vecs[pos[has]] @ q_new
        A = np.stack([u1, u2, np.ones_like(u1)], axis=1).astype(np.float64)
        v = exact_scores.astype(np.float64)
        AtA = A.T @ A + lam * np.eye(3)
        theta = np.linalg.solve(AtA, A.T @ v + lam * theta0)
        # variance matching: least-squares predictions are shrunk toward the
        # mean (spread ~ R * spread of targets), which systematically ranks
        # extreme (most relevant) unmigrated docs below exact-scored
        # competitors. Re-scale the affine map so predicted scores match the
        # target distribution's location AND spread on the fit sample.
        p = A @ theta
        sp, sv = float(p.std()), float(v.std())
        if sp > 1e-6 and sv > 1e-6:
            r = min(sv / sp, 3.0)
            mu_p, mu_v = float(p.mean()), float(v.mean())
            theta = theta * r
            theta[2] += mu_v - r * mu_p
        # Residuals of the FINAL affine map on this query's dual-scored
        # migrated candidates are a *calibration set* for the fusion error on
        # its unmigrated candidates.  Taking the split-conformal quantile
        # rather than a plain empirical one -- index ceil((n+1)(1-alpha)) of the
        # sorted residuals -- makes the resulting eps a distribution-free
        # (1-alpha) bound on the error of an exchangeable further candidate,
        # with no distributional assumption and no extra cost.  Exchangeability
        # between migrated and unmigrated candidates of the same query is the
        # assumption we cannot verify, so S5.9 measures whether it holds.
        resid = np.sort(np.abs(A @ theta - v))
        n = resid.shape[0]
        alpha = 1.0 - self.cfg.eps_quantile
        rank = int(np.ceil((n + 1) * (1.0 - alpha)))
        self._last_eps_hat = float(resid[-1] if rank > n else resid[rank - 1])
        return theta

    def _certified_recall_lb(self, cand: np.ndarray, scores: np.ndarray,
                             top: np.ndarray, eps: float, n_old: int) -> float:
        """Lower bound on this query's pool recall, from measurable quantities.

        A served document i can be displaced from the pool's true top-k only by
        a non-served candidate j whose *true* score could exceed i's, which
        requires ``s(j) >= s(i) - (eps_i + eps_j)``.  Migrated candidates carry
        exact scores (eps = 0), unmigrated ones carry the fusion-error surrogate
        ``eps``.  Counting, for every served document, the non-served candidates
        that clear its threshold gives the number of served documents that are
        provably in the true top-k --- a graded certificate whose value 1.0 is
        exactly the classical all-or-nothing margin condition.
        """
        if not np.isfinite(eps):
            return float("nan")
        k = top.shape[0]
        if k == 0:
            return float("nan")
        served_scores = scores[top]
        mask = np.ones(scores.shape[0], dtype=bool)
        mask[top] = False
        if not mask.any():
            return 1.0
        is_unmig = np.zeros(scores.shape[0], dtype=bool)
        is_unmig[:n_old] = True
        eps_i = np.where(top < n_old, eps, 0.0)   # migrated candidates are exact
        other = scores[mask]
        other_unmig = is_unmig[mask]
        exact_others = np.sort(other[~other_unmig])[::-1]
        fuzzy_others = np.sort(other[other_unmig])[::-1]
        # thresholds: exact competitors need s(j) >= s(i) - eps_i;
        #             fused competitors need s(j) >= s(i) - eps_i - eps
        t_exact = served_scores - eps_i
        t_fuzzy = served_scores - eps_i - eps
        n_e = np.searchsorted(-exact_others, -t_exact, side="right")
        n_f = np.searchsorted(-fuzzy_others, -t_fuzzy, side="right")
        return float(np.mean((n_e + n_f) == 0))

    def serve(self, q_old: np.ndarray | None, q_new: np.ndarray,
              doc_filter: np.ndarray | None = None,
              shadow_verify: bool = False) -> ServedResult:
        """Serve one query.

        q_old: old-model query embedding (None in new_only mode -> reverse adapter).
        q_new: new-model query embedding.
        doc_filter: optional bool[N] predicate mask (True = eligible).
        shadow_verify: EVALUATION ONLY — apply the configured query-time
            verification rescoring via an oracle peek WITHOUT charging budget
            or mutating migration state. Measures the quality of the
            verification-enabled query path counterfactually; the live
            workload's verification spend is still charged normally.
        """
        cfg = self.cfg
        k, K = cfg.k, cfg.k * cfg.overfetch
        q_new = l2_normalize(np.asarray(q_new, dtype=np.float32))

        with self.timer.track("route_query"):
            if q_old is None:
                if self.bank.g is None:
                    raise ValueError("new_only mode requires a reverse adapter")
                q_old = self.bank.g(q_new)
            q_old = l2_normalize(np.asarray(q_old, dtype=np.float32))

        migrated = self.state.migrated
        unmig_mask = ~migrated
        if doc_filter is not None:
            old_mask = unmig_mask & doc_filter
            new_mask = migrated & doc_filter
        else:
            old_mask, new_mask = unmig_mask, migrated

        with self.timer.track("search_old"):
            # When few unmigrated docs remain, the masked ANN search would
            # escalate to a near-full scan; score the survivors directly
            # instead (the serving analogue of per-partition NEW_ONLY cutover:
            # the old index stops being probed as migration completes).
            n_unmig = int(old_mask.sum())
            if n_unmig == 0:
                ids_old = np.empty(0, dtype=np.int64)
            elif n_unmig <= K:
                ids_old = np.flatnonzero(old_mask).astype(np.int64)
            else:
                ids_old, _ = self.old_index.search(q_old, K, mask=old_mask)
        with self.timer.track("search_new"):
            ids_new, _ = self.new_index.search(q_new, K, mask=new_mask if doc_filter is not None else None)
            # the new index only ever contains migrated docs; extra mask only for filters

        # ---- verification rerank: one common (new-space) score scale -------
        # Migrated candidates are scored exactly. Unmigrated candidates carry
        # two evidence scores — exact old-space cosine and adapted new-space
        # cosine — which are FUSED onto the exact-score scale via calibration
        # weights fitted per query on dual-scored migrated candidates (global
        # seed-sample fit as fallback/shrinkage prior).
        with self.timer.track("rerank"):
            cand = np.concatenate([ids_old, ids_new])
            scores = np.empty(cand.shape[0], dtype=np.float32)
            n_old = ids_old.shape[0]
            if ids_new.shape[0]:
                scores[n_old:] = self.store.peek_revealed(ids_new) @ q_new
            self._last_eps_hat = float("nan")
            if n_old:
                theta = self._fusion_theta(ids_new, scores[n_old:], q_old, q_new)
                u1 = self.store.old[ids_old] @ q_old
                u2 = self.bank.f(self.store.old[ids_old]) @ q_new
                scores[:n_old] = (theta[0] * u1 + theta[1] * u2
                                  + theta[2]).astype(np.float32)

        # ---- optional query-time verification ------------------------------
        verified = np.empty(0, dtype=np.int64)
        if cfg.verify_per_query > 0 and n_old:
            with self.timer.track("verify"):
                order = np.argsort(-scores[:n_old], kind="stable")[:cfg.verify_per_query]
                v_ids = ids_old[order]
                if shadow_verify:
                    true_new = self.store.oracle_full()[v_ids]  # eval-only peek
                else:
                    true_new = self.store.reveal(v_ids, reason="verify")
                    verified = v_ids
                scores[order] = true_new @ q_new

        with self.timer.track("finalize"):
            top = topk_indices(scores, k)
            final_ids, final_scores = cand[top], scores[top]
            # margin around rank k: gap between k-th kept and best dropped candidate
            if scores.shape[0] > k:
                kth = final_scores[-1]
                rest = np.delete(scores, top)
                margin = float(kth - rest.max()) if rest.size else 1.0
            else:
                margin = 1.0
        eps_hat = getattr(self, "_last_eps_hat", float("nan"))
        if n_old == 0:
            eps_hat = 0.0          # everything exact: no fusion error at all
        certified = bool(np.isfinite(eps_hat) and margin > 2.0 * eps_hat)
        cert_lb = self._certified_recall_lb(cand, scores, top, eps_hat, n_old)
        return ServedResult(ids=final_ids, scores=final_scores, candidates=cand,
                            margin=margin, n_from_old=n_old, n_from_new=ids_new.shape[0],
                            verified_ids=verified, eps_hat=eps_hat, margin_k=margin,
                            certified_topk=certified, cert_recall_lb=cert_lb,
                            cand_scores=scores)
