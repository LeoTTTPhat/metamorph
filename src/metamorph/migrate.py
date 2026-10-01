"""MigrationEngine: orchestrates the full online migration as a discrete-event
simulation over a query workload, with exact budget accounting.

One engine step =
  1. process `workload_per_step` live queries (updates scheduler signals;
     query-time verification may migrate docs),
  2. migrate the next scheduler batch (reveal new embeddings, insert into the
     incremental new-space HNSW, flip serving space),
  3. periodically run the recall certifier (its reveals are recycled),
  4. periodically evaluate quality on the held-out eval query set.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .adapters import AdapterBank
from .certify import Certifier
from .config import EngineConfig
from .index import BruteForceIndex, HnswIndex
from .metrics import mrr_at_k, ndcg_at_k, recall_at_k
from .partitions import Partitioner
from .scheduler import RandomScheduler, Scheduler, StaticPriorScheduler
from .serve import QueryRouter
from .state import MigrationState, Phase
from .store import EmbeddingStore
from .utils import CostMeter, Timer, set_seed, topk_indices


@dataclass
class MigrationData:
    """Everything the engine needs, precomputed by experiments/prepare_data.py."""

    doc_ids: list[str]
    old: np.ndarray                     # (N, d_old) doc embeddings, OLD model
    new: np.ndarray                     # (N, d_new) doc embeddings, NEW model (oracle)
    q_ids: list[str]
    q_old: np.ndarray                   # (Q, d_old) query embeddings, OLD model
    q_new: np.ndarray                   # (Q, d_new) query embeddings, NEW model
    qrels: dict[str, dict[int, float]]  # qid -> {doc_index: gain}
    meta: dict = field(default_factory=dict)

    @property
    def n_docs(self) -> int:
        return self.old.shape[0]

    def qindex(self) -> dict[str, int]:
        return {q: i for i, q in enumerate(self.q_ids)}


class MigrationEngine:
    def __init__(self, data: MigrationData, cfg: EngineConfig,
                 scheduler_kind: str = "priority", store_factory=None):
        self.data = data
        self.cfg = cfg
        self.scheduler_kind = scheduler_kind
        # store_factory(old, new, meter) -> EmbeddingStore. The default reads a
        # precomputed oracle; experiments/exp7 passes a LiveEmbeddingStore so
        # budget units become real encoder forward passes.
        self.store_factory = store_factory or EmbeddingStore
        self.rng = set_seed(cfg.seed)
        self.timer = Timer()
        self.timeline: list[dict] = []
        # optional hook: called as cert_callback(engine, report) right after a
        # certification round, BEFORE its reveals are absorbed as migrated —
        # i.e. while the serving state the certifier assessed is still live.
        # Used by exp5 to compare the estimate with ground-truth recall.
        self.cert_callback = None
        self._setup()

    # ------------------------------------------------------------------ setup
    def _setup(self) -> None:
        d, cfg = self.data, self.cfg
        self.meter = CostMeter()
        self.store = self.store_factory(d.old, d.new, self.meter)
        n = self.store.n

        self.partitioner = Partitioner(cfg.n_partitions, seed=cfg.seed).fit(self.store.old)
        self.state = MigrationState(n, self.partitioner.assignments)

        self.old_index = HnswIndex(self.store.d_old, n, M=cfg.hnsw_M,
                                   ef_construction=cfg.hnsw_ef_construction,
                                   ef_search=cfg.serve.ef_search, seed=cfg.seed)
        self.old_index.add(self.store.old, np.arange(n))
        self.new_index = HnswIndex(self.store.d_new, n, M=cfg.hnsw_M,
                                   ef_construction=cfg.hnsw_ef_construction,
                                   ef_search=cfg.serve.ef_search, seed=cfg.seed)

        # ---- seeding: stratified sample per partition, train adapters ------
        self.state.to_phase(Phase.SEEDING)
        seed_ids = self._stratified_seed(cfg.seed_fraction)
        seed_new = self.store.reveal(seed_ids, reason="seed")
        self.bank = AdapterBank(cfg.adapter, seed=cfg.seed).fit(
            self.store.old[seed_ids], seed_new)
        self._absorb_migrated(seed_ids)

        sched_cls = {"priority": Scheduler, "random": RandomScheduler,
                     "prior": StaticPriorScheduler,
                     "oracle_traffic": StaticPriorScheduler}[self.scheduler_kind]
        self.scheduler = sched_cls(n, self.partitioner.assignments, cfg.scheduler, self.rng)
        self.scheduler.set_disagreement(self.bank.disagreement(self.store.old))

        self.router = QueryRouter(self.store, self.state, self.old_index,
                                  self.new_index, self.bank, cfg.serve, self.timer)
        self.router.set_cv_adapted(seed_ids, self.bank.cv_adapted_seed)
        self.certifier = Certifier(self.store, self.state, self.router,
                                   cfg.certifier, self.rng)
        self.state.to_phase(Phase.ADAPTIVE_SERVING)
        self.budget_cap = int(np.ceil(cfg.max_budget_fraction * n))
        self.certified_at: float | None = None
        self.eval_seconds = 0.0
        self.compact_seconds = 0.0
        self.compact_count = 0
        self._old_index_live = n
        self._qidx = d.qindex()
        # destination ground truth (EVALUATION ONLY)
        self._dest_index = BruteForceIndex(self.store.oracle_full())
        self._dest_topk_cache: dict[str, np.ndarray] = {}

    def _stratified_seed(self, frac: float) -> np.ndarray:
        n = self.store.n
        target = max(len(self.bank_kinds()) + 2, int(round(frac * n)))
        ids: list[np.ndarray] = []
        for p in range(self.partitioner.n_partitions):
            members = self.partitioner.members(p)
            take = max(1, int(round(frac * members.size)))
            ids.append(self.rng.choice(members, size=min(take, members.size), replace=False))
        out = np.unique(np.concatenate(ids))
        if out.size < target:  # tiny corpora: top up randomly
            extra = self.rng.choice(np.setdiff1d(np.arange(n), out),
                                    size=min(n - out.size, target - out.size), replace=False)
            out = np.unique(np.concatenate([out, extra]))
        return out

    def bank_kinds(self) -> tuple[str, ...]:
        return self.cfg.adapter.kinds

    def _absorb_migrated(self, ids: np.ndarray) -> None:
        """Mark ids migrated and insert their (already revealed) vectors into the new index."""
        with self.timer.track("absorb"):
            fresh = self.state.mark_migrated(ids)
            if fresh.size:
                self.new_index.add(self.store.peek_revealed(fresh), fresh)

    # ------------------------------------------------------------- query path
    def serve_qid(self, qid: str, update_signals: bool = True,
                  shadow_verify: bool = False):
        i = self._qidx[qid]
        q_old = self.data.q_old[i] if self.cfg.serve.query_mode == "dual" else None
        res = self.router.serve(q_old, self.data.q_new[i],
                                shadow_verify=shadow_verify)
        if res.verified_ids.size:
            self._absorb_migrated(res.verified_ids)
        if update_signals:
            self.scheduler.observe_query(res.candidates, res.margin)
        return res

    def dest_topk(self, qid: str, k: int) -> np.ndarray:
        if qid not in self._dest_topk_cache:
            self.precompute_dest_topk([qid], k)
        return self._dest_topk_cache[qid]

    def precompute_dest_topk(self, qids: list[str], k: int,
                             chunk_q: int | None = None,
                             max_bytes: int = 2_000_000_000) -> None:
        """Exact destination top-k for many queries in one pass over the corpus.

        Brute-force destination search is memory-bandwidth bound: at 10^7 x 384
        one query streams 15 GB. Batching the queries into a single GEMM
        amortizes that pass over hundreds of queries instead of paying it per
        query, which is the difference between minutes and hours at scale.
        EVALUATION ONLY (never on the serving path).
        """
        todo = [q for q in dict.fromkeys(qids) if q not in self._dest_topk_cache]
        if not todo:
            return
        X = self.store.oracle_full()
        if chunk_q is None:      # cap the score matrix at max_bytes
            chunk_q = max(1, min(256, int(max_bytes // (4 * max(X.shape[0], 1)))))
        for s in range(0, len(todo), chunk_q):
            part = todo[s:s + chunk_q]
            Q = np.ascontiguousarray(
                self.data.q_new[[self._qidx[q] for q in part]], dtype=np.float32)
            scores = X @ Q.T                       # (N, q)
            for j, qid in enumerate(part):
                self._dest_topk_cache[qid] = topk_indices(scores[:, j], k)
            del scores

    # ------------------------------------------------------------------- eval
    def evaluate(self, q_ids: list[str], k: int | None = None,
                 suffix: str = "") -> dict[str, float]:
        """Quality of the CURRENT serving configuration (no signal updates,
        no budget spend, no state mutation). If query-time verification is
        configured (verify_per_query > 0), its rescoring is applied in SHADOW
        mode (oracle peek, uncharged) so the measured quality reflects the
        verification-enabled query path; the live workload's verification
        spend is charged normally in the timeline's budget accounting."""
        k = k or self.cfg.serve.k
        saved_verify = self.cfg.serve.verify_per_query
        try:
            recs, ndcgs, mrrs = [], [], []
            for qid in q_ids:
                res = self.serve_qid(qid, update_signals=False,
                                     shadow_verify=True)
                recs.append(recall_at_k(res.ids, self.dest_topk(qid, k), k))
                rel = self.data.qrels.get(qid)
                if rel:
                    ndcgs.append(ndcg_at_k(res.ids, rel, k))
                    mrrs.append(mrr_at_k(res.ids, rel, k))
            out = {f"recall@{k}_vs_dest{suffix}": float(np.mean(recs)) if recs else float("nan"),
                   f"ndcg@{k}{suffix}": float(np.mean(ndcgs)) if ndcgs else float("nan"),
                   f"mrr@{k}{suffix}": float(np.mean(mrrs)) if mrrs else float("nan")}
            return out
        finally:
            self.cfg.serve.verify_per_query = saved_verify

    def maybe_compact_old_index(self) -> bool:
        """Rebuild the old-space index over surviving unmigrated documents.

        Serving both spaces from one shared, filterable old index makes masked
        ANN search escalate as the unmigrated set shrinks: hnswlib overfetches
        and post-filters, so the fetch size grows like 1/|U_t|. Rebuilding
        geometrically (each time the live set halves) keeps every search on a
        dense index; because each rebuild inserts |U_t| vectors and |U_t| at
        least halves, total rebuild work is O(N) inserts over the migration.
        """
        ratio = self.cfg.compact_old_index_ratio
        if ratio <= 0:
            return False
        live = int((~self.state.migrated).sum())
        if live == 0 or live > ratio * self._old_index_live:
            return False
        t0 = time.perf_counter()
        ids = np.flatnonzero(~self.state.migrated).astype(np.int64)
        idx = HnswIndex(self.store.d_old, max(1, ids.size), M=self.cfg.hnsw_M,
                        ef_construction=self.cfg.hnsw_ef_construction,
                        ef_search=self.cfg.serve.ef_search, seed=self.cfg.seed)
        idx.add(self.store.old[ids], ids)
        self.old_index = idx
        self.router.old_index = idx
        self._old_index_live = live
        self.compact_seconds += time.perf_counter() - t0
        self.compact_count += 1
        return True

    def build_traffic_prior(self, q_old_mat: np.ndarray, weights: np.ndarray | None = None
                            ) -> None:
        """Seed the scheduler with an OFFLINE traffic prior (no new-model calls).

        Replays a historical query log against the pre-existing old-space index
        and counts candidate appearances. Used by the `prior` baseline (log from
        the live query distribution) and by the `oracle_traffic` upper bound
        (log = the held-out evaluation queries, i.e. future knowledge).
        """
        K = self.cfg.serve.k * self.cfg.serve.overfetch
        traffic = np.zeros(self.store.n, dtype=np.float32)
        for j in range(q_old_mat.shape[0]):
            ids, _ = self.old_index.search(q_old_mat[j], K)
            if ids.size:
                traffic[ids] += 1.0 if weights is None else float(weights[j])
        self.scheduler.traffic = traffic
        self.scheduler.hardness = traffic.copy()

    def true_quality(self, q_ids: list[str], k: int | None = None) -> dict[str, float]:
        """Ground-truth quality of the CURRENT configuration (EVALUATION ONLY).

        Returns top-k agreement with the destination, the *score-mass ratio*
        (the fraction of the destination top-k's exact new-space score mass the
        served list captures --- the quantity the score-mass certifier targets),
        and the NDCG ratio against the fully-migrated destination.
        """
        k = k or self.cfg.serve.k
        X = self.store.oracle_full()
        self.precompute_dest_topk(list(q_ids), k)
        agr, mass, nd_served, nd_dest = [], [], [], []
        for qid in q_ids:
            res = self.serve_qid(qid, update_signals=False, shadow_verify=True)
            dest = self.dest_topk(qid, k)
            agr.append(recall_at_k(res.ids, dest, k))
            qv = self.data.q_new[self._qidx[qid]]
            best = X[dest] @ qv + 1.0
            got = np.sort(X[res.ids] @ qv + 1.0)[::-1][:k]
            mass.append(min(1.0, float(got.sum()) / float(best.sum())) if best.sum() > 0 else 1.0)
            rel = self.data.qrels.get(qid)
            if rel:
                nd_served.append(ndcg_at_k(res.ids, rel, k))
                nd_dest.append(ndcg_at_k(dest, rel, k))
        nd_ratio = (float(np.sum(nd_served)) / float(np.sum(nd_dest))
                    if nd_dest and np.sum(nd_dest) > 0 else float("nan"))
        return {"true_recall": float(np.mean(agr)) if agr else float("nan"),
                "true_score_mass": float(np.mean(mass)) if mass else float("nan"),
                "true_ndcg_ratio": min(1.0, nd_ratio) if nd_ratio == nd_ratio else float("nan")}

    # -------------------------------------------------------------------- run
    def run(self, workload_stream, eval_qids: list[str],
            max_steps: int = 10_000, stop_when_certified: bool = False,
            eval_max_queries: int = 200) -> list[dict]:
        cfg = self.cfg
        eval_qids = list(eval_qids)[:eval_max_queries]
        # a fixed sample of the workload's UNIQUE queries: quality on the live
        # traffic distribution (where prioritization pays off) is recorded
        # alongside held-out quality (columns suffixed _wl)
        self._wl_eval_qids = list(workload_stream.ids)[:min(100, eval_max_queries)]
        # one batched pass over the corpus for every query we will ever score
        # against the destination (see precompute_dest_topk)
        self.precompute_dest_topk(eval_qids + self._wl_eval_qids, cfg.serve.k)
        cert_boost = np.zeros(self.state.n_partitions, dtype=np.float32)
        step = 0
        self._record(step, None, eval_qids)  # post-seed baseline row

        while step < max_steps:
            step += 1
            # 1. live workload
            for qid in workload_stream.next_batch(cfg.workload_per_step):
                self.serve_qid(qid, update_signals=True)
            self.scheduler.decay()

            # 2. certification (before the migration batch, so the cert_boost
            #    it produces informs the very next batch, and a certification
            #    fire is recorded at the pre-batch budget it actually needed)
            report = None
            if (cfg.certifier.enabled and step % cfg.certifier.period_steps == 0
                    and self.meter.total >= cfg.certifier.min_budget_frac * self.store.n
                    and self.meter.total < self.budget_cap):
                w_ids = workload_stream.ids
                idx = [self._qidx[q] for q in w_ids]
                q_old_pool = self.data.q_old[idx] if cfg.serve.query_mode == "dual" else None
                report = self.certifier.certify(q_old_pool, self.data.q_new[idx])
                if self.cert_callback is not None:
                    self.cert_callback(self, report)
                self._absorb_migrated(self.certifier.pop_recycled())
                cert_boost = np.minimum(report.miss_mass_by_partition.astype(np.float32), 1.0)
                self.scheduler.set_cert_boost(cert_boost)
                if report.certified and self.certified_at is None:
                    self.certified_at = self.meter.total / self.store.n
                    if self.state.phase == Phase.ADAPTIVE_SERVING:
                        self.state.to_phase(Phase.QUALITY_COMPLETE)

            # 3. migrate next batch under remaining budget
            remaining = self.budget_cap - self.meter.total
            if remaining > 0 and not self.state.migrated.all():
                batch = self.scheduler.next_batch(self.state.migrated,
                                                  min(cfg.batch_size, remaining))
                if batch.size:
                    self.store.reveal(batch, reason="scheduler")
                    self._absorb_migrated(batch)
            self.maybe_compact_old_index()

            # 4. evaluation + timeline row
            if step % cfg.eval_every_steps == 0:
                self._record(step, report, eval_qids)

            self.state.check_invariants()
            if stop_when_certified and self.certified_at is not None:
                break
            if self.state.migrated.all() or self.meter.total >= self.budget_cap:
                # exit certification: confirm final quality if never certified
                if cfg.certifier.enabled and self.certified_at is None:
                    w_ids = workload_stream.ids
                    idx = [self._qidx[q] for q in w_ids]
                    q_old_pool = (self.data.q_old[idx]
                                  if cfg.serve.query_mode == "dual" else None)
                    report = self.certifier.certify(q_old_pool, self.data.q_new[idx])
                    if self.cert_callback is not None:
                        self.cert_callback(self, report)
                    self._absorb_migrated(self.certifier.pop_recycled())
                    if report.certified:
                        self.certified_at = self.meter.total / self.store.n
                        if self.state.phase == Phase.ADAPTIVE_SERVING:
                            self.state.to_phase(Phase.QUALITY_COMPLETE)
                self._record(step, report, eval_qids)
                break
        return self.timeline

    def _record(self, step: int, report, eval_qids: list[str]) -> None:
        row: dict = {
            "step": step,
            "wall_time": time.time(),
            "budget_units": self.meter.total,
            "budget_frac": self.meter.total / self.store.n,
            "migrated_frac": self.state.migrated_fraction,
            "phase": self.state.phase.value,
            "n_parts_new_only": sum(1 for s in self.state.part_state if s.value == "NEW_ONLY"),
            "cert_recall_hat": report.recall_hat if report else np.nan,
            "cert_lb": report.lower_bound if report else np.nan,
            "cert_lb_pooled": report.pooled_lb if report else np.nan,
            "cert_n_pooled": report.n_pooled if report else np.nan,
            "cert_score_mass_hat": report.score_mass_hat if report else np.nan,
            "cert_score_mass_lb": report.score_mass_lb if report else np.nan,
            "cert_raw_lb": report.raw_pooled_lb if report else np.nan,
            "cert_bias_hat": report.bias_hat if report else np.nan,
            "cert_bias_ub": report.bias_ub if report else np.nan,
            "cert_eb_lb": report.eb_pooled_lb if report else np.nan,
            "certified_at_frac": self.certified_at if self.certified_at is not None else np.nan,
            "lat_ms_search_old": self.timer.mean_ms("search_old"),
            "lat_ms_search_new": self.timer.mean_ms("search_new"),
            "lat_ms_rerank": self.timer.mean_ms("rerank"),
            "compact_count": self.compact_count,
            "compact_seconds": round(self.compact_seconds, 3),
        }
        for reason, units in self.meter.snapshot().items():
            row[f"cost_{reason}"] = units
        _t0 = time.perf_counter()
        row.update(self.evaluate(eval_qids))
        if getattr(self, "_wl_eval_qids", None):
            row.update(self.evaluate(self._wl_eval_qids, suffix="_wl"))
        # the offline evaluation harness is not part of the system: a
        # deployment never runs it. Timed separately so exp7 can subtract it.
        self.eval_seconds += time.perf_counter() - _t0
        row["eval_seconds_cum"] = self.eval_seconds
        self.timeline.append(row)
