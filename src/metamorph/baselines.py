"""Baselines.

B1 full-re-embed-then-swap: serve the OLD system unchanged until 100% budget,
   then atomically swap to the NEW system. Timeline synthesized from the two
   endpoint evaluations (that is exactly what the strategy delivers).
B2 random-order progressive: MigrationEngine(scheduler_kind="random"),
   certifier off, same serving stack.
B3 static adapter (Drift-Adapter-style): MigrationEngine with budget capped at
   the seed fraction — adapter serving forever, no progressive re-embedding.
B4 Metamorph: full engine (default config).
"""
from __future__ import annotations

import numpy as np

from .config import EngineConfig
from .index import BruteForceIndex
from .metrics import mrr_at_k, ndcg_at_k, recall_at_k
from .migrate import MigrationData, MigrationEngine
from .utils import l2_normalize


def endpoint_quality(data: MigrationData, q_ids: list[str], k: int = 10,
                     space: str = "new") -> dict[str, float]:
    """Exact single-space retrieval quality (old system or destination system)."""
    if space == "new":
        index, q_mat = BruteForceIndex(data.new), l2_normalize(data.q_new)
    else:
        index, q_mat = BruteForceIndex(data.old), l2_normalize(data.q_old)
    dest = BruteForceIndex(data.new)
    qidx = data.qindex()
    recs, ndcgs, mrrs = [], [], []
    for qid in q_ids:
        i = qidx[qid]
        ids, _ = index.search(q_mat[i], k)
        dest_ids, _ = dest.search(l2_normalize(data.q_new[i]), k)
        recs.append(recall_at_k(ids, dest_ids, k))
        rel = data.qrels.get(qid)
        if rel:
            ndcgs.append(ndcg_at_k(ids, rel, k))
            mrrs.append(mrr_at_k(ids, rel, k))
    return {f"recall@{k}_vs_dest": float(np.mean(recs)) if recs else float("nan"),
            f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else float("nan"),
            f"mrr@{k}": float(np.mean(mrrs)) if mrrs else float("nan")}


def full_swap_timeline(data: MigrationData, eval_qids: list[str], k: int = 10,
                       n_points: int = 21) -> list[dict]:
    """B1: quality is the old system's until budget hits 1.0, then the new system's."""
    q_old_sys = endpoint_quality(data, eval_qids, k, space="old")
    q_new_sys = endpoint_quality(data, eval_qids, k, space="new")
    rows = []
    for b in np.linspace(0.0, 1.0, n_points):
        src = q_new_sys if b >= 1.0 else q_old_sys
        row = {"step": int(b * (n_points - 1)), "budget_units": int(b * data.n_docs),
               "budget_frac": float(b), "migrated_frac": float(b >= 1.0),
               "phase": "SWAP" if b >= 1.0 else "OLD_SERVING",
               "method": "full_swap"}
        row.update(src)
        rows.append(row)
    return rows


def make_engine(data: MigrationData, cfg: EngineConfig, method: str,
                store_factory=None) -> MigrationEngine:
    """Configure an engine for one of the progressive methods."""
    import copy

    cfg = copy.deepcopy(cfg)
    kw = {"store_factory": store_factory}
    if method == "metamorph":
        return MigrationEngine(data, cfg, scheduler_kind="priority", **kw)
    if method == "random_order":
        cfg.certifier.enabled = False
        return MigrationEngine(data, cfg, scheduler_kind="random", **kw)
    if method == "static_adapter":
        cfg.certifier.enabled = False
        cfg.max_budget_fraction = cfg.seed_fraction + 1e-9
        cfg.serve.verify_per_query = 0
        return MigrationEngine(data, cfg, scheduler_kind="priority", **kw)
    if method in ("popularity_prior", "oracle_traffic"):
        cfg.certifier.enabled = False
        kind = "prior" if method == "popularity_prior" else "oracle_traffic"
        return MigrationEngine(data, cfg, scheduler_kind=kind, **kw)
    raise ValueError(f"unknown method {method}")


def attach_prior(engine, data: MigrationData, method: str, workload_qids: list[str],
                 eval_qids: list[str], seed: int, n_log: int = 1500,
                 alpha: float = 1.1) -> None:
    """Give a prior-driven baseline its offline traffic prior.

    ``popularity_prior``: a historical log drawn from the SAME Zipf-skewed live
    query distribution (what last month's logs look like), replayed against the
    old index — realistic and available before migration starts.
    ``oracle_traffic``: the held-out evaluation queries themselves — future
    knowledge, reported only as an upper bound on what ordering can buy.
    """
    from .workload import ZipfStream

    if method not in ("popularity_prior", "oracle_traffic"):
        return
    qidx = data.qindex()
    if method == "oracle_traffic":
        ids = [qidx[q] for q in eval_qids]
        w = None
    else:
        stream = ZipfStream(workload_qids, alpha=alpha, seed=seed + 7919)
        log = stream.next_batch(n_log)
        uniq, counts = np.unique(np.array([qidx[q] for q in log]), return_counts=True)
        ids, w = list(uniq), counts.astype(np.float32)
    engine.build_traffic_prior(data.q_old[ids], w)
