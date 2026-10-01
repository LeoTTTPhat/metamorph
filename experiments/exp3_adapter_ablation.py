"""Exp 3: adapter ablation.

Part A (adapter quality, exact search): fit each adapter kind (procrustes /
ridge / mlp / ensemble-mean) on a stratified seed sample of size
seed_frac * N, map ALL old doc embeddings into the new space, and evaluate
exact retrieval against the destination system. Swept over seed fractions.
This isolates adapter quality from ANN/serving effects.

Part B (serving-stack effect): full Metamorph engine with each adapter as the
serving primary, plus query_mode dual vs new_only (new_only exercises the
reverse adapter g: new->old for routing). Reports budget-to-95%.

Dimension-mismatch pairs (d_old != d_new) are exercised by --pair albert-e5.

  python experiments/exp3_adapter_ablation.py --scale smoke
  python experiments/exp3_adapter_ablation.py --scale small --pair albert-e5
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np

from common import (add_common_args, base_meta, budget_to_reach, default_engine_config,
                    default_out, load_prepared, resolve_dataset, write_results)

from metamorph.adapters import AdapterBank, make_adapter
from metamorph.baselines import endpoint_quality, make_engine
from metamorph.config import AdapterConfig
from metamorph.index import BruteForceIndex
from metamorph.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from metamorph.utils import l2_normalize, set_seed
from metamorph.workload import ZipfStream, split_queries

KINDS = ("procrustes", "ridge", "mlp", "ensemble")


def stratified_sample(n: int, frac: float, rng: np.random.Generator) -> np.ndarray:
    take = max(8, int(round(frac * n)))
    return rng.choice(n, size=min(take, n), replace=False)


def adapter_map_all(kind: str, cfg: AdapterConfig, X_old, X_new_seed, seed_ids, seed: int):
    """Fit on the seed pairs; return mapped embeddings for ALL docs + fit time."""
    t0 = time.time()
    if kind == "ensemble":
        outs = []
        for k in ("procrustes", "ridge", "mlp"):
            a = make_adapter(k, cfg, seed).fit(X_old[seed_ids], X_new_seed)
            outs.append(a(X_old))
        mapped = l2_normalize(np.mean(outs, axis=0))
    else:
        a = make_adapter(kind, cfg, seed).fit(X_old[seed_ids], X_new_seed)
        mapped = a(X_old)
    return mapped, time.time() - t0


def eval_mapped(mapped, data, eval_qids, k: int) -> dict:
    """Exact retrieval over adapter-mapped docs, scored against destination."""
    idx = BruteForceIndex(mapped)
    dest = BruteForceIndex(data.new)
    qidx = data.qindex()
    recs, ndcgs, mrrs = [], [], []
    for qid in eval_qids:
        i = qidx[qid]
        q = l2_normalize(data.q_new[i])
        ids, _ = idx.search(q, k)
        dest_ids, _ = dest.search(q, k)
        recs.append(recall_at_k(ids, dest_ids, k))
        rel = data.qrels.get(qid)
        if rel:
            ndcgs.append(ndcg_at_k(ids, rel, k))
            mrrs.append(mrr_at_k(ids, rel, k))
    return {"recall_vs_dest": float(np.mean(recs)),
            "ndcg": float(np.mean(ndcgs)) if ndcgs else float("nan"),
            "mrr": float(np.mean(mrrs)) if mrrs else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--kinds", nargs="*", default=list(KINDS))
    ap.add_argument("--seed-fracs", nargs="*", type=float,
                    default=[0.005, 0.01, 0.02, 0.05, 0.10])
    ap.add_argument("--engine-kinds", nargs="*", default=["procrustes", "ridge", "mlp"],
                    help="adapters to run through the full engine (part B)")
    ap.add_argument("--query-modes", nargs="*", default=["dual", "new_only"])
    ap.add_argument("--skip-engine", action="store_true", help="only part A")
    ap.add_argument("--max-steps", type=int, default=10_000)
    args = ap.parse_args()

    data = load_prepared(resolve_dataset(args), args.pair, args.scale)
    cfg = default_engine_config(data.n_docs, args.scale, args.seed)
    k = cfg.serve.k
    workload_qids, eval_qids = split_queries(data.q_ids, seed=args.seed)
    rng = set_seed(args.seed)
    dest_q = endpoint_quality(data, eval_qids, k, space="new")
    old_q = endpoint_quality(data, eval_qids, k, space="old")
    print(f"endpoints: old={old_q} new={dest_q}")

    # ---------------- part A: adapter quality vs seed fraction --------------
    rows_a = []
    acfg = cfg.adapter
    for frac in args.seed_fracs:
        seed_ids = stratified_sample(data.n_docs, frac, rng)
        X_new_seed = l2_normalize(data.new[seed_ids])
        X_old = l2_normalize(data.old)
        for kind in args.kinds:
            mapped, fit_s = adapter_map_all(kind, acfg, X_old, X_new_seed,
                                            seed_ids, args.seed)
            row = {"part": "A", "kind": kind, "seed_frac": frac,
                   "n_seed": int(seed_ids.size), "fit_seconds": round(fit_s, 3),
                   "d_old": data.old.shape[1], "d_new": data.new.shape[1]}
            row.update(eval_mapped(mapped, data, eval_qids, k))
            row["dest_ndcg"] = dest_q.get(f"ndcg@{k}")
            rows_a.append(row)
            print(f"[A] kind={kind:10s} frac={frac:.3f} "
                  f"recall_vs_dest={row['recall_vs_dest']:.3f} ndcg={row['ndcg']:.3f}")

    # ---------------- part B: full engine, adapter as serving primary -------
    rows_b, summary = [], []
    if not args.skip_engine:
        rec_col = f"recall@{k}_vs_dest"
        for kind in args.engine_kinds:
            for qmode in args.query_modes:
                ecfg = copy.deepcopy(cfg)
                ecfg.adapter.primary = kind
                if kind not in ecfg.adapter.kinds:
                    ecfg.adapter.kinds = tuple(ecfg.adapter.kinds) + (kind,)
                ecfg.serve.query_mode = qmode
                t0 = time.time()
                engine = make_engine(data, ecfg, "metamorph")
                stream = ZipfStream(workload_qids, alpha=1.1, seed=args.seed)
                timeline = engine.run(stream, eval_qids, max_steps=args.max_steps)
                for r in timeline:
                    r.update({"part": "B", "kind": kind, "query_mode": qmode})
                rows_b += timeline
                summary.append({
                    "kind": kind, "query_mode": qmode,
                    "runtime_s": round(time.time() - t0, 2),
                    "budget_to_95pct_dest_recall": budget_to_reach(timeline, rec_col, 0.95),
                    "budget_to_99pct_dest_recall": budget_to_reach(timeline, rec_col, 0.99),
                    "certified_at_frac": engine.certified_at,
                    "final_recall_vs_dest": timeline[-1].get(rec_col)})
                print(f"[B] kind={kind} mode={qmode} -> {summary[-1]}")

    meta = base_meta(args, data, endpoints={"old": old_q, "new": dest_q})
    write_results(rows_a, args.out or default_out(args, "exp3_adapters"), meta)
    if rows_b:
        write_results(rows_b, default_out(args, "exp3_adapters", "_engine"), meta)
        write_results(summary, default_out(args, "exp3_adapters", "_summary"), meta)


if __name__ == "__main__":
    main()
