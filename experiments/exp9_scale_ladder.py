"""Exp 9: how the mechanisms behave as the corpus grows.

The two forward-looking claims in this work -- that prioritized ordering pays
off more as the corpus grows, and that warm-start construction saves more as
the corpus grows -- were previously two-point trends across two *different*
BEIR datasets, so scale was confounded with corpus.  This experiment holds the
dataset, the encoder pair and the workload fixed and sub-samples ONE prepared
bundle at increasing sizes, which is the only way to read a scale trend.

For every rung of the ladder it runs the full timeline for a set of methods
(default: metamorph, random order, and the offline-popularity-prior baseline)
under several seeds, and records the quantities the scale claims rest on:

  * top-10 agreement with the destination on live traffic at 25% / 50% budget
  * budget to reach 95% of destination NDCG@10 on held-out queries
  * budget at which the certifier fires
  * hot-set concentration: what fraction of the corpus the live workload
    actually touches (the mechanism behind the prioritization trend)

  python experiments/exp9_scale_ladder.py --dataset msmarco --scale xxl \
      --sizes 100000 300000 1000000 3000000 --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np

from common import (add_common_args, base_meta, budget_to_reach, default_engine_config,
                    default_out, load_bundle, resolve_dataset, write_results)

from metamorph.baselines import attach_prior, make_engine
from metamorph.workload import ZipfStream, split_queries


def at_budget(timeline: list[dict], col: str, b: float) -> float:
    """Value of `col` interpolated to exactly budget fraction `b`."""
    xs = np.array([r["budget_frac"] for r in timeline], dtype=float)
    ys = np.array([r.get(col, np.nan) for r in timeline], dtype=float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    if ok.sum() < 2:
        return float("nan")
    return float(np.interp(b, xs[ok], ys[ok]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--sizes", nargs="*", type=int, required=True)
    ap.add_argument("--methods", nargs="*",
                    default=["metamorph", "random_order", "popularity_prior"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[42])
    ap.add_argument("--max-queries", type=int, default=4000)
    ap.add_argument("--eval-queries", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=10_000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    dataset = resolve_dataset(args)
    rows: list[dict] = []
    timeline_rows: list[dict] = []
    for n_docs in args.sizes:
        data = load_bundle(dataset, args.pair, args.scale, n_docs=n_docs,
                           seed=args.seed, max_queries=args.max_queries)
        scale_key = "xxl" if data.n_docs > 1_000_000 else ("xl" if data.n_docs > 60_000
                                                           else "full")
        base_cfg = default_engine_config(data.n_docs, scale_key, args.seed)
        k = base_cfg.serve.k
        rec_col = f"recall@{k}_vs_dest"
        wl_col = rec_col + "_wl"
        ndcg_col = f"ndcg@{k}"
        print(f"\n=== N={data.n_docs:,}  Q={len(data.q_ids)} "
              f"batch={base_cfg.batch_size} parts={base_cfg.n_partitions} ===", flush=True)

        for seed in args.seeds:
            cfg = copy.deepcopy(base_cfg)
            cfg.seed = seed
            workload_qids, eval_qids = split_queries(data.q_ids, seed=seed)
            eval_qids = eval_qids[:args.eval_queries]
            from metamorph.baselines import endpoint_quality
            dest = endpoint_quality(data, eval_qids, k, space="new")

            # hot-set concentration: the mechanism behind the scale trend
            hot = None
            for method in args.methods:
                t0 = time.time()
                engine = make_engine(data, cfg, method)
                attach_prior(engine, data, method, workload_qids, eval_qids, seed)
                stream = ZipfStream(workload_qids, alpha=1.1, seed=seed)
                tl = engine.run(stream, eval_qids, max_steps=args.max_steps,
                                eval_max_queries=len(eval_qids))
                for r in tl:
                    r.update({"method": method, "seed": seed, "n_docs": data.n_docs})
                timeline_rows += tl
                if hot is None:
                    touched = int((engine.scheduler.traffic > 0).sum())
                    hot = touched / data.n_docs
                rows.append({
                    "n_docs": data.n_docs, "method": method, "seed": seed,
                    "wl_agree@25": at_budget(tl, wl_col, 0.25),
                    "wl_agree@50": at_budget(tl, wl_col, 0.50),
                    "heldout_agree@50": at_budget(tl, rec_col, 0.50),
                    "budget_to_95pct_dest_ndcg": budget_to_reach(
                        tl, ndcg_col, 0.95 * dest[ndcg_col]) if dest.get(ndcg_col) else float("nan"),
                    "budget_to_95pct_wl_agreement": budget_to_reach(tl, wl_col, 0.95),
                    "certified_at_frac": getattr(engine, "certified_at", None),
                    "hot_set_frac": hot,
                    "final_" + rec_col: tl[-1].get(rec_col),
                    "dest_ndcg": dest.get(ndcg_col),
                    "compact_count": engine.compact_count,
                    "runtime_s": round(time.time() - t0, 1),
                })
                print(f"  [{method} seed={seed}] wl@50={rows[-1]['wl_agree@50']:.3f} "
                      f"b95={rows[-1]['budget_to_95pct_dest_ndcg']} "
                      f"hot={hot:.4f} {rows[-1]['runtime_s']}s", flush=True)
        del data

    meta = base_meta(args, None, dataset=dataset, pair=args.pair, scale=args.scale,
                     sizes=list(args.sizes), seeds=list(args.seeds),
                     methods=list(args.methods))
    suf = f"_{args.tag}" if args.tag else ""
    write_results(rows, args.out or default_out(args, "exp9_ladder", suf), meta)
    write_results(timeline_rows, default_out(args, "exp9_ladder", suf + "_timelines"), meta)


if __name__ == "__main__":
    main()
