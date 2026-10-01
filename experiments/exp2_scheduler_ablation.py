"""Exp 2: scheduler signal ablation.

Variants (all run the full Metamorph engine; only the scheduler differs):
  full               traffic + hardness + disagreement (+ cert boost)
  traffic_only       w_h = w_d = 0
  hardness_only      w_t = w_d = 0
  disagreement_only  w_t = w_h = 0
  no_signals         all weights 0 except the base floor (uniform priorities)
  random             RandomScheduler baseline
  partition_major    full signals, whole-partition completion order

Output: timeline CSV (variant column) + summary CSV (budget-to-95/99%).

  python experiments/exp2_scheduler_ablation.py --scale smoke
"""
from __future__ import annotations

import argparse
import copy
import time

from common import (add_common_args, base_meta, budget_to_reach, default_engine_config,
                    default_out, load_any, resolve_dataset, write_results)

from metamorph.baselines import attach_prior, make_engine
from metamorph.workload import ZipfStream, split_queries

VARIANTS = ("full", "traffic_only", "hardness_only", "disagreement_only",
            "no_signals", "random", "partition_major", "no_anneal",
            "anneal_perdoc", "popularity_prior", "oracle_traffic")


def variant_config(base_cfg, variant: str):
    cfg = copy.deepcopy(base_cfg)
    s = cfg.scheduler
    if variant == "traffic_only":
        s.w_hardness = s.w_disagreement = 0.0
    elif variant == "hardness_only":
        s.w_traffic = s.w_disagreement = 0.0
        s.w_hardness = 1.0
    elif variant == "disagreement_only":
        s.w_traffic = s.w_hardness = 0.0
    elif variant == "no_signals":
        s.w_traffic = s.w_hardness = s.w_disagreement = 0.0
        s.cert_boost = 0.0
    elif variant == "partition_major":
        s.partition_major = True
    elif variant == "no_anneal":
        s.anneal_disagreement = False
    elif variant == "anneal_perdoc":
        s.anneal_disagreement = "per_doc"
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--n-docs", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-steps", type=int, default=10_000)
    args = ap.parse_args()

    data = load_any(resolve_dataset(args), args.pair, args.scale, n_docs=args.n_docs,
                    seed=args.seed)
    base_cfg = default_engine_config(data.n_docs, args.scale, args.seed)
    rec_col = f"recall@{base_cfg.serve.k}_vs_dest"
    wl_col = rec_col + "_wl"
    seeds = args.seeds or [args.seed]

    rows, summary = [], []
    for seed in seeds:
        workload_qids, eval_qids = split_queries(data.q_ids, seed=seed)
        for variant in args.variants:
            t0 = time.time()
            cfg = variant_config(base_cfg, variant)
            cfg.seed = seed
            method = {"random": "random_order",
                      "popularity_prior": "popularity_prior",
                      "oracle_traffic": "oracle_traffic"}.get(variant, "metamorph")
            engine = make_engine(data, cfg, method)
            attach_prior(engine, data, method, workload_qids, eval_qids, seed)
            stream = ZipfStream(workload_qids, alpha=1.1, seed=seed)
            timeline = engine.run(stream, eval_qids, max_steps=args.max_steps)
            for r in timeline:
                r["variant"] = variant
                r["seed"] = seed
            rows += timeline
            summary.append({
                "variant": variant, "seed": seed,
                "runtime_s": round(time.time() - t0, 2),
                "budget_to_95pct_dest_recall": budget_to_reach(timeline, rec_col, 0.95),
                "budget_to_99pct_dest_recall": budget_to_reach(timeline, rec_col, 0.99),
                "budget_to_95pct_wl_agreement": budget_to_reach(timeline, wl_col, 0.95),
                "certified_at_frac": engine.certified_at,
                "final_" + rec_col: timeline[-1].get(rec_col)})
            print(f"[seed {seed}][{variant}] {summary[-1]}", flush=True)

    meta = base_meta(args, data, variants=list(args.variants), seeds=seeds)
    suf = f"_{args.tag}" if args.tag else ""
    write_results(rows, args.out or default_out(args, "exp2_sched", suf), meta)
    write_results(summary, default_out(args, "exp2_sched", suf + "_summary"), meta)


if __name__ == "__main__":
    main()
