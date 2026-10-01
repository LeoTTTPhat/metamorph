"""Exp 1 (headline): retrieval quality vs re-embedding budget for all methods.

Methods:
  full_swap      B1: serve OLD unchanged, atomic swap at 100% budget
  random_order   B2: progressive re-embedding in random order (same serving stack)
  static_adapter B3: Drift-Adapter-style — adapter only, budget = seed fraction
  metamorph      B4: full system (prioritized scheduler + certifier)

Output: one long-format CSV of timeline rows (method column), plus a summary
CSV with budget-to-reach-95%/99%-of-destination-quality per method.

  python experiments/exp1_migration_timeline.py --dataset scifact --pair minilm-e5 --scale smoke
"""
from __future__ import annotations

import argparse
import time

from common import (add_common_args, base_meta, budget_to_reach, default_engine_config,
                    default_out, load_any, resolve_dataset, write_results)

from metamorph.baselines import (attach_prior, endpoint_quality, full_swap_timeline,
                                 make_engine)
from metamorph.workload import ZipfStream, split_queries

METHODS = ("full_swap", "random_order", "static_adapter", "metamorph")
# stronger ordering baselines than uniform random (B5, and an oracle upper bound)
EXTRA_METHODS = ("popularity_prior", "oracle_traffic")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    ap.add_argument("--max-steps", type=int, default=10_000)
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="run each method under several seeds (variance bands)")
    ap.add_argument("--n-docs", type=int, default=None,
                    help="sub-sample a large bundle to this corpus size")
    ap.add_argument("--tag", default="", help="extra suffix for output files")
    ap.add_argument("--verify", type=int, default=0,
                    help="query-time verification budget per query (re-embeds "
                         "top-v unmigrated candidates exactly; spends budget). "
                         "Methods are tagged _v<N>, output file suffixed.")
    args = ap.parse_args()

    data = load_any(resolve_dataset(args), args.pair, args.scale,
                    n_docs=args.n_docs, seed=args.seed)
    base_cfg = default_engine_config(data.n_docs, args.scale, args.seed)
    if args.verify:
        base_cfg.serve.verify_per_query = args.verify
    seeds = args.seeds or [args.seed]

    rows: list[dict] = []
    summary: list[dict] = []
    rec_col = f"recall@{base_cfg.serve.k}_vs_dest"
    ndcg_col = f"ndcg@{base_cfg.serve.k}"
    endpoints: dict = {}
    for seed in seeds:
        import copy
        cfg = copy.deepcopy(base_cfg)
        cfg.seed = seed
        # the workload/held-out split is part of the random draw: re-splitting
        # per seed is what makes the variance band an honest estimate of run-to-
        # run spread rather than of scheduler jitter alone
        workload_qids, eval_qids = split_queries(data.q_ids, seed=seed)
        q_old_sys = endpoint_quality(data, eval_qids, cfg.serve.k, space="old")
        q_new_sys = endpoint_quality(data, eval_qids, cfg.serve.k, space="new")
        endpoints[str(seed)] = {"old": q_old_sys, "new": q_new_sys}
        print(f"[seed {seed}] endpoints: old={q_old_sys}  new={q_new_sys}", flush=True)

        for method in args.methods:
            t0 = time.time()
            engine = None
            if method == "full_swap":
                timeline = full_swap_timeline(data, eval_qids, k=cfg.serve.k)
            else:
                engine = make_engine(data, cfg, method)
                attach_prior(engine, data, method, workload_qids, eval_qids, seed)
                stream = ZipfStream(workload_qids, alpha=1.1, seed=seed)
                # eval over the FULL eval split (endpoint thresholds in the
                # summary are computed on the same set; the engine default of
                # 200 would silently mismatch on datasets with >200 eval queries)
                timeline = engine.run(stream, eval_qids, max_steps=args.max_steps,
                                      eval_max_queries=len(eval_qids))
            mtag = f"{method}_v{args.verify}" if args.verify else method
            for r in timeline:
                r["method"] = mtag
                r["seed"] = seed
            rows += timeline
            srow = {"method": mtag, "seed": seed,
                    "runtime_s": round(time.time() - t0, 2),
                    "final_" + rec_col: timeline[-1].get(rec_col),
                    "final_" + ndcg_col: timeline[-1].get(ndcg_col),
                    "final_budget_frac": timeline[-1]["budget_frac"],
                    "dest_ndcg": q_new_sys.get(ndcg_col),
                    "old_ndcg": q_old_sys.get(ndcg_col)}
            for tgt_name, tgt in (("95", 0.95), ("99", 0.99)):
                srow[f"budget_to_{tgt_name}pct_dest_recall"] = budget_to_reach(
                    timeline, rec_col, tgt)
                if q_new_sys.get(ndcg_col):
                    srow[f"budget_to_{tgt_name}pct_dest_ndcg"] = budget_to_reach(
                        timeline, ndcg_col, tgt * q_new_sys[ndcg_col])
            wl_col = rec_col + "_wl"
            srow["budget_to_95pct_wl_agreement"] = budget_to_reach(timeline, wl_col, 0.95)
            if engine is not None:
                srow["certified_at_frac"] = getattr(engine, "certified_at", None)
                srow["cert_cost_frac"] = engine.meter.by_reason.get("certify", 0) / data.n_docs
                srow["verify_cost_frac"] = engine.meter.by_reason.get("verify", 0) / data.n_docs
            summary.append(srow)
            print(f"[seed {seed}][{method}] {srow['runtime_s']}s  "
                  f"b95_ndcg={srow.get('budget_to_95pct_dest_ndcg')}", flush=True)

    vsuf = (f"_v{args.verify}" if args.verify else "") + (f"_{args.tag}" if args.tag else "")
    meta = base_meta(args, data, endpoints=endpoints, seeds=seeds,
                     engine_config=str(base_cfg), verify_per_query=args.verify)
    write_results(rows, args.out or default_out(args, "exp1_timeline", vsuf), meta)
    write_results(summary, default_out(args, "exp1_timeline", vsuf + "_summary"), meta)


if __name__ == "__main__":
    main()
