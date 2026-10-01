"""Exp 5: online quality certification — estimator bias, coverage, fire timing.

For each configuration (sample size m, pool depth C, bound family, target
metric, pooling window W) and each rep seed, run a full Metamorph migration.
At every certification round we record, via the engine's cert_callback (which
fires BEFORE the round's reveals are absorbed as migrated, i.e. while the
serving state the certifier assessed is still live):

  R_hat        the certifier's pool-restricted agreement estimate
  LB           the DECISION bound (bias-corrected, anytime-valid by default)
  raw LB       the same bound before the pool-truncation correction
  EB LB        the fixed-sample empirical-Bernstein bound, for ablation
  mass_hat/lb  the score-mass estimate and its bound
  R_true       ground-truth serving agreement vs the destination system
  mass_true    ground-truth score-mass ratio vs the destination
  ndcg_true    ground-truth NDCG ratio vs the destination
               (all three by oracle access, evaluation only)

Reported per config: SIGNED estimator bias and |error| (mean and max), coverage
= P(LB <= truth) (nominal >= 1 - delta), the budget at which certification
fires vs the budget at which truth actually crosses tau, whether it fired at
all, and certification cost.

  python experiments/exp5_certification.py --scale smoke
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np

from common import (add_common_args, base_meta, default_engine_config, default_out,
                    load_any, resolve_dataset, write_results)

from metamorph.baselines import make_engine
from metamorph.workload import ZipfStream, split_queries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--sample-sizes", nargs="*", type=int, default=[10, 30, 60])
    ap.add_argument("--pool-extras", nargs="*", type=int, default=[0, 20, 50])
    ap.add_argument("--bounds", nargs="*", default=["betting"],
                    help="confidence-bound families to sweep: betting | eb")
    ap.add_argument("--metrics", nargs="*", default=["recall"],
                    help="certification targets to sweep: recall | score_mass")
    ap.add_argument("--windows", nargs="*", type=int, default=[0],
                    help="pooling windows W (0 = pool the whole stream)")
    ap.add_argument("--probe-frac", type=float, default=1.0)
    ap.add_argument("--reps", type=int, default=3, help="seeds per configuration")
    ap.add_argument("--true-eval-queries", type=int, default=100,
                    help="workload queries used to measure ground truth")
    ap.add_argument("--n-docs", type=int, default=None,
                    help="sub-sample a large bundle to this corpus size")
    ap.add_argument("--tag", default="", help="suffix for output files")
    ap.add_argument("--max-steps", type=int, default=10_000)
    args = ap.parse_args()

    data = load_any(resolve_dataset(args), args.pair, args.scale, n_docs=args.n_docs,
                    seed=args.seed)
    base_cfg = default_engine_config(data.n_docs, args.scale, args.seed)
    workload_qids, eval_qids = split_queries(data.q_ids, seed=args.seed)
    k = base_cfg.serve.k
    true_qids = workload_qids[:args.true_eval_queries]

    rounds_rows: list[dict] = []
    summary: list[dict] = []
    for m in args.sample_sizes:
        for C in args.pool_extras:
            for bound in args.bounds:
                for metric in args.metrics:
                    for W in args.windows:
                        for rep in range(args.reps):
                            seed = args.seed + 1000 * rep
                            cfg = copy.deepcopy(base_cfg)
                            cfg.seed = seed
                            cfg.certifier.sample_queries = m
                            cfg.certifier.pool_extra = C
                            cfg.certifier.bound = bound
                            cfg.certifier.metric = metric
                            cfg.certifier.history_window = W
                            cfg.certifier.probe_frac = args.probe_frac if C > 0 else 0.0
                            engine = make_engine(data, cfg, "metamorph")
                            rec: list[dict] = []
                            tag = dict(m=m, pool_extra=C, bound=bound,
                                       metric=metric, window=W, rep=rep)

                            def cb(eng, report, _rec=rec, _tag=tag):
                                t = eng.true_quality(true_qids, k)
                                _rec.append({
                                    **_tag,
                                    "budget_frac": eng.meter.total / eng.store.n,
                                    "migrated_frac": eng.state.migrated_fraction,
                                    "recall_hat": report.recall_hat,
                                    "lower_bound": report.lower_bound,
                                    "pooled_lb": report.pooled_lb,
                                    "raw_pooled_lb": report.raw_pooled_lb,
                                    "eb_pooled_lb": report.eb_pooled_lb,
                                    "score_mass_hat": report.score_mass_hat,
                                    "score_mass_lb": report.score_mass_lb,
                                    "bias_hat": report.bias_hat,
                                    "bias_ub": report.bias_ub,
                                    "n_probe": report.n_probe,
                                    "n_pooled": report.n_pooled,
                                    "recall_true": t["true_recall"],
                                    "score_mass_true": t["true_score_mass"],
                                    "ndcg_ratio_true": t["true_ndcg_ratio"],
                                    "cert_cost_units": report.cost_units,
                                })

                            engine.cert_callback = cb
                            t0 = time.time()
                            stream = ZipfStream(workload_qids, alpha=1.1, seed=seed)
                            engine.run(stream, eval_qids, max_steps=args.max_steps)
                            rounds_rows += rec

                            tgt = "score_mass" if metric == "score_mass" else "recall"
                            truth_col = "score_mass_true" if tgt == "score_mass" else "recall_true"
                            est_col = "score_mass_hat" if tgt == "score_mass" else "recall_hat"
                            sgn = [r[est_col] - r[truth_col] for r in rec]
                            cover = [r["pooled_lb"] <= r[truth_col] for r in rec]
                            cover_raw = [r["raw_pooled_lb"] <= r[truth_col] for r in rec]
                            tau = cfg.certifier.target_recall
                            fire = next((r["budget_frac"] for r in rec
                                         if r["pooled_lb"] >= tau), float("nan"))
                            true_cross = next((r["budget_frac"] for r in rec
                                               if r[truth_col] >= tau), float("nan"))
                            cert_cost = engine.meter.by_reason.get("certify", 0)
                            summary.append({
                                **tag, "n_rounds": len(rec),
                                "signed_bias": float(np.mean(sgn)) if sgn else float("nan"),
                                "mean_abs_err": float(np.mean(np.abs(sgn))) if sgn else float("nan"),
                                "max_abs_err": float(np.max(np.abs(sgn))) if sgn else float("nan"),
                                "frac_overestimate": float(np.mean(np.array(sgn) > 0)) if sgn else float("nan"),
                                "ci_coverage_pooled": float(np.mean(cover)) if cover else float("nan"),
                                "ci_coverage_uncorrected": float(np.mean(cover_raw)) if cover_raw else float("nan"),
                                "mean_slack": float(np.mean([r[est_col] - r["pooled_lb"] for r in rec])) if rec else float("nan"),
                                "mean_bias_ub": float(np.nanmean([r["bias_ub"] for r in rec])) if rec else float("nan"),
                                "fired": bool(np.isfinite(fire)),
                                "fire_budget_frac": fire,
                                "true_cross_budget_frac": true_cross,
                                "fire_lead": fire - true_cross,
                                "cert_cost_frac": cert_cost / data.n_docs,
                                "runtime_s": round(time.time() - t0, 2),
                            })
                            print(f"[{tag}] {summary[-1]}", flush=True)

    meta = base_meta(args, data, delta=base_cfg.certifier.delta,
                     tau=base_cfg.certifier.target_recall,
                     probe_frac=args.probe_frac)
    suf = f"_{args.tag}" if args.tag else ""
    write_results(rounds_rows, args.out or default_out(args, "exp5_cert", suf), meta)
    write_results(summary, default_out(args, "exp5_cert", suf + "_summary"), meta)


if __name__ == "__main__":
    main()
