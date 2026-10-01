"""Exp 6: per-query serving latency across migration stages.

For each target budget fraction (migration stage), run a fresh engine to that
budget (certifier off, so the spend is exactly the scheduler's), then probe
per-query latency: each probe query is individually wall-clocked end-to-end
through the dual-space router, and stage means (old search / new search /
adapter rerank / verify / finalize) come from the router's Timer deltas.
Also swept: query-time verification budget v (verify_per_query), which trades
extra re-embeds for exactness on the head of the ranking.

Output: one row per (budget_frac, verify) with p50/p95/p99/mean latency and
the stage breakdown.

  python experiments/exp6_latency.py --scale smoke
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np

from common import (MODEL_PAIRS, add_common_args, base_meta, bundle_path,
                    default_engine_config, default_out, load_any, resolve_dataset,
                    write_results)

from metamorph.baselines import make_engine
from metamorph.workload import ZipfStream, split_queries

# "absorb" is the new-index insert for documents that verification just
# re-embedded: real work on the query path, and large enough that leaving
# it out of the breakdown would misattribute verification's cost.
STAGES = ("route_query", "search_old", "search_new", "rerank", "verify",
          "finalize", "absorb")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--budget-fracs", nargs="*", type=float,
                    default=[0.06, 0.25, 0.5, 0.75, 1.0],
                    help="migration stages to probe (first ~= seed-only)")
    ap.add_argument("--verify", nargs="*", type=int, default=[0, 2],
                    help="verify_per_query values to probe")
    ap.add_argument("--probe-queries", type=int, default=200)
    ap.add_argument("--n-docs", type=int, default=None)
    ap.add_argument("--compaction", nargs="*", type=float, default=[0.5, 0.0],
                    help="old-index compaction ratios to compare (0 = disabled, "
                         "i.e. the shared masked index of the first version)")
    ap.add_argument("--live-encoder", action="store_true",
                    help="run verification through the REAL encoder so its "
                         "latency is measured rather than stood in for")
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-steps", type=int, default=10_000)
    args = ap.parse_args()

    data = load_any(resolve_dataset(args), args.pair, args.scale, n_docs=args.n_docs,
                    seed=args.seed)
    base_cfg = default_engine_config(data.n_docs, args.scale, args.seed)
    workload_qids, eval_qids = split_queries(data.q_ids, seed=args.seed)
    probe_qids = (eval_qids * ((args.probe_queries // len(eval_qids)) + 1)
                  )[:args.probe_queries]

    store_factory = None
    if args.live_encoder:
        import json as _json
        txt_path = bundle_path(resolve_dataset(args), args.pair, args.scale) / "doc_texts.jsonl"
        if not txt_path.exists():
            raise SystemExit("--live-encoder needs a large bundle with doc_texts.jsonl")
        with open(txt_path, encoding="utf-8") as f:
            all_texts = [_json.loads(line) for line in f]
        if len(all_texts) != data.n_docs:
            full_ids = [str(x) for x in np.load(
                bundle_path(resolve_dataset(args), args.pair, args.scale) / "doc_ids.npy")]
            pos = {d: i for i, d in enumerate(full_ids)}
            all_texts = [all_texts[pos[d]] for d in data.doc_ids]
        new_model = MODEL_PAIRS[args.pair][1]

        def store_factory(old, new, meter, _t=all_texts, _m=new_model):
            from metamorph.live import LiveEmbeddingStore
            return LiveEmbeddingStore(old, new, _t, _m, meter, batch_size=64)

    rows: list[dict] = []
    for compact in args.compaction:
      for frac in args.budget_fracs:
        cfg = copy.deepcopy(base_cfg)
        cfg.certifier.enabled = False
        cfg.max_budget_fraction = frac
        cfg.compact_old_index_ratio = compact
        t0 = time.time()
        engine = make_engine(data, cfg, "metamorph", store_factory=store_factory)
        stream = ZipfStream(workload_qids, alpha=1.1, seed=args.seed)
        engine.run(stream, eval_qids, max_steps=args.max_steps, eval_max_queries=20)
        setup_s = time.time() - t0

        for v in args.verify:
            engine.cfg.serve.verify_per_query = v
            timer = engine.router.timer
            etimer = engine.timer
            before = {s: ((etimer if s == "absorb" else timer).totals[s],
                          (etimer if s == "absorb" else timer).counts[s])
                      for s in STAGES}
            budget_before = engine.meter.total
            lat_ms = np.empty(len(probe_qids), dtype=np.float64)
            for j, qid in enumerate(probe_qids):
                t1 = time.perf_counter()
                engine.serve_qid(qid, update_signals=False)
                lat_ms[j] = 1000.0 * (time.perf_counter() - t1)
            row = {
                "compaction_ratio": compact,
                "compact_count": engine.compact_count,
                "compact_seconds": round(engine.compact_seconds, 3),
                "live_encoder": bool(args.live_encoder),
                "budget_frac_target": frac,
                "budget_frac_actual": budget_before / engine.store.n,
                "migrated_frac": engine.state.migrated_fraction,
                "verify_per_query": v,
                "n_probe": len(probe_qids),
                "lat_ms_p50": float(np.percentile(lat_ms, 50)),
                "lat_ms_p95": float(np.percentile(lat_ms, 95)),
                "lat_ms_p99": float(np.percentile(lat_ms, 99)),
                "lat_ms_mean": float(lat_ms.mean()),
                "verify_budget_spent": engine.meter.total - budget_before,
                "engine_setup_s": round(setup_s, 2),
            }
            acc = 0.0
            for s in STAGES:
                t = etimer if s == "absorb" else timer
                dt = t.totals[s] - before[s][0]
                dc = t.counts[s] - before[s][1]
                row[f"stage_ms_{s}"] = 1000.0 * dt / dc if dc else 0.0
                acc += row[f"stage_ms_{s}"]
            # the stage timers do not sum to the end-to-end mean: Python call
            # overhead, array allocation and the state flip after verification
            # sit between them. Reported rather than left implicit.
            row["stage_ms_unattributed"] = max(0.0, row["lat_ms_mean"] - acc)
            row["stage_sum_ms"] = acc
            rows.append(row)
            print(f"[compact={compact} frac={frac} v={v}] p50={row['lat_ms_p50']:.2f}ms "
                  f"p99={row['lat_ms_p99']:.2f}ms migrated={row['migrated_frac']:.2f} "
                  f"compactions={engine.compact_count}", flush=True)

    meta = base_meta(args, data, probe_queries=len(probe_qids),
                     compaction=list(args.compaction), live_encoder=args.live_encoder)
    suf = f"_{args.tag}" if args.tag else ""
    write_results(rows, args.out or default_out(args, "exp6_latency", suf), meta)


if __name__ == "__main__":
    main()
