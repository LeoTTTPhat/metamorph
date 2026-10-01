"""Exp 7: what a budget unit actually costs.

Everywhere else the new-model embeddings come from a precomputed oracle, so the
budget meter is an abstraction.  This experiment measures the thing the system
claims to save.

Part A -- encoder throughput.  Real forward passes over real corpus text for
both models of the pair, on the accelerator and on CPU, at several batch sizes.
Yields docs/s and s/1k-docs, plus the corpus's own length distribution.

Part B -- live end-to-end migration.  The same MigrationEngine, but the store
is a :class:`~metamorph.live.LiveEmbeddingStore`: every budget unit runs the
new encoder for real.  Reports wall-clock split into encoder / index / serving
time for the progressive migration and for re-embed-then-swap, so budget
fractions convert into hours.  The precomputed oracle is retained for
evaluation only and the live vectors are checked against it.

Part C -- cost projection.  Measured docs/s on this machine, plus a projection
to datacenter accelerators and to public embedding-API list prices, so a budget
fraction reads as GPU-hours and dollars at 10^7-10^9 documents.

  python experiments/exp7_encoder_cost.py --dataset quora --scale xl --n-docs 200000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from common import (MODEL_PAIRS, add_common_args, base_meta, bundle_path,
                    default_engine_config, default_out, load_any, resolve_dataset,
                    write_results)

from metamorph.baselines import make_engine
from metamorph.live import LiveEmbeddingStore
from metamorph.workload import ZipfStream, split_queries

# Public list prices per 1M tokens for hosted embedding APIs (USD, 2026-08).
# Used only to convert a measured token count into an order-of-magnitude dollar
# figure; we report the arithmetic, not a vendor recommendation.
API_PRICE_PER_MTOK = {"budget-tier": 0.02, "mid-tier": 0.13}
# Reference accelerator throughput multipliers vs the measured device, from
# peak-FLOPs ratios for fp16 dense matmul. Explicitly a projection.
ACCEL_SPEEDUP = {"M3 Ultra (MPS, measured)": 1.0, "A100-80GB (projected)": 4.0,
                 "H100-80GB (projected)": 8.0}


def corpus_texts(dataset: str, pair: str, scale: str, n: int | None) -> list[str] | None:
    p = bundle_path(dataset, pair, scale) / "doc_texts.jsonl"
    if not p.exists():
        return None
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def bench_throughput(model_name: str, texts: list[str], kind: str, device: str,
                     batch_sizes: list[int], n: int) -> list[dict]:
    from sentence_transformers import SentenceTransformer

    from metamorph.embed import PREFIXES

    prefix = PREFIXES.get(model_name, ("", ""))[1 if kind == "passage" else 0]
    inp = [prefix + t for t in texts[:n]]
    model = SentenceTransformer(model_name, device=device)
    model.encode(inp[:min(256, len(inp))], batch_size=64, convert_to_numpy=True)
    rows = []
    for bs in batch_sizes:
        t0 = time.perf_counter()
        model.encode(inp, batch_size=bs, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False)
        dt = time.perf_counter() - t0
        rows.append({"model": model_name, "device": device, "batch_size": bs,
                     "n_docs": len(inp), "seconds": round(dt, 3),
                     "docs_per_s": len(inp) / dt,
                     "mean_chars": float(np.mean([len(t) for t in inp]))})
        print(f"  {model_name} {device} bs={bs}: {len(inp)/dt:.0f} docs/s", flush=True)
    del model
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--n-docs", type=int, default=None,
                    help="corpus size for the live migration")
    ap.add_argument("--bench-docs", type=int, default=8192)
    ap.add_argument("--batch-sizes", nargs="*", type=int, default=[64, 256, 512])
    ap.add_argument("--devices", nargs="*", default=["mps", "cpu"])
    ap.add_argument("--live-methods", nargs="*", default=["metamorph", "full_swap"])
    ap.add_argument("--max-steps", type=int, default=10_000)
    ap.add_argument("--skip-live", action="store_true")
    args = ap.parse_args()

    dataset = resolve_dataset(args)
    old_model, new_model = MODEL_PAIRS[args.pair]
    texts = corpus_texts(dataset, args.pair, args.scale, args.n_docs)
    if texts is None:
        raise SystemExit(
            f"exp7 needs a large bundle with doc_texts.jsonl; run "
            f"experiments/prepare_large.py --dataset {dataset} --pair {args.pair} "
            f"--tag {args.scale}")

    # ---------------------------------------------------------------- Part A
    print("== Part A: encoder throughput ==", flush=True)
    bench_rows: list[dict] = []
    for device in args.devices:
        for model in (old_model, new_model):
            bench_rows += bench_throughput(model, texts, "passage", device,
                                           args.batch_sizes, args.bench_docs)
    # A2: throughput as a function of REVEAL batch size. A progressive
    # migration encodes in scheduler-sized batches, not one big pass, so this
    # is what converts a budget unit into seconds for the real system.
    print("-- throughput vs reveal batch size (new model) --", flush=True)
    from sentence_transformers import SentenceTransformer

    from metamorph.embed import PREFIXES as _PFX
    _pref = _PFX.get(new_model, ("", ""))[1]
    _m = SentenceTransformer(new_model, device=args.devices[0])
    _m.encode([_pref + t for t in texts[:256]], batch_size=256, convert_to_numpy=True)
    for reveal_n in (32, 128, 512, 2048, 8192):
        reps = max(1, min(20, 16384 // reveal_n))
        t0 = time.perf_counter()
        for r in range(reps):
            chunk = texts[r * reveal_n:(r + 1) * reveal_n] or texts[:reveal_n]
            _m.encode([_pref + t for t in chunk], batch_size=min(256, reveal_n),
                      convert_to_numpy=True, normalize_embeddings=True,
                      show_progress_bar=False)
        dt = time.perf_counter() - t0
        bench_rows.append({"model": new_model, "device": args.devices[0],
                           "batch_size": min(256, reveal_n), "reveal_batch": reveal_n,
                           "n_docs": reveal_n * reps, "seconds": round(dt, 3),
                           "docs_per_s": reveal_n * reps / dt,
                           "mean_chars": float(np.mean([len(t) for t in texts[:4096]]))})
        print(f"  reveal_batch={reveal_n}: {reveal_n*reps/dt:.0f} docs/s", flush=True)
    del _m

    best = {}
    for r in bench_rows:
        if r.get("reveal_batch"):
            continue          # A2 rows are a separate sweep
        key = (r["model"], r["device"])
        if r["docs_per_s"] > best.get(key, {"docs_per_s": 0})["docs_per_s"]:
            best[key] = r
    best = {k: v for k, v in best.items()}
    new_best = max((r for (m, d), r in best.items() if m == new_model),
                   key=lambda r: r["docs_per_s"])
    print(f"best NEW-model throughput: {new_best['docs_per_s']:.0f} docs/s "
          f"({new_best['device']}, bs={new_best['batch_size']})")

    # ---------------------------------------------------------------- Part B
    live_rows: list[dict] = []
    if not args.skip_live:
        print("== Part B: live end-to-end migration (real encoder calls) ==", flush=True)
        data = load_any(dataset, args.pair, args.scale, n_docs=args.n_docs,
                        seed=args.seed)
        sub_texts = texts
        if data.n_docs != len(texts):
            # load_any sub-sampled: recover the kept rows by doc id
            full_ids = [str(x) for x in np.load(
                bundle_path(dataset, args.pair, args.scale) / "doc_ids.npy")]
            pos = {d: i for i, d in enumerate(full_ids)}
            sub_texts = [texts[pos[d]] for d in data.doc_ids]
        cfg = default_engine_config(data.n_docs, args.scale, args.seed)
        workload_qids, eval_qids = split_queries(data.q_ids, seed=args.seed)

        for method in args.live_methods:
            t_wall = time.perf_counter()
            if method == "full_swap":
                # re-embed-then-swap: the entire corpus through the new encoder,
                # then one index build. No serving benefit until it finishes.
                from metamorph.index import HnswIndex
                from metamorph.utils import CostMeter
                store = LiveEmbeddingStore(data.old, data.new, sub_texts, new_model,
                                           CostMeter(), batch_size=new_best["batch_size"])
                t0 = time.perf_counter()
                vecs = store.reveal(np.arange(data.n_docs), reason="scheduler")
                enc_s = store.encode_seconds
                t_idx = time.perf_counter()
                idx = HnswIndex(store.d_new, data.n_docs, M=cfg.hnsw_M,
                                ef_construction=cfg.hnsw_ef_construction,
                                ef_search=cfg.serve.ef_search, seed=cfg.seed)
                idx.add(vecs, np.arange(data.n_docs))
                idx_s = time.perf_counter() - t_idx
                row = {"method": method, "wall_s": time.perf_counter() - t_wall,
                       "encode_s": enc_s, "index_s": idx_s,
                       "other_s": time.perf_counter() - t_wall - enc_s - idx_s,
                       "budget_frac_at_stop": 1.0, "eval_harness_s": 0.0,
                       "system_s": time.perf_counter() - t_wall,
                       "quality_at_stop": 1.0, **store.encoder_stats()}
            else:
                def factory(old, new, meter, _t=sub_texts, _m=new_model,
                            _bs=new_best["batch_size"]):
                    return LiveEmbeddingStore(old, new, _t, _m, meter, batch_size=_bs)

                engine = make_engine(data, cfg, method, store_factory=factory)
                stream = ZipfStream(workload_qids, alpha=1.1, seed=args.seed)
                tl = engine.run(stream, eval_qids, max_steps=args.max_steps,
                                eval_max_queries=min(200, len(eval_qids)))
                st = engine.store.encoder_stats()
                enc_s = st["encode_seconds"]
                wall = time.perf_counter() - t_wall
                # the operator's stopping point: the certified budget if the
                # certifier fired, else the full migration
                stop = engine.certified_at if engine.certified_at else 1.0
                near = min(tl, key=lambda r: abs(r["budget_frac"] - stop))
                ev = getattr(engine, "eval_seconds", 0.0)
                row = {"method": method, "wall_s": wall, "encode_s": enc_s,
                       "index_s": float("nan"), "eval_harness_s": ev,
                       "system_s": wall - ev,
                       "other_s": wall - enc_s - ev,
                       "budget_frac_at_stop": stop,
                       "quality_at_stop": near.get(f"ndcg@{cfg.serve.k}"), **st}
            row.update({"n_docs": data.n_docs, "dataset": dataset, "pair": args.pair})
            live_rows.append(row)
            print(f"  [{method}] wall={row['wall_s']:.1f}s encode={row['encode_s']:.1f}s "
                  f"stop@{row['budget_frac_at_stop']:.3f} "
                  f"oracle_gap={row.get('max_oracle_cosine_gap', 0):.2e}", flush=True)

    # ---------------------------------------------------------------- Part C
    print("== Part C: cost projection ==", flush=True)
    dps = new_best["docs_per_s"]
    mean_chars = new_best["mean_chars"]
    tok_per_doc = mean_chars / 4.0        # ~4 chars/token for English subwords
    proj_rows: list[dict] = []
    for corpus_n in (10 ** 6, 10 ** 7, 10 ** 8, 10 ** 9):
        for accel, mult in ACCEL_SPEEDUP.items():
            hours = corpus_n / (dps * mult) / 3600.0
            row = {"corpus_n": corpus_n, "accelerator": accel,
                   "docs_per_s": dps * mult, "full_reembed_hours": hours}
            for tier, price in API_PRICE_PER_MTOK.items():
                row[f"api_usd_{tier.replace('-', '_')}"] = (
                    corpus_n * tok_per_doc / 1e6 * price)
            # what the measured budget-to-quality figures buy
            for b in (0.5, 0.62, 0.2):
                row[f"hours_at_budget_{int(b*100)}"] = hours * b
            proj_rows.append(row)
    for r in proj_rows:
        if r["corpus_n"] in (10 ** 7, 10 ** 9):
            print(f"  N={r['corpus_n']:.0e} {r['accelerator']}: "
                  f"{r['full_reembed_hours']:.1f} h full re-embed", flush=True)

    meta = base_meta(args, None, dataset=dataset, pair=args.pair,
                     n_docs=args.n_docs, old_model=old_model, new_model=new_model,
                     tokens_per_doc_est=tok_per_doc,
                     best_new_docs_per_s=dps)
    write_results(bench_rows, default_out(args, "exp7_throughput"), meta)
    if live_rows:
        write_results(live_rows, default_out(args, "exp7_live"), meta)
    write_results(proj_rows, default_out(args, "exp7_projection"), meta)


if __name__ == "__main__":
    main()
