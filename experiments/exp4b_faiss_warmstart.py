"""Exp 4b: warm-start construction inside a REAL index (FAISS IVF).

Exp 4 measures warm-start on our own NN-descent and k-means so that work can be
counted exactly, but that is not an index anyone serves from.  This experiment
runs the same idea end to end inside FAISS's IVF-Flat -- the coarse-quantizer
family behind Milvus, pgvector's ivfflat and FAISS deployments -- and reports
wall-clock build time and query recall, not distance counts.

The migration setting is what makes it free: the old index already partitions
the corpus, so the new-space centroid of each old partition is available for the
price of one pass over the vectors that have been re-embedded anyway.  Warm
start therefore *skips coarse training entirely*; the ablation adds 0, 1, 2 and
5 Lloyd refinement iterations to trace the build-time / recall trade-off.

Reported per variant: coarse-training seconds, add seconds, total build
seconds, and recall@10 versus exact search at several nprobe settings, plus
time-to-matched-recall against the cold baseline.

  python experiments/exp4b_faiss_warmstart.py --dataset quora --scale xl --n-docs 200000
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import (add_common_args, base_meta, default_out, load_any,
                    resolve_dataset, write_results)

from metamorph.utils import l2_normalize, set_seed, topk_indices


def exact_topk(X: np.ndarray, Q: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((Q.shape[0], k), dtype=np.int64)
    chunk = max(1, 2_000_000_000 // max(X.shape[0], 1))
    for s in range(0, Q.shape[0], chunk):
        scores = X @ Q[s:s + chunk].T
        for j in range(scores.shape[1]):
            out[s + j] = topk_indices(scores[:, j], k)
    return out


def old_partition_assignments(X_old: np.ndarray, nlist: int, seed: int) -> np.ndarray:
    """The partitioning the OLD index already has (it is being migrated)."""
    import faiss

    d = X_old.shape[1]
    km = faiss.Kmeans(d, nlist, niter=10, seed=seed, verbose=False)
    km.train(np.ascontiguousarray(X_old, dtype=np.float32))
    _, assign = km.index.search(np.ascontiguousarray(X_old, dtype=np.float32), 1)
    return assign.ravel().astype(np.int32)


def build_ivf(X_new: np.ndarray, nlist: int, seed: int, centroids: np.ndarray | None,
              refine_iters: int) -> tuple[object, dict]:
    """Build an IVF-Flat index; `centroids` None = cold k-means training."""
    import faiss

    d = X_new.shape[1]
    quant = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
    t0 = time.perf_counter()
    if centroids is None:
        index.cp.niter = 10
        index.cp.seed = seed
        index.train(X_new)
    else:
        C = np.ascontiguousarray(centroids, dtype=np.float32)
        if refine_iters > 0:
            km = faiss.Kmeans(d, nlist, niter=refine_iters, seed=seed, verbose=False)
            km.train(X_new, init_centroids=C)
            C = np.ascontiguousarray(
                np.asarray(km.centroids).reshape(nlist, d), dtype=np.float32)
        quant.reset()
        quant.add(C)
        index.is_trained = True
    train_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    index.add(X_new)
    add_s = time.perf_counter() - t1
    return index, {"train_s": train_s, "add_s": add_s, "build_s": train_s + add_s}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--n-docs", type=int, default=None)
    ap.add_argument("--nlist", type=int, default=None,
                    help="IVF lists (default ~ 4*sqrt(N))")
    ap.add_argument("--nprobes", nargs="*", type=int, default=[1, 4, 8, 16, 32])
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--refine-iters", nargs="*", type=int, default=[0, 1, 2, 5])
    args = ap.parse_args()

    import faiss

    data = load_any(resolve_dataset(args), args.pair, args.scale, n_docs=args.n_docs,
                    seed=args.seed)
    set_seed(args.seed)
    X_old = np.ascontiguousarray(l2_normalize(data.old))
    X_new = np.ascontiguousarray(l2_normalize(data.new))
    n, d = X_new.shape
    nlist = args.nlist or int(max(16, min(65536, 4 * np.sqrt(n))))
    Q = np.ascontiguousarray(l2_normalize(data.q_new[:args.queries]))
    print(f"N={n:,} d={d} nlist={nlist} queries={Q.shape[0]}", flush=True)

    t0 = time.perf_counter()
    truth = exact_topk(X_new, Q, args.k)
    print(f"exact ground truth in {time.perf_counter()-t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    old_assign = old_partition_assignments(X_old, nlist, args.seed)
    print(f"old-index partitioning recovered in {time.perf_counter()-t0:.1f}s "
          f"(free: it is the index being migrated)", flush=True)

    # warm centroids: new-space mean of each OLD partition
    t0 = time.perf_counter()
    C = np.zeros((nlist, d), dtype=np.float32)
    counts = np.bincount(old_assign, minlength=nlist).astype(np.float32)
    np.add.at(C, old_assign, X_new)
    nz = counts > 0
    C[nz] /= counts[nz, None]
    if (~nz).any():
        rng = np.random.default_rng(args.seed)
        C[~nz] = X_new[rng.choice(n, size=int((~nz).sum()))]
    warm_init_s = time.perf_counter() - t0
    print(f"warm centroid init: {warm_init_s:.2f}s", flush=True)

    variants = [("cold", None, 0)] + [("warm", C, it) for it in args.refine_iters]
    rows: list[dict] = []
    for name, cent, it in variants:
        index, timing = build_ivf(X_new, nlist, args.seed, cent, it)
        if cent is not None:
            timing["train_s"] += warm_init_s
            timing["build_s"] += warm_init_s
        for nprobe in args.nprobes:
            index.nprobe = nprobe
            t0 = time.perf_counter()
            _, ids = index.search(Q, args.k)
            search_s = time.perf_counter() - t0
            rec = float(np.mean([np.intersect1d(ids[i], truth[i]).size / args.k
                                 for i in range(Q.shape[0])]))
            rows.append({"variant": name, "refine_iters": it, "nlist": nlist,
                         "nprobe": nprobe, "recall@%d" % args.k: rec,
                         "search_ms_per_query": 1000 * search_s / Q.shape[0],
                         **timing, "n_docs": n})
            print(f"  [{name} refine={it}] nprobe={nprobe} recall={rec:.4f} "
                  f"build={timing['build_s']:.2f}s (train {timing['train_s']:.2f}s)",
                  flush=True)
        del index

    # ---- time-to-matched-recall -------------------------------------------
    rec_col = "recall@%d" % args.k
    summary: list[dict] = []
    import pandas as pd
    df = pd.DataFrame(rows)
    for nprobe in args.nprobes:
        sub = df[df.nprobe == nprobe]
        cold = sub[sub.variant == "cold"].iloc[0]
        warm = sub[sub.variant == "warm"]
        ok = warm[warm[rec_col] >= cold[rec_col] - 1e-9]
        best = ok.loc[ok.build_s.idxmin()] if len(ok) else None
        summary.append({
            "nprobe": nprobe, "cold_recall": cold[rec_col],
            "cold_build_s": cold.build_s, "cold_train_s": cold.train_s,
            "warm_matched": bool(best is not None),
            "warm_refine_iters": (int(best.refine_iters) if best is not None else np.nan),
            "warm_recall": (float(best[rec_col]) if best is not None else np.nan),
            "warm_build_s": (float(best.build_s) if best is not None else np.nan),
            "warm_train_s": (float(best.train_s) if best is not None else np.nan),
            "build_speedup": (float(cold.build_s / best.build_s) if best is not None else np.nan),
            "train_speedup": (float(cold.train_s / best.train_s) if best is not None else np.nan),
            "n_docs": n, "nlist": nlist,
        })
        print(f"[nprobe={nprobe}] cold {cold[rec_col]:.4f}@{cold.build_s:.1f}s -> "
              f"warm matched: {summary[-1]['warm_matched']} "
              f"build x{summary[-1]['build_speedup']:.2f} "
              f"train x{summary[-1]['train_speedup']:.2f}", flush=True)

    meta = base_meta(args, data, nlist=nlist, k=args.k, n_queries_probe=Q.shape[0])
    # the corpus size is part of the identity of this run: without it, a sweep
    # over --n-docs silently overwrites its own summaries
    suf = f"_n{n}"
    write_results(rows, args.out or default_out(args, "exp4b_faiss", suf), meta)
    write_results(summary, default_out(args, "exp4b_faiss", suf + "_summary"), meta)


if __name__ == "__main__":
    main()
