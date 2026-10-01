"""Exp 4: warm-start new-space index construction from old-space structure.

Mechanism 1 — warm NN-descent (graph index):
  cold: NN-descent from a random kNN graph over the NEW space.
  warm: NN-descent initialized from the OLD-space exact kNN graph, rescored
        with new-space similarities (the old graph is recycled from the old
        index, so its construction cost is NOT charged; the rescoring is).
  Measured per iteration: cumulative distance computations, graph recall
  (fraction of true new-space kNN edges present, on a row sample).
  Final graphs are served with beam search; query recall@k vs exact search.

Mechanism 2 — warm k-means (IVF/partitions):
  cold: k-means++ init (sklearn) then Lloyd; warm: init centroids = new-space
  means of the OLD clusters. Measured: Lloyd iterations, final inertia.

  python experiments/exp4_warmstart_index.py --scale smoke
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.cluster import KMeans, kmeans_plusplus

from common import (add_common_args, base_meta, default_out, load_prepared,
                    resolve_dataset, write_results)

from metamorph.index import BruteForceIndex
from metamorph.metrics import recall_at_k
from metamorph.utils import l2_normalize, set_seed
from metamorph.warmstart import (KnnGraph, beam_search, exact_knn, lloyd_kmeans,
                                 nn_descent, warm_kmeans_init)


def graph_recall(graph: KnnGraph, rows: np.ndarray, truth: np.ndarray) -> float:
    hits, total = 0, 0
    for r, i in enumerate(rows):
        hits += np.intersect1d(graph.neighbors[i], truth[r]).size
        total += truth.shape[1]
    return hits / max(total, 1)


def run_nn_descent(tag: str, X_new, init_graph: KnnGraph, counter: list[int],
                   rows, truth, n_iters: int, rng) -> tuple[list[dict], KnnGraph]:
    rows_out = [{"mechanism": "nn_descent", "variant": tag, "iter": 0,
                 "dist_comps": counter[0],
                 "graph_recall": graph_recall(init_graph, rows, truth)}]

    def cb(it, g):
        rows_out.append({"mechanism": "nn_descent", "variant": tag, "iter": it,
                         "dist_comps": counter[0],
                         "graph_recall": graph_recall(g, rows, truth)})

    g = nn_descent(X_new, init_graph, n_iters=n_iters, rng=rng,
                   counter=counter, callback=cb)
    return rows_out, g


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--graph-k", type=int, default=10)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--recall-rows", type=int, default=500,
                    help="row sample for graph-recall measurement")
    ap.add_argument("--beam-queries", type=int, default=100)
    ap.add_argument("--beam-ef", type=int, default=64)
    ap.add_argument("--kmeans-k", type=int, default=None,
                    help="clusters for mechanism 2 (default: engine partitions)")
    args = ap.parse_args()

    data = load_prepared(resolve_dataset(args), args.pair, args.scale)
    rng = set_seed(args.seed)
    X_old = l2_normalize(data.old)
    X_new = l2_normalize(data.new)
    n = X_new.shape[0]
    gk = args.graph_k

    # ---- ground truth for graph recall (row sample) ------------------------
    rows_sample = rng.choice(n, size=min(args.recall_rows, n), replace=False)
    t0 = time.time()
    truth = exact_knn(X_new, gk, rows=rows_sample)
    print(f"truth kNN on {rows_sample.size} rows: {time.time() - t0:.1f}s")

    out_rows: list[dict] = []

    # ---- mechanism 1: NN-descent ------------------------------------------
    # cold: random init (its init distance comps ARE charged)
    c_cold = [0]
    g0_cold = KnnGraph.random_init(X_new, gk, rng=np.random.default_rng(args.seed),
                                   counter=c_cold)
    t0 = time.time()
    r_cold, g_cold = run_nn_descent("cold", X_new, g0_cold, c_cold, rows_sample,
                                    truth, args.iters, np.random.default_rng(args.seed))
    cold_s = time.time() - t0
    # warm: old-space exact kNN graph (recycled structure — construction free),
    # rescored in the new space (rescoring IS charged)
    t0 = time.time()
    old_graph = exact_knn(X_old, gk)
    old_graph_s = time.time() - t0
    c_warm = [0]
    g0_warm = KnnGraph.from_old_graph(X_new, old_graph, counter=c_warm)
    t0 = time.time()
    r_warm, g_warm = run_nn_descent("warm", X_new, g0_warm, c_warm, rows_sample,
                                    truth, args.iters, np.random.default_rng(args.seed))
    warm_s = time.time() - t0
    out_rows += r_cold + r_warm

    # comps to reach fixed graph-recall targets (linear interpolation between
    # iterations); full curves are in the CSV. Each variant may plateau at a
    # slightly different final recall — targets above a plateau report nan.
    def comps_at(rows_v, target):
        xs = [r["graph_recall"] for r in rows_v]
        ys = [r["dist_comps"] for r in rows_v]
        for a in range(1, len(xs)):
            if xs[a] >= target:
                if xs[a] == xs[a - 1]:
                    return float(ys[a])
                t = (target - xs[a - 1]) / (xs[a] - xs[a - 1])
                return float(ys[a - 1] + t * (ys[a] - ys[a - 1]))
        return float("nan")

    final_cold = r_cold[-1]["graph_recall"]
    final_warm = r_warm[-1]["graph_recall"]
    targets = [0.7, 0.8, 0.85, 0.9, min(final_cold, final_warm)]
    speed_rows = {}
    for tgt in targets:
        c, w = comps_at(r_cold, tgt), comps_at(r_warm, tgt)
        speed_rows[round(tgt, 4)] = (c, w, c / w if w and np.isfinite(c) and np.isfinite(w) else float("nan"))
        print(f"target recall {tgt:.3f}: cold {c:.0f} comps, warm {w:.0f} "
              f"(speedup {speed_rows[round(tgt,4)][2]:.2f}x)")
    print(f"finals: cold={final_cold:.3f} warm={final_warm:.3f}")

    # ---- beam-search query recall on the finished graphs -------------------
    dest = BruteForceIndex(X_new)
    qidx = rng.choice(len(data.q_ids), size=min(args.beam_queries, len(data.q_ids)),
                      replace=False)
    for tag, g in (("cold", g_cold), ("warm", g_warm)):
        recs, t0 = [], time.time()
        for qi in qidx:
            q = l2_normalize(data.q_new[qi])
            served = beam_search(X_new, g, q, k=10, ef=args.beam_ef,
                                 rng=np.random.default_rng(args.seed))
            true_ids, _ = dest.search(q, 10)
            recs.append(recall_at_k(served, true_ids, 10))
        out_rows.append({"mechanism": "beam_serve", "variant": tag,
                         "beam_recall@10": float(np.mean(recs)),
                         "n_queries": int(qidx.size),
                         "seconds": round(time.time() - t0, 2)})
        print(f"beam[{tag}] recall@10={np.mean(recs):.3f}")

    # ---- mechanism 2: warm k-means ----------------------------------------
    kk = args.kmeans_k or max(8, int(np.sqrt(n)))
    old_assign = KMeans(n_clusters=kk, random_state=args.seed, n_init=3
                        ).fit_predict(X_old).astype(np.int32)
    C_warm = warm_kmeans_init(X_new, old_assign, kk)
    C_cold, _ = kmeans_plusplus(X_new, n_clusters=kk, random_state=args.seed)
    for tag, C0 in (("cold", C_cold), ("warm", C_warm)):
        t0 = time.time()
        res = lloyd_kmeans(X_new, C0, max_iter=200)
        out_rows.append({"mechanism": "kmeans", "variant": tag, "k": kk,
                         "n_iter": res.n_iter, "inertia": res.inertia,
                         "seconds": round(time.time() - t0, 2)})
        print(f"kmeans[{tag}] iters={res.n_iter} inertia={res.inertia:.1f}")

    srow = {
        "graph_k": gk, "iters": args.iters,
        "cold_final_recall": final_cold, "cold_dist_comps": r_cold[-1]["dist_comps"],
        "warm_final_recall": final_warm,
        "warm_dist_comps": r_warm[-1]["dist_comps"],
        "old_graph_seconds_uncharged": round(old_graph_s, 2),
        "cold_seconds": round(cold_s, 2), "warm_seconds": round(warm_s, 2),
    }
    for tgt, (c, w, s) in speed_rows.items():
        srow[f"cold_comps@{tgt}"] = c
        srow[f"warm_comps@{tgt}"] = w
        srow[f"speedup@{tgt}"] = s
    summary = [srow]
    meta = base_meta(args, data, graph_k=gk, kmeans_k=kk)
    write_results(out_rows, args.out or default_out(args, "exp4_warmstart"), meta)
    write_results(summary, default_out(args, "exp4_warmstart", "_summary"), meta)


if __name__ == "__main__":
    main()
