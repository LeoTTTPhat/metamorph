"""Exp 8: does the margin bound (Prop. 2) hold, and is it usable?

Proposition 2 says: if the served score margin at rank k exceeds 2*eps_q, where
eps_q is the maximum fusion error over unmigrated pool candidates, the served
top-k equals the true new-space top-k of the pool.  Two things were missing.

(a) It was never checked.  Here we check it directly: with oracle access we can
    compute eps_q exactly and verify that the implication never fails.

(b) eps_q is not observable at serving time -- computing it needs exactly the
    new-model embeddings the migration is trying to avoid buying.  The router
    therefore also emits eps_hat: a high quantile of the fusion residuals on
    this query's MIGRATED candidates, whose exact new-space scores are already
    paid for.  This experiment measures how often the operational precondition
    (margin > 2*eps_hat) fires and how reliable its conclusion is -- i.e.
    whether the proposition can be used as a per-query certificate rather than
    decoration.

Also measured here: the empirical monotonicity premise behind Prop. 3(b) --
the number of budget steps at which serving agreement with the destination
decreases, and the worst such decrease.

  python experiments/exp8_margin_bound.py --dataset scifact --pair minilm-e5 --scale small
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import (add_common_args, base_meta, default_engine_config, default_out,
                    load_any, resolve_dataset, write_results)

from metamorph.baselines import make_engine
from metamorph.utils import topk_indices
from metamorph.workload import ZipfStream, split_queries


def probe(engine, qids: list[str], k: int) -> list[dict]:
    """Per-query certificate audit at the CURRENT migration state."""
    X = engine.store.oracle_full()
    out = []
    for qid in qids:
        i = engine._qidx[qid]
        q_new = engine.data.q_new[i]
        q_old = (engine.data.q_old[i]
                 if engine.cfg.serve.query_mode == "dual" else None)
        res = engine.router.serve(q_old, q_new)
        cand, sc = res.candidates, res.cand_scores
        if cand.size == 0:
            continue
        exact = X[cand] @ q_new
        unmig = ~engine.state.migrated[cand]
        eps_true = float(np.max(np.abs(sc[unmig] - exact[unmig]))) if unmig.any() else 0.0
        true_top = cand[topk_indices(exact, min(k, cand.size))]
        conclusion = bool(np.array_equal(np.sort(res.ids), np.sort(true_top)))
        agree = float(np.intersect1d(res.ids, true_top).size / max(true_top.size, 1))
        # the graded certificate recomputed with the ORACLE eps, to separate
        # "the bound is loose" from "the eps surrogate is wrong"
        lb_oracle = engine.router._certified_recall_lb(
            cand, sc, topk_indices(sc, min(k, cand.size)), eps_true, res.n_from_old)
        out.append({
            "qid": qid, "margin_k": res.margin_k, "eps_true": eps_true,
            "eps_hat": res.eps_hat,
            "cond_true": bool(res.margin_k > 2.0 * eps_true),
            "cond_hat": bool(res.certified_topk),
            "eps_hat_covers": bool(np.isfinite(res.eps_hat) and res.eps_hat >= eps_true),
            "conclusion": conclusion, "pool_agreement": agree,
            "cert_lb_hat": res.cert_recall_lb, "cert_lb_oracle": lb_oracle,
            "cert_lb_hat_valid": bool(np.isfinite(res.cert_recall_lb)
                                      and res.cert_recall_lb <= agree + 1e-9),
            "cert_lb_oracle_valid": bool(np.isfinite(lb_oracle) and lb_oracle <= agree + 1e-9),
            "n_unmigrated_cands": int(unmig.sum()),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--probe-every", type=int, default=6, help="steps between probes")
    ap.add_argument("--probe-queries", type=int, default=150)
    ap.add_argument("--verify", type=int, default=0)
    ap.add_argument("--n-docs", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=10_000)
    args = ap.parse_args()

    data = load_any(resolve_dataset(args), args.pair, args.scale, n_docs=args.n_docs,
                    seed=args.seed)
    cfg = default_engine_config(data.n_docs, args.scale, args.seed)
    cfg.serve.verify_per_query = args.verify
    workload_qids, eval_qids = split_queries(data.q_ids, seed=args.seed)
    k = cfg.serve.k
    probe_qids = eval_qids[:args.probe_queries]

    engine = make_engine(data, cfg, "metamorph")
    stream = ZipfStream(workload_qids, alpha=1.1, seed=args.seed)

    rows: list[dict] = []
    t0 = time.time()
    step = 0
    # step the engine manually so we can audit at chosen budget points
    while step < args.max_steps and not engine.state.migrated.all():
        engine.run(stream, eval_qids[:20], max_steps=args.probe_every,
                   eval_max_queries=20)
        step += args.probe_every
        b = engine.meter.total / engine.store.n
        recs = probe(engine, probe_qids, k)
        if not recs:
            break
        arr = {key: np.array([r[key] for r in recs]) for key in recs[0] if key != "qid"}
        ct, ch = arr["cond_true"].astype(bool), arr["cond_hat"].astype(bool)
        conc = arr["conclusion"].astype(bool)
        rows.append({
            "budget_frac": b, "step": step, "n_queries": len(recs),
            "frac_cond_true": float(ct.mean()),
            "frac_cond_hat": float(ch.mean()),
            "prop2_holds_given_cond_true": float(conc[ct].mean()) if ct.any() else float("nan"),
            "prop2_holds_given_cond_hat": float(conc[ch].mean()) if ch.any() else float("nan"),
            "frac_conclusion_overall": float(conc.mean()),
            "eps_hat_coverage": float(arr["eps_hat_covers"].astype(float).mean()),
            "mean_eps_true": float(np.mean(arr["eps_true"])),
            "mean_eps_hat": float(np.nanmean(arr["eps_hat"])),
            "mean_margin_k": float(np.mean(arr["margin_k"])),
            "mean_pool_agreement": float(np.mean(arr["pool_agreement"])),
            "mean_pool_agreement_certified": (
                float(np.mean(arr["pool_agreement"][ch])) if ch.any() else float("nan")),
            "mean_pool_agreement_uncertified": (
                float(np.mean(arr["pool_agreement"][~ch])) if (~ch).any() else float("nan")),
            # the graded certificate: how much recall can be PROVEN per query
            # with only measurable quantities, and is the proof ever wrong?
            "mean_cert_lb_hat": float(np.nanmean(arr["cert_lb_hat"])),
            "mean_cert_lb_oracle": float(np.nanmean(arr["cert_lb_oracle"])),
            "cert_lb_hat_validity": float(np.mean(arr["cert_lb_hat_valid"].astype(float))),
            "cert_lb_oracle_validity": float(np.mean(arr["cert_lb_oracle_valid"].astype(float))),
            "verify": args.verify,
        })
        print(f"[b={b:.3f}] cond_true={rows[-1]['frac_cond_true']:.3f} "
              f"cert_lb_hat={rows[-1]['mean_cert_lb_hat']:.3f} "
              f"cert_lb_oracle={rows[-1]['mean_cert_lb_oracle']:.3f} "
              f"true_agr={rows[-1]['mean_pool_agreement']:.3f} "
              f"valid={rows[-1]['cert_lb_hat_validity']:.3f}", flush=True)
        if engine.meter.total >= engine.budget_cap:
            break

    # ---- Prop. 3(b): is serving agreement really monotone in budget? -------
    rec_col = f"recall@{k}_vs_dest"
    tl = [r for r in engine.timeline if r.get(rec_col) == r.get(rec_col)]
    vals = np.array([r[rec_col] for r in tl], dtype=float)
    diffs = np.diff(vals)
    mono = {
        "n_steps": int(diffs.size),
        "n_decreasing": int((diffs < 0).sum()),
        "frac_decreasing": float((diffs < 0).mean()) if diffs.size else float("nan"),
        "worst_decrease": float(diffs.min()) if diffs.size else float("nan"),
        "mean_decrease_when_negative": float(diffs[diffs < 0].mean()) if (diffs < 0).any() else 0.0,
        "runtime_s": round(time.time() - t0, 2),
    }
    print("monotonicity:", mono)

    meta = base_meta(args, data, probe_queries=len(probe_qids), verify=args.verify)
    write_results(rows, args.out or default_out(args, "exp8_margin"), meta)
    write_results([mono], default_out(args, "exp8_monotonicity"), meta)


if __name__ == "__main__":
    main()
