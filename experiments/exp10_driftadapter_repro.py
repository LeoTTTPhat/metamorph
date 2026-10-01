"""Exp 10: faithful reproduction of Drift-Adapter, and the factors separating
its 95-99% recovery from Metamorph's low destination agreement.

Drift-Adapter (Vejendla, EMNLP 2025) protocol, as stated (Sec. 3-4, App. A):
  direction   new-model QUERY -> old space; the old index is searched unchanged
  adapters    OP   closed-form orthogonal Procrustes on all Np pairs, no DSM
              LA   g(x) = U V^T x + t, rank r = 64, with diagonal scaling (DSM)
              MLP  g(x) = x + W2 GELU(W1 x + b1) + b2, 256 hidden units,
                   dropout 0.1 between hidden and output layer, with DSM
  training    MSE; AdamW lr 3e-4, weight decay 0.01, batch 256, <= 50 epochs,
              early stopping on validation MSE (patience 5); LA/MLP use an 80/20
              train/validation split of the pairs, OP uses all of them
  pairs       Np = 20,000 random database documents (2% of a 1M database);
              5 runs, each with a fresh random sample
  embeddings  L2-normalized before any adapter step or indexing
  index       FAISS HNSW over the OLD vectors, M=32, efConstruction=200,
              efSearch=50
  truth       exhaustive kNN with NEW-model embeddings for queries and database
  metrics     Recall@10 and MRR against that truth; ARR = metric of the adapter
              system / metric of the re-embedded system (new queries on a
              new-model index built with the same HNSW parameters)

Not stated in the Drift-Adapter paper; our choices (also written to the .meta.json):
  * d_new != d_old (their own main pairs are 384->768 and 512->768): OP is the
    rectangular Procrustes solution R = U V^T of SVD(A B^T); the MLP's residual
    path uses that fixed R as a projection shortcut; LA needs nothing.
  * MRR@10 = reciprocal rank of the true nearest neighbour in the returned top 10.
  * DSM initialized at 1; the best-validation checkpoint is restored.
  * the database is the full training split where Drift-Adapter's 1M exceeds it
    (see prepare_drift.py).

Beyond the reproduction, each setting also reports
  * old_direct   the un-upgraded system (old queries, old index) vs the same truth
  * misaligned   new queries on the old index without an adapter (same-dim pairs)
  * corpus_*     Metamorph's pure-adapter direction: the same adapter family fitted
                 old -> new on the same pairs, applied to every stored vector,
                 new queries searched exactly (the quantity exp3 reports)
  * qrels NDCG@10 / Recall@10 on BEIR corpora, and task recovery = ratio of a
    system's NDCG@10 to the re-embedded system's.

  python experiments/exp10_driftadapter_repro.py --settings agnews:minilm:mpnet
  python experiments/exp10_driftadapter_repro.py --settings fiqa:minilm:e5s \
      --np 1153 20000
"""
from __future__ import annotations

import argparse

import json
import time
from pathlib import Path

import numpy as np

from common import PREPARED_DIR, RESULTS_DIR, load_bundle, load_prepared

from metamorph.metrics import ndcg_at_k
from metamorph.utils import l2_normalize

DRIFT_DIR = PREPARED_DIR / "drift"
K = 10
HNSW_M, HNSW_EFC, HNSW_EFS = 32, 200, 50
DEFAULT_NP = 20_000
FIQA_BUNDLE = {"e5s": "minilm-e5", "gtes": "minilm-gte", "e5large": "minilm-e5large"}


# ------------------------------------------------------------------ settings
class Setting:
    def __init__(self, name, X_old, X_new, Q_old, Q_new, qrels=None, meta=None,
                 labels=None):
        self.name = name
        self.X_old = np.ascontiguousarray(l2_normalize(np.asarray(X_old, dtype=np.float32)))
        self.X_new = np.ascontiguousarray(l2_normalize(np.asarray(X_new, dtype=np.float32)))
        self.Q_old = np.ascontiguousarray(l2_normalize(np.asarray(Q_old, dtype=np.float32)))
        self.Q_new = np.ascontiguousarray(l2_normalize(np.asarray(Q_new, dtype=np.float32)))
        self.qrels = qrels  # list aligned with queries: {doc_idx: gain} or None
        self.meta = meta or {}
        self.labels = labels  # (db_labels, q_labels) for the classification corpora

    @property
    def n(self):
        return self.X_old.shape[0]


def _drift_arrays(ds: str, tag: str):
    root = DRIFT_DIR / ds
    return np.load(root / f"db_{tag}.npy"), np.load(root / f"q_{tag}.npy")


def load_setting(spec: str) -> Setting:
    """spec = dataset:old_tag:new_tag, e.g. agnews:minilm:mpnet or fiqa:minilm:e5s."""
    ds, old_tag, new_tag = spec.split(":")
    assert old_tag == "minilm", "every pair in this study starts from all-MiniLM-L6-v2"
    if new_tag in ("identity", "rotation"):
        # pipeline sanity checks: an upgrade a correct pipeline must fully recover
        base = load_setting(f"{ds}:minilm:e5s")
        Rot = np.eye(base.X_old.shape[1], dtype=np.float32)
        if new_tag == "rotation":
            Rot = np.linalg.qr(np.random.default_rng(0).standard_normal(
                (base.X_old.shape[1],) * 2))[0].astype(np.float32)
        return Setting(spec, base.X_old, base.X_old @ Rot, base.Q_old, base.Q_old @ Rot,
                       base.qrels, dict(base.meta, sanity=new_tag), base.labels)
    if ds in ("agnews", "dbpedia14", "emotion"):
        X_old, Q_old = _drift_arrays(ds, old_tag)
        X_new, Q_new = _drift_arrays(ds, new_tag)
        mp = DRIFT_DIR / ds / "meta.json"  # written once every encoder has finished
        meta = json.loads(mp.read_text()) if mp.exists() else {}
        meta.setdefault("task", "symmetric document-to-document search")
        labels = None
        if (DRIFT_DIR / ds / "db_labels.npy").exists():
            labels = (np.load(DRIFT_DIR / ds / "db_labels.npy"),
                      np.load(DRIFT_DIR / ds / "q_labels.npy"))
        return Setting(spec, X_old, X_new, Q_old, Q_new, None, meta, labels)
    if ds == "fiqa":
        bundle = FIQA_BUNDLE.get(new_tag, "minilm-e5")
        data = load_prepared("fiqa", bundle, "full")
        if new_tag == "mpnet":
            X_new, Q_new = _drift_arrays("fiqa", "mpnet")
        else:
            X_new, Q_new = data.new, data.q_new
        qrels = [data.qrels.get(q) for q in data.q_ids]
        return Setting(spec, data.old, X_new, data.q_old, Q_new, qrels,
                       {"n_db": data.n_docs, "task": "asymmetric passage retrieval"})
    if ds == "msmarco1m":
        data = load_bundle("msmarco", "minilm-e5", "xxl", n_docs=1_000_000, seed=42)
        if new_tag == "mpnet":
            X_new, Q_new = _drift_arrays("msmarco1m", "mpnet")
            assert X_new.shape[0] == data.old.shape[0] and Q_new.shape[0] == data.q_old.shape[0]
        else:
            assert new_tag == "e5s"
            X_new, Q_new = data.new, data.q_new
        qrels = [data.qrels.get(q) for q in data.q_ids]
        return Setting(spec, data.old, X_new, data.q_old, Q_new, qrels,
                       {"n_db": int(data.old.shape[0]), "task": "asymmetric passage retrieval"})
    raise ValueError(spec)


# ------------------------------------------------------------------- search
def faiss_threads():
    import faiss
    faiss.omp_set_num_threads(32)


def exact_topk(X: np.ndarray, Q: np.ndarray, k: int = K) -> np.ndarray:
    import faiss
    idx = faiss.IndexFlatIP(X.shape[1])
    idx.add(X)
    _, I = idx.search(np.ascontiguousarray(Q, dtype=np.float32), k)
    return I


def hnsw_index(X: np.ndarray, cache: Path | None):
    import faiss
    if cache is not None and cache.exists():
        idx = faiss.read_index(str(cache))
    else:
        idx = faiss.IndexHNSWFlat(X.shape[1], HNSW_M)  # L2 on unit vectors == cosine order
        idx.hnsw.efConstruction = HNSW_EFC
        idx.add(X)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(idx, str(cache))
    idx.hnsw.efSearch = HNSW_EFS
    return idx


def hnsw_topk(idx, Q: np.ndarray, k: int = K) -> np.ndarray:
    _, I = idx.search(np.ascontiguousarray(l2_normalize(Q), dtype=np.float32), k)
    return I


# ------------------------------------------------------------------ metrics
def recall_vs_truth(I: np.ndarray, T: np.ndarray) -> float:
    return float(np.mean([len(set(a[a >= 0]) & set(b)) / K for a, b in zip(I, T)]))


def recall_in_deeper(I: np.ndarray, T100: np.ndarray) -> float:
    """Lenient variant: share of the returned top-10 inside the new model's top-100."""
    return float(np.mean([len(set(a[a >= 0]) & set(b)) / K for a, b in zip(I, T100)]))


def label_precision(I: np.ndarray, labels) -> dict:
    """Task-level metric on the classification corpora: share of the returned
    top-10 carrying the query's class label."""
    if labels is None:
        return {}
    db, q = labels
    return {"label_prec10": float(np.mean((db[I] == q[:, None]).mean(1)))}


def mrr_nn1(I: np.ndarray, T: np.ndarray) -> float:
    rr = []
    for a, b in zip(I, T):
        hit = np.where(a == b[0])[0]
        rr.append(1.0 / (hit[0] + 1) if hit.size else 0.0)
    return float(np.mean(rr))


def qrels_metrics(I: np.ndarray, qrels) -> dict:
    if qrels is None:
        return {}
    nd, rc = [], []
    for ids, rel in zip(I, qrels):
        if not rel:
            continue
        ids = [int(x) for x in ids if x >= 0]
        nd.append(ndcg_at_k(ids, rel, K))
        rc.append(len(set(ids) & set(rel)) / len(rel))
    return {"ndcg10": float(np.mean(nd)), "qrels_recall10": float(np.mean(rc))}


# ----------------------------------------------------------------- adapters
# Training runs in spawned torch-only workers (drift_train.py): PyTorch and
# FAISS link separate libomp copies on macOS and abort if both load in one
# process. Weights come back as NumPy arrays; inference here is NumPy.
from drift_train import apply_weights  # noqa: E402  (numpy/scipy only)

_POOL = None


def pool(workers: int = 8):
    global _POOL
    if _POOL is None:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        _POOL = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
    return _POOL


RECIPE = "paper"


def fit_many(jobs):
    """jobs: list of (key, kind, X_in, Y_out, seed) -> {key: weights}.

    Every (adapter, configuration) pair is its own pool job, so a multi-
    configuration recipe runs in parallel; per adapter the configuration with
    the lowest validation MSE is kept (test queries are never consulted)."""
    import drift_train
    futs = []
    for key, kind, X_in, Y_out, seed in jobs:
        configs = (drift_train.RECIPES[RECIPE] if kind in ("LA", "MLP")
                   else drift_train.RECIPES["paper"][:1])
        for cfg in configs:
            futs.append((key, pool().submit(drift_train.train_adapter, kind, X_in, Y_out,
                                            seed, 3, **cfg)))
    out = {}
    for key, f in futs:
        w = f.result()
        assert w["info"]["check_max_abs"] < 1e-3, (key, w["info"])
        w["info"]["recipe"] = RECIPE
        if key not in out or w["info"]["val_mse"] < out[key]["info"]["val_mse"]:
            out[key] = w
    return out


# --------------------------------------------------------------------- main
def run_setting(st: Setting, np_list, runs, adapters, directions, rows):
    faiss_threads()
    ds, old_tag, new_tag = st.name.split(":")
    cache = DRIFT_DIR / ds / "hnsw"
    t0 = time.time()
    T100 = exact_topk(st.X_new, st.Q_new, 100)
    T = np.ascontiguousarray(T100[:, :K])
    h_old = hnsw_index(st.X_old, cache / f"{old_tag}_M{HNSW_M}_efc{HNSW_EFC}.faiss")
    h_new = hnsw_index(st.X_new, cache / f"{new_tag}_M{HNSW_M}_efc{HNSW_EFC}.faiss")
    print(f"[{st.name}] N={st.n} Q={st.Q_new.shape[0]} truth+indexes {time.time()-t0:.0f}s",
          flush=True)
    base = {"setting": st.name, "dataset": ds, "old": old_tag, "new": new_tag,
            "d_old": st.X_old.shape[1], "d_new": st.X_new.shape[1], "n_db": st.n,
            "n_q": st.Q_new.shape[0], "task": st.meta.get("task", "")}

    def record(method, direction, index, I, extra):
        r = dict(base, method=method, direction=direction, index=index,
                 recall10_vs_new=recall_vs_truth(I, T), mrr10_vs_new=mrr_nn1(I, T),
                 recall10_in_new100=recall_in_deeper(I, T100))
        r.update(qrels_metrics(I, st.qrels))
        r.update(label_precision(I, st.labels))
        r.update(extra)
        rows.append(r)
        return r

    ref = record("new_direct", "reembedded", "hnsw", hnsw_topk(h_new, st.Q_new),
                 {"np": 0, "run": -1})
    record("new_exact", "reembedded", "exact", T, {"np": 0, "run": -1})
    record("old_direct", "none", "hnsw", hnsw_topk(h_old, st.Q_old), {"np": 0, "run": -1})
    if st.X_old.shape[1] == st.X_new.shape[1]:
        record("misaligned", "query->old", "hnsw", hnsw_topk(h_old, st.Q_new),
               {"np": 0, "run": -1})
    for Np in np_list:
        Np_eff = min(Np, st.n)
        for run in range(runs):
            rng = np.random.default_rng(1000 * run + 7)
            pick = rng.choice(st.n, size=Np_eff, replace=False)
            A_old, B_new = st.X_old[pick], st.X_new[pick]
            jobs = []
            for kind in adapters:
                if "query->old" in directions:
                    jobs.append((("q", kind), kind, B_new, A_old, run))
                if "corpus->new" in directions and kind != "OP+DSM":
                    jobs.append((("c", kind), kind, A_old, B_new, run))
            fitted = fit_many(jobs)
            for kind in adapters:
                if ("q", kind) in fitted:
                    w = fitted[("q", kind)]
                    Qm = apply_weights(w, st.Q_new)
                    ex = {"np": Np_eff, "np_requested": Np, "run": run,
                          **{k: v for k, v in w["info"].items() if k != "kind"}}
                    rh = record(f"drift_{kind}", "query->old", "hnsw", hnsw_topk(h_old, Qm), ex)
                    record(f"drift_{kind}", "query->old", "exact",
                           exact_topk(st.X_old, l2_normalize(Qm)), ex)
                    print(f"  Np={Np_eff:6d} run={run} drift_{kind:6s} R@10={rh['recall10_vs_new']:.3f}"
                          f" ARR={rh['recall10_vs_new']/ref['recall10_vs_new']:.3f}"
                          f" ep={w['info']['epochs']}", flush=True)
                if ("c", kind) in fitted:
                    w = fitted[("c", kind)]
                    Xm = l2_normalize(apply_weights(w, st.X_old))
                    ex = {"np": Np_eff, "np_requested": Np, "run": run,
                          **{k: v for k, v in w["info"].items() if k != "kind"}}
                    rc = record(f"corpus_{kind}", "corpus->new", "exact",
                                exact_topk(Xm, st.Q_new), ex)
                    print(f"  Np={Np_eff:6d} run={run} corpus_{kind:5s} R@10={rc['recall10_vs_new']:.3f}",
                          flush=True)
    # ratios against the re-embedded system (Drift-Adapter's ARR)
    for r in rows:
        if r["setting"] != st.name:
            continue
        r["arr_recall"] = r["recall10_vs_new"] / ref["recall10_vs_new"]
        r["arr_mrr"] = r["mrr10_vs_new"] / ref["mrr10_vs_new"]
        if "ndcg10" in r and "ndcg10" in ref:
            r["task_recovery_ndcg"] = r["ndcg10"] / ref["ndcg10"]
        if "label_prec10" in r and "label_prec10" in ref:
            r["task_recovery_label"] = r["label_prec10"] / ref["label_prec10"]
        r["arr_lenient"] = r["recall10_in_new100"] / ref["recall10_in_new100"]


def summarize(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    keys = ["setting", "method", "direction", "index", "np"]
    num = [c for c in ("recall10_vs_new", "mrr10_vs_new", "arr_recall", "arr_mrr",
                       "recall10_in_new100", "arr_lenient", "label_prec10",
                       "task_recovery_label", "ndcg10", "qrels_recall10",
                       "task_recovery_ndcg", "fit_s", "epochs", "best_epoch")
           if c in df.columns]
    g = df.groupby(keys, dropna=False)[num]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd")).join(
        g.size().rename("runs")).reset_index()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", nargs="+", required=True)
    ap.add_argument("--np", nargs="+", type=int, default=[DEFAULT_NP])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--adapters", nargs="+", default=["OP", "LA", "MLP"])
    ap.add_argument("--directions", nargs="+", default=["query->old", "corpus->new"])
    ap.add_argument("--tag", default="main")
    ap.add_argument("--recipe", default="paper", choices=["paper", "converged", "grid"])
    args = ap.parse_args()
    global RECIPE
    RECIPE = args.recipe

    import faiss
    rows = []
    t0 = time.time()
    for spec in args.settings:
        run_setting(load_setting(spec), args.np, args.runs, args.adapters,
                    args.directions, rows)
    import pandas as pd
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"exp10_driftadapter_{args.tag}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    summ = summarize(rows)
    summ.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    meta = {
        "experiment": "exp10 Drift-Adapter faithful reproduction + factor analysis",
        "argv": vars(args), "runtime_s": round(time.time() - t0, 1),
        "protocol_stated": {
            "direction": "new query -> old space, old index unchanged",
            "adapters": {"OP": "closed form, all pairs, no DSM",
                         "LA": "rank 64 + bias + DSM",
                         "MLP": "1 hidden layer 256, GELU, dropout 0.1, residual, DSM"},
            "training": "MSE; AdamW lr 3e-4 wd 0.01 batch 256 <=50 epochs; early stop "
                        "patience 5 on val MSE; 80/20 split",
            "pairs": "Np=20,000 random database docs; 5 runs, fresh sample each",
            "index": f"FAISS HNSW M={HNSW_M} efC={HNSW_EFC} efS={HNSW_EFS}",
            "truth": "exhaustive new-model kNN (k=10)",
            "metrics": "Recall@10, MRR, ARR = system / re-embedded system"},
        "our_choices_not_stated": {
            "dim_mismatch": "rectangular Procrustes R=UV^T; MLP skip = fixed R",
            "mrr": "MRR@10 of the true nearest neighbour",
            "dsm_init": "ones", "checkpoint": "best validation",
            "database": "full training split where 1M exceeds it"},
        "faiss": faiss.__version__,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    out.with_suffix(".csv.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out} ({len(rows)} rows) in {time.time()-t0:.0f}s")
    cols = ["setting", "method", "index", "np", "recall10_vs_new_mean", "arr_recall_mean",
            "arr_mrr_mean", "arr_lenient_mean"]
    cols += [c for c in ("task_recovery_label_mean", "ndcg10_mean", "task_recovery_ndcg_mean")
             if c in summ.columns]
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(summ[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
