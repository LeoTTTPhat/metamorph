"""Summarize the Drift-Adapter reproduction (exp10) as Markdown tables.

Reads results/exp10_driftadapter_*_summary.csv (mean and s.d. over runs, written
by exp10_driftadapter_repro.py), prints the tables, and writes them to
results/exp10_summary.md. All numbers are ARR unless stated otherwise: Recall@10
against the new model's exhaustive nearest neighbours, divided by the Recall@10 of
the re-embedded system (new queries on a new-model index with the same HNSW
parameters), which is Drift-Adapter's own metric.

    python experiments/summarize_exp10.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

BENCH = [("agnews", "AG-News"), ("dbpedia14", "DBpedia-14"), ("emotion", "Emotion"),
         ("fiqa", "fiqa"), ("msmarco1m", "MS MARCO (1M rung)")]
DRIFT_SETS = ["agnews", "dbpedia14", "emotion"]
# R@10 ARR published by Drift-Adapter (Vejendla, EMNLP 2025), Table 1: OP, LA, MLP
PUBLISHED = {"agnews": (0.974, 0.983, 0.992), "dbpedia14": (0.968, 0.979, 0.990),
             "emotion": (0.953, 0.967, 0.984)}


def load(prefixes: tuple[str, ...]) -> pd.DataFrame:
    frames = []
    for p in sorted(RES.glob("exp10_driftadapter_*_summary.csv")):
        tag = p.name[len("exp10_driftadapter_"):-len("_summary.csv")]
        if tag.startswith(prefixes):
            d = pd.read_csv(p)
            d["tag"] = tag
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def val(df, setting, method, col="arr_recall_mean", index="hnsw", np_=None):
    if df.empty or col not in df.columns:
        return None
    d = df[(df.setting == setting) & (df.method == method) & (df["index"] == index)]
    if np_ is not None:
        d = d[d.np == np_]
    d = d[d[col].notna()]
    return float(d[col].iloc[0]) if len(d) else None


def f(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def main() -> None:
    base = load(("paper",))            # Drift-Adapter's published training recipe
    conv = load(("converged",))
    grid = load(("grid",))
    sweep = load(("npsweep",))
    seed2 = load(("seed2pct",))
    out = []

    out += ["## Drift-Adapter protocol, MiniLM-L6 → MPNet (all-mpnet-base-v2)", "",
            "Query mapped into the old space; 20,000 training pairs (Emotion: its 15,956 "
            "items); FAISS HNSW M=32, efConstruction=200, efSearch=50; 5 runs. "
            "*Old* is the un-upgraded system (old queries, old index, no adapter). "
            "*Conv.* and *Grid* are the best LA/MLP trained to convergence or tuned over "
            "Drift-Adapter's own hyperparameter ranges. *Lenient* counts the best adapter's "
            "top-10 against the new model's top-100. *Task* is class-label precision@10 "
            "(classification corpora) or NDCG@10 (fiqa, MS MARCO) of the best adapter and "
            "of the old model, relative to the re-embedded system.", "",
            "| Benchmark | Old | OP | OP+DSM | LA | MLP | Conv. | Grid | Lenient | Task (adapter / old) | Published OP / LA / MLP |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for ds, name in BENCH:
        st = f"{ds}:minilm:mpnet"
        npv = None if ds == "emotion" else 20000
        a = {k: val(base, st, f"drift_{k}", np_=npv) for k in ("OP", "OP+DSM", "LA", "MLP")}
        if a["OP"] is None:
            continue
        best = max(("OP", "LA", "MLP"), key=lambda k: a[k] or -1)
        tcol = "task_recovery_label_mean" if ds in DRIFT_SETS else "task_recovery_ndcg_mean"

        def best_of(df):
            v = [val(df, st, f"drift_{k}") for k in ("LA", "MLP")]
            v = [x for x in v if x is not None]
            return max(v) if v else None
        pub = " / ".join(f"{x:.3f}" for x in PUBLISHED[ds]) if ds in PUBLISHED else "–"
        out.append(f"| {name} | {f(val(base, st, 'old_direct'))} | {f(a['OP'])} | {f(a['OP+DSM'])} | "
                   f"{f(a['LA'])} | {f(a['MLP'])} | {f(best_of(conv))} | {f(best_of(grid))} | "
                   f"{f(val(base, st, f'drift_{best}', 'arr_lenient_mean'))} | "
                   f"{f(val(base, st, f'drift_{best}', tcol), 2)} / {f(val(base, st, 'old_direct', tcol), 2)} | {pub} |")

    out += ["", "## Checks", "", "| Check | Result (ARR) |", "|---|---|"]
    for k in ("identity", "rotation"):
        st = f"emotion:minilm:{k}"
        out.append(f"| Sanity, {k} upgrade (Emotion) | OP {f(val(base, st, 'drift_OP'))}, "
                   f"LA {f(val(base, st, 'drift_LA'))}, MLP {f(val(base, st, 'drift_MLP'))} |")
    if not sweep.empty:
        for k in ("OP", "MLP"):
            pts = [f"{int(n) // 1000}k {f(val(sweep, 'agnews:minilm:mpnet', f'drift_{k}', np_=int(n)))}"
                   for n in sorted(sweep.np.unique()) if n > 0]
            out.append(f"| Training pairs, AG-News, {k} | " + ", ".join(pts) + " |")
    for ds in DRIFT_SETS:
        st = f"{ds}:minilm:mpnet"
        c = [val(base, st, f"corpus_{k}", index="exact") for k in ("OP", "LA", "MLP")]
        c = [x for x in c if x is not None]
        if c:
            out.append(f"| Corpus direction (old corpus → new space), {ds} | best {f(max(c))} |")
    for p, lab in (("e5s", "e5-small-v2"), ("gtes", "gte-small")):
        for ds in DRIFT_SETS:
            st = f"{ds}:minilm:{p}"
            v = [val(base, st, f"drift_{k}") for k in ("OP", "LA", "MLP")]
            v = [x for x in v if x is not None]
            if v:
                out.append(f"| Destination {lab}, {ds} | best adapter {f(max(v))}, "
                           f"old {f(val(base, st, 'old_direct'))} |")
    for st in ("fiqa:minilm:e5s", "msmarco1m:minilm:e5s"):
        o = val(base, st, "old_direct")
        if o is not None:
            out.append(f"| {st} | old {f(o)}; OP {f(val(base, st, 'drift_OP', np_=20000))}, "
                       f"LA {f(val(base, st, 'drift_LA', np_=20000))}, MLP {f(val(base, st, 'drift_MLP', np_=20000))} |")
    for st in ("fiqa:minilm:mpnet", "fiqa:minilm:e5s"):
        if val(seed2, st, "drift_OP", np_=1153) is not None:
            out.append(f"| {st}, 1,153 pairs (2% of the corpus) | OP {f(val(seed2, st, 'drift_OP', np_=1153))}, "
                       f"LA {f(val(seed2, st, 'drift_LA', np_=1153))}, MLP {f(val(seed2, st, 'drift_MLP', np_=1153))} |")

    sds = base[(base["index"] == "hnsw") & base.method.str.startswith("drift")]["arr_recall_sd"]
    out += ["", f"Largest run-to-run standard deviation of any adapter ARR above: {sds.max():.4f}."]
    text = "\n".join(out) + "\n"
    (RES / "exp10_summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
