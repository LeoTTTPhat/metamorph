"""Generate the figures (PDF, one chart per file) from the experiment CSVs in
results/. Every value shown is also available in the CSVs themselves; the
series of the fiqa timeline, ladder and latency figures are also written to
results/figures/data/*.dat, for re-plotting with other tools.

Usage:
  python experiments/make_figures.py            # regenerates everything found

Design rules: single-column width (3.3in), Okabe-Ito colorblind-safe palette,
fixed method->color mapping across ALL figures, secondary encoding via
linestyle/marker (grayscale-safe), one axis per chart, recessive grid.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(exist_ok=True)

# ---------------------------------------------------------------- style
plt.rcParams.update({
    "figure.figsize": (3.3, 1.76),
    "figure.dpi": 150,
    "font.size": 7.5,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.4,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})

# Okabe-Ito, fixed identity mapping (validated CVD-safe; secondary encoding
# via linestyle+marker keeps grayscale/print safe).
C = {
    "metamorph": "#0072B2",      # blue
    "random_order": "#E69F00",   # orange
    "static_adapter": "#CC79A7", # purple-pink
    "full_swap": "#009E73",      # green
    "popularity_prior": "#D55E00",   # vermillion
    "oracle_traffic": "#56B4E9",     # sky blue
    "old": "#888888",
    "new": "#333333",
}
LS = {"metamorph": "-", "random_order": "--", "static_adapter": "-.",
      "full_swap": ":", "popularity_prior": (0, (3, 1, 1, 1)),
      "oracle_traffic": (0, (1, 1))}
MK = {"metamorph": "o", "random_order": "s", "static_adapter": "^",
      "full_swap": "D", "popularity_prior": "v", "oracle_traffic": "*"}
LABEL = {"metamorph": "Metamorph", "random_order": "random order",
         "static_adapter": "static adapter", "full_swap": "full re-embed+swap",
         "popularity_prior": "offline popularity prior",
         "oracle_traffic": "oracle traffic (upper bd.)"}
# verification-enabled variants inherit the base method's identity
for _m in list(C):
    C[f"{_m}_v2"] = C[_m]
    LS[f"{_m}_v2"] = LS.get(_m, "-")
    MK[f"{_m}_v2"] = MK.get(_m, "o")
    LABEL[f"{_m}_v2"] = f"{LABEL.get(_m, _m)} + verify"

GRID = np.linspace(0.0, 1.0, 101)


def seed_band(sub: pd.DataFrame, x: str, y: str, grid=GRID):
    """Interpolate every seed's curve onto a common budget grid.

    Returns (grid, mean, lo, hi). Timelines from different seeds land on
    slightly different budget points (verification and certification spend
    depend on the run), so comparing them requires a common abscissa.
    """
    curves = []
    seeds = sub["seed"].unique() if "seed" in sub.columns else [None]
    for sd in seeds:
        d = sub if sd is None else sub[sub.seed == sd]
        d = d.dropna(subset=[x, y]).sort_values(x)
        if len(d) < 2:
            continue
        curves.append(np.interp(grid, d[x].to_numpy(), d[y].to_numpy(),
                                left=np.nan, right=d[y].to_numpy()[-1]))
    if not curves:
        return grid, None, None, None
    A = np.vstack(curves)
    return grid, np.nanmean(A, 0), np.nanmin(A, 0), np.nanmax(A, 0)


def mean_sd(series) -> tuple[float, float]:
    a = np.asarray([v for v in series if v is not None and np.isfinite(v)], dtype=float)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0

VARIANT_C = {  # exp2 scheduler variants
    "full": "#0072B2", "traffic_only": "#E69F00", "hardness_only": "#009E73",
    "disagreement_only": "#CC79A7", "no_signals": "#56B4E9", "random": "#D55E00",
    "partition_major": "#000000", "no_anneal": "#8C564B",
    "popularity_prior": "#7F7F7F", "oracle_traffic": "#17BECF",
}
KIND_C = {  # exp3 adapter kinds
    "procrustes": "#0072B2", "ridge": "#E69F00", "mlp": "#009E73",
    "ensemble": "#CC79A7",
}
KIND_MK = {"procrustes": "o", "ridge": "s", "mlp": "^", "ensemble": "D"}


def endpoint_of(m: dict, which: str, key: str = "ndcg@10"):
    """Endpoint value from a meta dict, tolerating both layouts.

    Single-seed runs stored {"old": {...}, "new": {...}}; multi-seed runs store
    {seed: {"old": {...}, "new": {...}}}, in which case we average over seeds.
    """
    eps = m.get("endpoints") or {}
    if which in eps and isinstance(eps[which], dict) and key in eps[which]:
        return float(eps[which][key])
    vals = [v[which][key] for v in eps.values()
            if isinstance(v, dict) and which in v and key in v[which]]
    return float(np.mean(vals)) if vals else None


def load(name: str) -> pd.DataFrame | None:
    p = RES / name
    if not p.exists():
        print(f"  [skip] missing {name}")
        return None
    return pd.read_csv(p)


def meta(name: str) -> dict:
    p = RES / (name + ".meta.json")
    return json.loads(p.read_text()) if p.exists() else {}


def legend_below(ax, handles=None, labels=None, ncol=None, fontsize=None):
    """Place the legend outside the axes, centred beneath the plot.

    Journal style here is a legend that never occludes data: entries are laid
    out in a single row where they fit, wrapping to at most three columns.
    """
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if ncol is None:
        ncol = len(handles) if len(handles) <= 3 else 3
    kw = {"loc": "upper center", "bbox_to_anchor": (0.5, -0.30),
          "ncol": ncol, "frameon": False, "borderaxespad": 0.0,
          "handlelength": 1.6, "columnspacing": 1.1}
    if fontsize is not None:
        kw["fontsize"] = fontsize
    ax.legend(handles, labels, **kw)


def savefig(fig, name: str):
    out = FIG / name
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


def write_dat(name: str, columns: dict):
    """Whitespace-separated table of plotted series in results/figures/data/,
    readable by most plotting tools; NaN marks a missing point."""
    out = FIG / "data" / name
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(columns).to_csv(out, sep=" ", index=False, float_format="%.6g",
                                 na_rep="nan")
    print(f"  wrote {out.relative_to(ROOT)}")






# ================================================================ exp1
def fig_exp1_timeline(ds: str, pair: str, scale: str, metric: str, ylabel: str,
                      fname: str, endpoints: dict | None = None,
                      legend_loc="lower right"):
    df = load(f"exp1_timeline_{ds}_{pair}_{scale}.csv")
    if df is None:
        return
    m = meta(f"exp1_timeline_{ds}_{pair}_{scale}.csv")
    fig, ax = plt.subplots()
    _exp1_draw(ax, df, m, metric)
    ax.set_xlabel("re-embedding budget (fraction of corpus)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.02, 1.02)
    legend_below(ax)
    savefig(fig, fname)


def fig_exp1_stack(ds: str, pair: str, scale: str, fname: str,
                   rows=(("ndcg@10", "NDCG@10\n(held-out queries)"),
                         ("recall@10_vs_dest_wl", "destination agreement\n(live workload)"))):
    """Several exp1 metrics stacked on one budget axis, with one legend shared
    by the panels."""
    df = load(f"exp1_timeline_{ds}_{pair}_{scale}.csv")
    if df is None:
        return
    m = meta(f"exp1_timeline_{ds}_{pair}_{scale}.csv")
    fig, axes = plt.subplots(len(rows), 1, sharex=True,
                             figsize=(3.35, 0.55 + 1.25 * len(rows)),
                             gridspec_kw={"hspace": 0.14, "left": 0.19, "right": 0.975,
                                          "bottom": 0.13, "top": 0.89})
    for ax, (metric, ylabel) in zip(axes, rows):
        _exp1_draw(ax, df, m, metric)
        ax.set_ylabel(ylabel)
    axes[-1].set_xlabel("re-embedding budget (fraction of corpus)")
    axes[-1].set_xlim(-0.02, 1.02)
    # one legend for all panels, in a fixed reading order
    entries = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            entries.setdefault(lab, h)
    # column-major fill: Metamorph leads, and the two longest labels share a
    # column so three columns fit the figure width
    order = [LABEL[k] for k in ("metamorph", "random_order", "popularity_prior",
                                "full_swap", "static_adapter", "oracle_traffic")
             if LABEL[k] in entries]
    fig.legend([entries[k] for k in order], order, loc="lower center",
               bbox_to_anchor=(0.5, axes[0].get_position().y1 + 0.01), ncol=3,
               frameon=False, borderaxespad=0.0, handlelength=1.4,
               handletextpad=0.5, columnspacing=0.8)
    savefig(fig, fname)


def dat_exp1_timeline(ds: str, pair: str, scale: str, metric: str, tag: str):
    """The series _exp1_draw plots, on the common budget grid: per-method seed
    mean with min/max band, the static adapter's frozen level (and the budget
    where it froze), and the endpoint reference levels."""
    df = load(f"exp1_timeline_{ds}_{pair}_{scale}.csv")
    if df is None:
        return
    m = meta(f"exp1_timeline_{ds}_{pair}_{scale}.csv")
    cols = {"b": GRID}
    y_new = endpoint_of(m, "new", metric)
    y_old = endpoint_of(m, "old", metric)
    if y_new is not None:
        cols["new_model"] = np.full_like(GRID, y_new)
    if y_old is not None and metric != "recall@10_vs_dest":
        cols["old_model"] = np.full_like(GRID, y_old)
    for method in ("full_swap", "random_order", "popularity_prior",
                   "oracle_traffic", "static_adapter", "metamorph"):
        sub = df[df.method == method].dropna(subset=[metric])
        if not len(sub):
            continue
        if method == "static_adapter":
            y = float(sub.sort_values("budget_frac")[metric].iloc[-1])
            frozen = int(np.abs(GRID - sub.budget_frac.max()).argmin())
            assert abs(GRID[frozen] - sub.budget_frac.max()) < 1e-3
            cols[method] = np.full_like(GRID, y)
            cols[f"{method}_mark"] = np.where(np.arange(len(GRID)) == frozen, y, np.nan)
            continue
        g, mu, lo, hi = seed_band(sub, "budget_frac", metric)
        if mu is None:
            continue
        cols[method] = mu
        if (sub.seed.nunique() if "seed" in sub.columns else 1) > 1:
            cols[f"{method}_lo"], cols[f"{method}_hi"] = lo, hi
    write_dat(f"exp1_{tag}_{ds}_{pair}_{scale}.dat", cols)


def _exp1_draw(ax, df: pd.DataFrame, m: dict, metric: str):
    """Endpoint references plus one seed-banded curve per method."""
    # endpoint reference lines (averaged over seeds)
    y_new = endpoint_of(m, "new", metric)
    y_old = endpoint_of(m, "old", metric)
    if y_new is not None:
        ax.axhline(y_new, color=C["new"], lw=0.8, ls=(0, (1, 1)), zorder=1)
        ax.annotate("new model (destination)", xy=(0.02, y_new), xytext=(0, 2),
                    textcoords="offset points", fontsize=6, color=C["new"])
    if y_old is not None and metric != "recall@10_vs_dest":
        ax.axhline(y_old, color=C["old"], lw=0.8, ls=(0, (1, 1)), zorder=1)
        ax.annotate("old model", xy=(0.02, y_old), xytext=(0, -7),
                    textcoords="offset points", fontsize=6, color=C["old"])
    methods = [m for m in ("full_swap", "random_order", "popularity_prior",
                           "oracle_traffic", "static_adapter", "metamorph")
               if m in set(df.method.unique())]
    for method in methods:
        sub = df[df.method == method].dropna(subset=[metric])
        if not len(sub):
            continue
        if method == "static_adapter":
            # serving is frozen after the seed spend: draw its steady-state
            # quality across the whole budget axis (its true x-extent is
            # [0, seed_frac], marked by the marker at its actual budget)
            y = float(sub.sort_values("budget_frac")[metric].iloc[-1])
            ax.plot([0, 1], [y, y], color=C[method], ls=LS[method], lw=1.1,
                    label=LABEL[method], zorder=2)
            ax.plot(sub.budget_frac.max(), y, marker=MK[method], ms=3,
                    color=C[method])
            continue
        g, mu, lo, hi = seed_band(sub, "budget_frac", metric)
        if mu is None:
            continue
        n_seeds = sub.seed.nunique() if "seed" in sub.columns else 1
        if n_seeds > 1:
            ax.fill_between(g, lo, hi, color=C[method], alpha=0.16, lw=0, zorder=1)
        ax.plot(g, mu, color=C[method], ls=LS[method], marker=MK[method], ms=2.2,
                markevery=8, label=LABEL[method],
                zorder=3 if method == "metamorph" else 2)


def fig_exp1_pairs(ds: str, scale: str, pairs: list[str], metric: str,
                   ylabel: str, fname: str):
    """Metamorph timeline across model pairs (robustness)."""
    pc = {"minilm-e5": "#0072B2", "minilm-gte": "#E69F00", "albert-e5": "#009E73"}
    pl = {"minilm-e5": "MiniLM$\\to$E5", "minilm-gte": "MiniLM$\\to$GTE",
          "albert-e5": "ALBERT$\\to$E5 (768$\\to$384)"}
    pm = {"minilm-e5": "o", "minilm-gte": "s", "albert-e5": "^"}
    fig, ax = plt.subplots()
    any_data = False
    for pair in pairs:
        df = load(f"exp1_timeline_{ds}_{pair}_{scale}.csv")
        if df is None:
            continue
        sub = df[df.method == "metamorph"].dropna(subset=[metric]).sort_values("budget_frac")
        if not len(sub):
            continue
        any_data = True
        ax.plot(sub.budget_frac, sub[metric], color=pc[pair], marker=pm[pair],
                ms=2.2, markevery=max(1, len(sub) // 12), label=pl[pair])
    if not any_data:
        plt.close(fig)
        return
    ax.set_xlabel("re-embedding budget (fraction of corpus)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.02, 1.02)
    legend_below(ax)
    savefig(fig, fname)


PAIR_TEX = {"minilm-e5": "m$\\to$e", "minilm-gte": "m$\\to$g",
            "albert-e5": "a$\\to$e", "minilm-e5base": "m$\\to$eb"}






def fig_exp1_verify(ds: str, pair: str, scale: str, fname: str):
    """Separating the two mechanisms: verification is applied to BOTH orderings,
    so its effect is not confounded with the scheduler's."""
    df0 = load(f"exp1_timeline_{ds}_{pair}_{scale}.csv")
    df2 = load(f"exp1_timeline_{ds}_{pair}_{scale}_v2.csv")
    if df0 is None or df2 is None:
        return
    m = meta(f"exp1_timeline_{ds}_{pair}_{scale}.csv")
    fig, ax = plt.subplots()
    for label, y, c in (("old model", endpoint_of(m, "old"), C["old"]),
                        ("new model (destination)", endpoint_of(m, "new"), C["new"])):
        if y is None:
            continue
        ax.axhline(y, color=c, lw=0.8, ls=(0, (1, 1)), zorder=1)
        ax.annotate(label, xy=(0.02, y), xytext=(0, 2), textcoords="offset points",
                    fontsize=6, color=c)
    series = [
        (df0, "metamorph", "Metamorph, $v$=0", C["metamorph"], "--"),
        (df2, "metamorph_v2", "Metamorph, $v$=2", C["metamorph"], "-"),
        (df0, "random_order", "random, $v$=0", C["random_order"], "--"),
        (df2, "random_order_v2", "random, $v$=2", C["random_order"], "-"),
    ]
    for df, mth, lab, c, ls in series:
        sub = df[df.method == mth].dropna(subset=["ndcg@10"])
        if not len(sub):
            continue
        g, mu, lo, hi = seed_band(sub, "budget_frac", "ndcg@10")
        if mu is None:
            continue
        if ls == "-" and (hi - lo).max() > 0:
            ax.fill_between(g, lo, hi, color=c, alpha=0.13, lw=0, zorder=1)
        ax.plot(g, mu, color=c, ls=ls, marker="o" if ls == "-" else None,
                ms=2.0, markevery=10, label=lab, alpha=1.0 if ls == "-" else 0.75)
    ax.set_xlabel("re-embedding budget (fraction of corpus)")
    ax.set_ylabel("NDCG@10 (held-out queries)")
    ax.set_xlim(-0.02, 1.02)
    legend_below(ax, ncol=2, fontsize=5.6)
    savefig(fig, fname)


# ================================================================ exp2
def fig_exp2(ds: str, pair: str, scale: str, metric: str, ylabel: str, fname: str,
             variants=("full", "traffic_only", "disagreement_only", "no_signals",
                       "random", "partition_major")):
    df = load(f"exp2_sched_{ds}_{pair}_{scale}.csv")
    if df is None:
        return
    vls = {"full": "-", "traffic_only": "--", "hardness_only": ":",
           "disagreement_only": "-.", "no_signals": ":", "random": "--",
           "partition_major": "-"}
    fig, ax = plt.subplots()
    for v in variants:
        sub = df[df.variant == v].dropna(subset=[metric]).sort_values("budget_frac")
        if not len(sub):
            continue
        ax.plot(sub.budget_frac, sub[metric], color=VARIANT_C[v], ls=vls[v],
                lw=1.6 if v == "full" else 1.1,
                label=v.replace("_", " "), zorder=3 if v == "full" else 2)
    ax.set_xlabel("re-embedding budget (fraction of corpus)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.02, 1.02)
    legend_below(ax, ncol=2)
    savefig(fig, fname)


def budget_to(df: pd.DataFrame, col: str, tgt: float) -> float:
    sub = df.dropna(subset=[col]).sort_values("budget_frac")
    hit = sub[sub[col] >= tgt]
    return float(hit.budget_frac.iloc[0]) if len(hit) else float("nan")


VARIANT_LABEL = {
    "full": "Metamorph (traffic+hardness+annealed dis.)",
    "traffic_only": "traffic only", "hardness_only": "hardness only",
    "disagreement_only": "disagreement only", "no_signals": "no signals",
    "random": "random order", "partition_major": "partition-major",
    "no_anneal": "no annealing (static mixture)",
    "anneal_perdoc": "per-document annealing",
    "popularity_prior": "offline popularity prior",
    "oracle_traffic": "oracle traffic (upper bd.)",
}
VARIANT_ORDER = ["popularity_prior", "full", "anneal_perdoc", "no_anneal", "traffic_only",
                 "hardness_only", "partition_major", "disagreement_only",
                 "random", "oracle_traffic"]
VARIANT_SKIP = {"no_signals"}   # uniform priorities: reported in the text only




# ================================================================ exp3
def fig_exp3_seed_sweep(ds: str, pair: str, scale: str, fname: str,
                        metric="recall_vs_dest",
                        ylabel="recall@10 vs destination (exact search)"):
    df = load(f"exp3_adapters_{ds}_{pair}_{scale}.csv")
    if df is None:
        return
    df = df[df.part == "A"]
    fig, ax = plt.subplots()
    for kind in ("procrustes", "ridge", "mlp", "ensemble"):
        sub = df[df.kind == kind].sort_values("seed_frac")
        if not len(sub):
            continue
        ax.plot(sub.seed_frac * 100, sub[metric], color=KIND_C[kind],
                marker=KIND_MK[kind], ms=3, label=kind)
    ax.set_xlabel("adapter seed sample (% of corpus)")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    legend_below(ax)
    savefig(fig, fname)




# ================================================================ exp4
def fig_exp4_curves(ds: str, pair: str, scale: str, fname: str):
    df = load(f"exp4_warmstart_{ds}_{pair}_{scale}.csv")
    if df is None:
        return
    nn = df[df.mechanism == "nn_descent"]
    fig, ax = plt.subplots()
    for tag, col, ls, mk in (("cold", "#E69F00", "--", "s"),
                             ("warm (old-graph init)", "#0072B2", "-", "o")):
        v = "cold" if tag == "cold" else "warm"
        sub = nn[nn.variant == v].sort_values("iter")
        ax.plot(sub.dist_comps / 1e6, sub.graph_recall, color=col, ls=ls,
                marker=mk, ms=2.5, label=tag)
    ax.set_xlabel("distance computations (millions)")
    ax.set_ylabel("kNN-graph recall (new space)")
    legend_below(ax)
    savefig(fig, fname)




# ================================================================ exp5
def fig_exp5_timeline(ds: str, pair: str, scale: str, fname: str,
                      m_sel=60, c_sel=50, rep_sel=0):
    df = load(f"exp5_cert_{ds}_{pair}_{scale}.csv")
    if df is None:
        return
    sub = df[(df.m == m_sel) & (df.pool_extra == c_sel) & (df.rep == rep_sel)
             ].sort_values("budget_frac")
    if not len(sub):
        return
    fig, ax = plt.subplots()
    ax.plot(sub.budget_frac, sub.recall_true, color="#333333", ls="-", marker="o",
            ms=2.2, label="true recall vs dest.")
    ax.plot(sub.budget_frac, sub.recall_hat, color="#0072B2", ls="--", marker="s",
            ms=2.2, label="estimate $\\hat{R}$")
    ax.plot(sub.budget_frac, sub.pooled_lb, color="#E69F00", ls="-.", marker="^",
            ms=2.2, label="anytime lower bound")
    tau = meta(f"exp5_cert_{ds}_{pair}_{scale}.csv").get("tau", 0.9)
    ax.axhline(tau, color="#888888", lw=0.8, ls=(0, (1, 1)))
    ax.annotate(f"$\\tau$={tau}", xy=(0.02, tau), xytext=(0, 2),
                textcoords="offset points", fontsize=6, color="#666666")
    ax.set_xlabel("re-embedding budget (fraction of corpus)")
    ax.set_ylabel("serving recall@10 vs destination")
    legend_below(ax)
    savefig(fig, fname)


def fig_exp5_tau_sweep(ds: str, pair: str, scale: str, fname: str,
                       taus=(0.80, 0.85, 0.90, 0.95)):
    """Fire budget (pooled LB >= tau) vs true-cross budget, per tau, averaged
    over (m, C, rep) runs. Computed post-hoc from the rounds CSV."""
    df = load(f"exp5_cert_{ds}_{pair}_{scale}.csv")
    if df is None:
        return
    rows = []
    for (m_, c_, rep_), g in df.groupby(["m", "pool_extra", "rep"]):
        g = g.sort_values("budget_frac")
        for tau in taus:
            fire = g.loc[g.pooled_lb >= tau, "budget_frac"].min()
            cross = g.loc[g.recall_true >= tau, "budget_frac"].min()
            rows.append({"m": m_, "C": c_, "rep": rep_, "tau": tau,
                         "fire": fire, "cross": cross})
    t = pd.DataFrame(rows)
    agg = t.groupby("tau")[["fire", "cross"]].mean()
    fig, ax = plt.subplots()
    x = np.arange(len(agg))
    w = 0.36
    ax.bar(x - w / 2, agg["cross"], w, color="#333333", label="true quality crosses $\\tau$")
    ax.bar(x + w / 2, agg["fire"], w, color="#E69F00", label="certifier fires (LB$\\geq\\tau$)")
    ax.set_xticks(x, [f"{t_:.2f}" for t_ in agg.index])
    ax.set_xlabel("certification target $\\tau$")
    ax.set_ylabel("budget fraction")
    legend_below(ax)
    savefig(fig, fname)
    return t




# ================================================================ exp6
def fig_exp6(ds: str, pair: str, scale: str, fname: str):
    df = load(f"exp6_latency_{ds}_{pair}_{scale}.csv")
    if df is None:
        return
    stages = ["search_old", "search_new", "rerank", "verify"]
    scol = {"search_old": "#0072B2", "search_new": "#56B4E9",
            "rerank": "#E69F00", "verify": "#CC79A7"}
    fig, ax = plt.subplots()
    for v, hatch in ((0, None), (2, "//")):
        sub = df[df.verify_per_query == v].sort_values("budget_frac_target")
        if not len(sub):
            continue
        x = np.arange(len(sub)) + (0 if v == 0 else 0.38)
        bottom = np.zeros(len(sub))
        for st in stages:
            vals = sub[f"stage_ms_{st}"].to_numpy()
            ax.bar(x, vals, 0.34, bottom=bottom, color=scol[st], hatch=hatch,
                   edgecolor="white", linewidth=0.4,
                   label=st.replace("_", " ") if v == 0 else None)
            bottom += vals
    sub0 = df[df.verify_per_query == 0].sort_values("budget_frac_target")
    ax.set_xticks(np.arange(len(sub0)) + 0.19,
                  [f"{b:.0%}" for b in sub0.budget_frac_target])
    ax.set_xlabel("migration progress (budget fraction); right bar: verify $v$=2")
    ax.set_ylabel("mean per-query latency (ms)")
    legend_below(ax, fontsize=6)
    savefig(fig, fname)




# ================================================================ main
def main():
    small_runs = [("scifact", "minilm-e5", "small"),
                  ("scifact", "minilm-gte", "small"),
                  ("scifact", "albert-e5", "small"),
                  ("scifact", "minilm-e5base", "small"),
                  ("nfcorpus", "minilm-e5", "small")]
    full_runs = [("fiqa", "minilm-e5", "full"), ("fiqa", "minilm-gte", "full")]

    print("== exp1 figures")
    for ds, pair, scale in small_runs + full_runs:
        fig_exp1_timeline(ds, pair, scale, "ndcg@10", "NDCG@10 (held-out queries)",
                          f"exp1_ndcg_{ds}_{pair}_{scale}.pdf")
        fig_exp1_timeline(ds, pair, scale, "recall@10_vs_dest",
                          "recall@10 vs destination",
                          f"exp1_recall_{ds}_{pair}_{scale}.pdf")
        fig_exp1_timeline(ds, pair, scale, "recall@10_vs_dest_wl",
                          "recall@10 vs destination (workload dist.)",
                          f"exp1_recall_wl_{ds}_{pair}_{scale}.pdf")
    fig_exp1_stack("fiqa", "minilm-e5", "full", "exp1_stack_fiqa_minilm-e5_full.pdf")
    dat_exp1_timeline("fiqa", "minilm-e5", "full", "ndcg@10", "ndcg")
    dat_exp1_timeline("fiqa", "minilm-e5", "full", "recall@10_vs_dest_wl", "wl")
    fig_exp1_verify("fiqa", "minilm-e5", "full", "exp1_verify_fiqa_minilm-e5_full.pdf")
    fig_exp1_verify("scifact", "albert-e5", "small", "exp1_verify_scifact_albert-e5_small.pdf")
    fig_exp1_pairs("scifact", "small",
                   ["minilm-e5", "minilm-gte", "albert-e5"],
                   "recall@10_vs_dest", "recall@10 vs destination",
                   "exp1_pairs_scifact_small.pdf")

    print("== exp2 figures")
    for ds, pair, scale in [("scifact", "minilm-e5", "small")] + full_runs:
        fig_exp2(ds, pair, scale, "recall@10_vs_dest_wl",
                 "recall@10 vs destination (workload dist.)",
                 f"exp2_sched_wl_{ds}_{pair}_{scale}.pdf")
        fig_exp2(ds, pair, scale, "recall@10_vs_dest",
                 "recall@10 vs destination",
                 f"exp2_sched_{ds}_{pair}_{scale}.pdf")

    print("== exp3 figures")
    for ds, pair, scale in [("scifact", "minilm-e5", "small"),
                            ("scifact", "albert-e5", "small")]:
        fig_exp3_seed_sweep(ds, pair, scale, f"exp3_seed_{ds}_{pair}_{scale}.pdf")

    print("== exp4 figures")
    for ds, pair, scale in [("scifact", "minilm-e5", "small")] + full_runs:
        fig_exp4_curves(ds, pair, scale, f"exp4_nn_descent_{ds}_{pair}_{scale}.pdf")

    print("== exp5 figures")
    fig_exp5_timeline("scifact", "minilm-e5", "small",
                      "exp5_cert_timeline_scifact_minilm-e5_small.pdf")
    fig_exp5_tau_sweep("scifact", "minilm-e5", "small",
                       "exp5_tau_scifact_minilm-e5_small.pdf")
    fig_exp5_timeline("fiqa", "minilm-e5", "full",
                      "exp5_cert_timeline_fiqa_minilm-e5_full.pdf",
                      m_sel=60, c_sel=50, rep_sel=0)
    fig_exp5_tau_sweep("fiqa", "minilm-e5", "full",
                       "exp5_tau_fiqa_minilm-e5_full.pdf", taus=(0.80, 0.85, 0.90))

    print("== exp6 figures")
    for ds, pair, scale in [("scifact", "minilm-e5", "small")] + full_runs:
        fig_exp6(ds, pair, scale, f"exp6_latency_{ds}_{pair}_{scale}.pdf")
        fig_exp6_compaction(ds, pair, scale, f"exp6_compaction_{ds}_{pair}_{scale}.pdf")
        dat_exp6_compaction(ds, pair, scale)

    print("== exp5 metric comparison")
    for ds, pair, scale in [("scifact", "minilm-e5", "small")] + full_runs:
        fig_exp5_metric_compare(ds, pair, scale,
                                f"exp5_metrics_{ds}_{pair}_{scale}.pdf")

    print("== exp7 cost")
    for ds, pair, scale in [("quora", "minilm-e5", "xl"),
                            ("msmarco", "minilm-e5", "xxl")]:
        fig_exp7_throughput(ds, pair, scale)

    print("== exp8 per-query certificate")
    cert_runs = [("scifact", "minilm-e5", "small"), ("scifact", "albert-e5", "small")]
    for ds, pair, scale in cert_runs:
        fig_exp8_certificate(ds, pair, scale, f"exp8_cert_{ds}_{pair}_{scale}.pdf")

    print("== exp9 scale ladder")
    import glob as _glob
    quora_lad, ms_lad = ladder_files()
    if quora_lad:
        fig_exp9_ladder(quora_lad, "exp9_ladder_quora.pdf")
        fig_exp9_gap(quora_lad, "exp9_gap_quora.pdf")
    if ms_lad:
        fig_exp9_ladder(ms_lad, "exp9_ladder_msmarco.pdf")
        fig_exp9_gap(ms_lad, "exp9_gap_msmarco.pdf", "MS MARCO")
    if quora_lad or ms_lad:
        fig_exp9_both(quora_lad, ms_lad)
        dat_exp9_gap(quora_lad, "quora")
        dat_exp9_gap(ms_lad, "msmarco")

    print("== exp4b FAISS warm start")
    faiss_files = sorted(Path(f).name for f in _glob.glob(str(RES / "exp4b_faiss_*_summary.csv")))
    if faiss_files:
        quora_faiss = [f for f in faiss_files if "quora" in f or "scifact" in f]
        fig_exp4b_scale(quora_faiss)



# ================================================================ exp7: cost
def fig_exp7_throughput(ds: str, pair: str, scale: str):
    """Encoder throughput against the scheduler batch size (exp7)."""
    tp = load(f"exp7_throughput_{ds}_{pair}_{scale}.csv")
    if tp is not None and "reveal_batch" in tp.columns:
        fig, ax = plt.subplots()
        drew = False
        for other, lab, c, mk in ((f"exp7_throughput_{ds}_{pair}_{scale}.csv",
                                   f"{ds}", C["metamorph"], "o"),
                                  ("exp7_throughput_msmarco_minilm-e5_xxl.csv",
                                   "MS MARCO", C["random_order"], "s")):
            d2 = load(other)
            if d2 is None or "reveal_batch" not in d2.columns:
                continue
            sub = d2.dropna(subset=["reveal_batch"]).sort_values("reveal_batch")
            if not len(sub):
                continue
            drew = True
            ax.plot(sub.reveal_batch, sub.docs_per_s, color=c, marker=mk, ms=3,
                    label=f"{lab} ({sub.docs_per_s.max()/sub.docs_per_s.min():.1f}$\\times$)")
        if drew:
            ax.set_xscale("log", base=2)
            ax.set_xlabel("documents re-embedded per scheduler batch")
            ax.set_ylabel("encoder throughput (docs/s)")
            legend_below(ax)
            savefig(fig, f"exp7_batch_throughput_{ds}_{pair}_{scale}.pdf")
        else:
            plt.close(fig)


# ================================================================ exp9: ladder
LADDER_MAX_QUERIES = 4000   # the workload every reported ladder rung shares


def _ladder_workload(name: str) -> int | None:
    """The --max-queries recorded in a ladder CSV's sidecar.

    Rungs are comparable only when the workload is identical, so an
    exploratory run at a different query count must not be pooled into the
    per-N mean: the 8.84M pass at 3,000 queries was superseded by a matched
    4,000-query run, and averaging the two silently mixes workloads at the
    one point where the scale claim is read off.
    """
    meta = RES / f"{name}.meta.json"
    if not meta.exists():
        return None
    argv = json.loads(meta.read_text()).get("argv", [])
    if "--max-queries" in argv:
        return int(argv[argv.index("--max-queries") + 1])
    return None


def ladder_files() -> tuple[list[str], list[str]]:
    """The Quora and MS MARCO ladder CSVs run at the reported workload."""
    import glob
    lad = sorted(Path(f).name for f in glob.glob(str(RES / "exp9_ladder_*.csv"))
                 if "timelines" not in f and not f.endswith(".meta.json"))
    dropped = [f for f in lad if _ladder_workload(f) != LADDER_MAX_QUERIES]
    for f in dropped:
        print(f"   skipping {f}: workload {_ladder_workload(f)} queries "
              f"!= ladder's {LADDER_MAX_QUERIES}")
    lad = [f for f in lad if _ladder_workload(f) == LADDER_MAX_QUERIES]
    return ([f for f in lad if "quora" in f and "smoke" not in f],
            [f for f in lad if "msmarco" in f])


def _ladder(tag_files: list[str]) -> pd.DataFrame | None:
    parts = [load(f) for f in tag_files]
    parts = [p for p in parts if p is not None]
    return pd.concat(parts, ignore_index=True) if parts else None


def fig_exp9_ladder(files: list[str], fname: str, metric: str = "wl_agree@50",
                    ylabel: str = "agreement with destination @ 50\\% budget"):
    df = _ladder(files)
    if df is None or not len(df):
        return
    fig, ax = plt.subplots()
    for method in ("metamorph", "popularity_prior", "random_order"):
        sub = df[df.method == method]
        if not len(sub):
            continue
        g = sub.groupby("n_docs")[metric]
        xs = np.array(sorted(g.groups.keys()), dtype=float)
        mu = np.array([g.get_group(x).mean() for x in xs])
        lo = np.array([g.get_group(x).min() for x in xs])
        hi = np.array([g.get_group(x).max() for x in xs])
        if (hi - lo).max() > 0:
            ax.fill_between(xs, lo, hi, color=C[method], alpha=0.16, lw=0)
        ax.plot(xs, mu, color=C[method], ls=LS[method], marker=MK[method], ms=3.2,
                label=LABEL[method])
    ax.set_xscale("log")
    ax.set_xlabel("corpus size $N$ (documents)")
    ax.set_ylabel(ylabel.replace("\\%", "%"))
    legend_below(ax)
    savefig(fig, fname)


def fig_exp9_gap(files: list[str], fname: str, label: str = ""):
    """The scale claim itself: advantage over random, and the hot-set fraction."""
    df = _ladder(files)
    if df is None or not len(df):
        return
    piv = df.groupby(["n_docs", "method"])["wl_agree@50"].mean().unstack()
    if "metamorph" not in piv or "random_order" not in piv:
        return
    fig, ax = plt.subplots()
    xs = piv.index.to_numpy(dtype=float)
    ax.plot(xs, piv["metamorph"] - piv["random_order"], color=C["metamorph"],
            marker="o", ms=3.4, label="vs. random order")
    if "popularity_prior" in piv:
        ax.plot(xs, piv["metamorph"] - piv["popularity_prior"],
                color=C["popularity_prior"], ls=LS["popularity_prior"],
                marker=MK["popularity_prior"], ms=3.4, label="vs. offline prior")
    ax.axhline(0, color="#888888", lw=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("corpus size $N$ (documents)" + (f" --- {label}" if label else ""))
    ax.set_ylabel("agreement gap @ 50% budget")
    ax2 = ax.twinx()
    hot = df.groupby("n_docs")["hot_set_frac"].mean()
    ax2.plot(hot.index.to_numpy(dtype=float), hot.to_numpy(), color="#888888",
             ls=(0, (1, 1)), lw=1.0, label="hot-set fraction")
    ax2.set_ylabel("corpus fraction touched by traffic", color="#555555")
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    legend_below(ax, h1 + h2, l1 + l2)
    savefig(fig, fname)


def fig_exp9_both(quora: list[str], msmarco: list[str], fname: str = "exp9_gap_both.pdf"):
    """Both ladders on one axis: the trend is the claim, the level is not."""
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    for files, lab, c, mk in ((quora, "quora", C["metamorph"], "o"),
                              (msmarco, "MS MARCO", C["random_order"], "s")):
        df = _ladder(files)
        if df is None or not len(df):
            continue
        piv = df.groupby(["n_docs", "method"])["wl_agree@50"].mean().unstack()
        if "metamorph" not in piv or "random_order" not in piv:
            continue
        xs = piv.index.to_numpy(dtype=float)
        ax.plot(xs, piv["metamorph"] - piv["random_order"], color=c, marker=mk,
                ms=3.4, label=f"{lab}: vs. random")
        if "popularity_prior" in piv:
            ax.plot(xs, piv["metamorph"] - piv["popularity_prior"], color=c,
                    ls="--", marker=mk, ms=2.6, alpha=0.6,
                    label=f"{lab}: vs. prior")
        hot = df.groupby("n_docs")["hot_set_frac"].mean()
        ax2.plot(hot.index.to_numpy(dtype=float), hot.to_numpy(), color=c,
                 ls=(0, (1, 1)), lw=0.9, alpha=0.5)
    ax.axhline(0, color="#888888", lw=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("corpus size $N$ (documents)")
    ax.set_ylabel("agreement gap @ 50% budget")
    ax2.set_ylabel("hot-set fraction (dotted)", color="#555555")
    ax2.grid(False)
    legend_below(ax, fontsize=5.8)
    savefig(fig, fname)


def dat_exp9_gap(files: list[str], tag: str):
    """fig_exp9_both's series for one ladder: Metamorph's half-budget agreement
    gap over random order and over the offline prior, and the hot-set fraction."""
    df = _ladder(files)
    if df is None or not len(df):
        return
    piv = df.groupby(["n_docs", "method"])["wl_agree@50"].mean().unstack()
    if "metamorph" not in piv or "random_order" not in piv:
        return
    cols = {"n": piv.index.to_numpy(dtype=float),
            "gap_random": (piv["metamorph"] - piv["random_order"]).to_numpy()}
    if "popularity_prior" in piv:
        cols["gap_prior"] = (piv["metamorph"] - piv["popularity_prior"]).to_numpy()
    cols["hot"] = df.groupby("n_docs")["hot_set_frac"].mean().reindex(piv.index).to_numpy()
    write_dat(f"exp9_gap_{tag}.dat", cols)




# ================================================================ exp4b: FAISS


# ================================================================ exp8: certificate
def fig_exp8_certificate(ds: str, pair: str, scale: str, fname: str):
    df = load(f"exp8_margin_{ds}_{pair}_{scale}.csv")
    if df is None or not len(df):
        return
    df = df.sort_values("budget_frac")
    fig, ax = plt.subplots()
    ax.plot(df.budget_frac, df.mean_pool_agreement, color=C["new"], lw=1.2,
            label="true pool agreement")
    ax.plot(df.budget_frac, df.mean_cert_lb_hat, color=C["metamorph"], marker="o",
            ms=2.6, label="certified LB (measurable $\\hat\\varepsilon$)")
    ax.plot(df.budget_frac, df.mean_cert_lb_oracle, color=C["random_order"],
            ls="--", marker="s", ms=2.6, label="certified LB (oracle $\\varepsilon$)")
    ax.plot(df.budget_frac, df.frac_cond_true, color=C["static_adapter"], ls=":",
            label="classical margin condition")
    ax.set_xlabel("re-embedding budget (fraction of corpus)")
    ax.set_ylabel("per-query certified recall")
    ax.set_ylim(-0.03, 1.03)
    legend_below(ax)
    savefig(fig, fname)




# ================================================================ exp5: target metrics


def fig_exp5_metric_compare(ds: str, pair: str, scale: str, fname: str):
    """Why certifying score mass fires earlier than certifying set agreement."""
    df = load(f"exp5_cert_{ds}_{pair}_{scale}.csv")
    if df is None or not len(df):
        return
    sub = df[(df.pool_extra >= 20)] if "pool_extra" in df else df
    if "bound" in sub:
        sub = sub[sub.bound == "betting"]
    if not len(sub):
        return
    fig, ax = plt.subplots()
    for col, lab, c, ls in (("recall_true", "true top-10 agreement", C["random_order"], "--"),
                            ("score_mass_true", "true score mass", C["metamorph"], "-"),
                            ("ndcg_ratio_true", "true NDCG ratio", C["full_swap"], "-.")):
        if col not in sub:
            continue
        g = sub.groupby(pd.cut(sub.budget_frac, np.linspace(0, 1, 21)),
                        observed=True)[col].mean()
        xs = np.array([iv.mid for iv in g.index])
        ax.plot(xs, g.to_numpy(), color=c, ls=ls, marker="o", ms=2.2, label=lab)
    ax.set_xlabel("re-embedding budget (fraction of corpus)")
    ax.set_ylabel("quality vs destination")
    legend_below(ax)
    savefig(fig, fname)


# ================================================================ exp6: compaction


def fig_exp6_compaction(ds: str, pair: str, scale: str, fname: str):
    df = load(f"exp6_latency_{ds}_{pair}_{scale}.csv")
    if df is None or "compaction_ratio" not in df.columns:
        return
    fig, ax = plt.subplots()
    for ratio, lab, c, ls in ((0.0, "shared masked old index", C["random_order"], "--"),
                              (0.5, "compacting old index", C["metamorph"], "-")):
        sub = df[(df.compaction_ratio == ratio) & (df.verify_per_query == 0)]
        if not len(sub):
            continue
        sub = sub.sort_values("budget_frac_target")
        ax.plot(sub.budget_frac_target, sub.lat_ms_p50, color=c, ls=ls, marker="o",
                ms=3, label=lab + " (p50)")
        ax.plot(sub.budget_frac_target, sub.lat_ms_p99, color=c, ls=ls, marker="^",
                ms=3, alpha=0.55, label=lab + " (p99)")
    ax.set_xlabel("migration progress (budget fraction)")
    ax.set_ylabel("per-query latency (ms)")
    legend_below(ax, ncol=2)
    savefig(fig, fname)


def dat_exp6_compaction(ds: str, pair: str, scale: str):
    """fig_exp6_compaction's series: p50/p99 latency without verification, for
    the shared masked source index and the compacting one."""
    df = load(f"exp6_latency_{ds}_{pair}_{scale}.csv")
    if df is None or "compaction_ratio" not in df.columns:
        return
    for ratio, tag in ((0.0, "masked"), (0.5, "compacting")):
        sub = df[(df.compaction_ratio == ratio) & (df.verify_per_query == 0)]
        if not len(sub):
            continue
        sub = sub.sort_values("budget_frac_target")
        write_dat(f"exp6_compaction_{tag}_{ds}_{pair}_{scale}.dat",
                  {"b": sub.budget_frac_target.to_numpy(),
                   "p50": sub.lat_ms_p50.to_numpy(), "p99": sub.lat_ms_p99.to_numpy()})

def fig_exp4b_scale(files: list[str], fname: str = "exp4b_faiss_scale.pdf"):
    """Warm-start advantage in a production index, against corpus size."""
    parts = [load(f) for f in files]
    parts = [p for p in parts if p is not None]
    if not parts:
        return
    df = pd.concat(parts, ignore_index=True)
    g = df.groupby("n_docs").agg(build=("build_speedup", "mean"),
                                 train=("train_speedup", "mean"))
    if len(g) < 2:
        return
    fig, ax = plt.subplots()
    xs = g.index.to_numpy(dtype=float)
    ax.plot(xs, g.build.to_numpy(), color=C["metamorph"], marker="o", ms=3.2,
            label="index build time")
    ax.plot(xs, g.train.to_numpy(), color=C["random_order"], ls="--", marker="s",
            ms=3.2, label="coarse-quantizer training time")
    ax.axhline(1.0, color="#888888", lw=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("corpus size $N$ (documents)")
    ax.set_ylabel("speed-up at matched recall ($\\times$)")
    legend_below(ax)
    savefig(fig, fname)


if __name__ == "__main__":
    main()
