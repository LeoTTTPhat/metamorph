"""Shared infrastructure for the experiment scripts.

Every experiment is parameterized by --dataset, --pair, --scale, --seed and
writes CSV rows to results/ plus a .meta.json sidecar recording the exact
configuration (honest-scale reporting: the actual N always lands in the meta
and in every CSV row).

Prepared data (embeddings for both models) is produced once by
prepare_data.py and cached as an .npz under data/prepared/.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PREPARED_DIR = DATA_DIR / "prepared"
RESULTS_DIR = ROOT / "results"

sys.path.insert(0, str(ROOT / "src"))  # allow running without pip install

from metamorph.config import (AdapterConfig, CertifierConfig, EngineConfig,  # noqa: E402
                              SchedulerConfig, ServeConfig)
from metamorph.migrate import MigrationData  # noqa: E402

# ---------------------------------------------------------------- model pairs
# OLD -> NEW upgrade pairs. All small models (laptop download budget):
#   MiniLM ~90MB, e5-small-v2 ~130MB, gte-small ~70MB, albert-small ~45MB.
MODEL_PAIRS: dict[str, tuple[str, str]] = {
    # P1: the canonical upgrade of the evaluation (same dim, 384 -> 384)
    "minilm-e5": ("sentence-transformers/all-MiniLM-L6-v2", "intfloat/e5-small-v2"),
    # P2: second real pair (384 -> 384)
    "minilm-gte": ("sentence-transformers/all-MiniLM-L6-v2", "thenlper/gte-small"),
    # P3: dimension-DECREASING upgrade (768 -> 384): an older, weaker 768d model
    # to a newer, better 384d model.
    "albert-e5": ("sentence-transformers/paraphrase-albert-small-v2", "intfloat/e5-small-v2"),
    # P4: dimension-INCREASING upgrade (384 -> 768), so the pair sweep covers
    # both directions of dimension change rather than only shrinkage.
    "minilm-e5base": ("sentence-transformers/all-MiniLM-L6-v2", "intfloat/e5-base-v2"),
    # P5: LARGE destination encoder (384 -> 1024, 22M -> 335M params). The other
    # four pairs are all compact sentence encoders; this one tests whether the
    # per-query fusion calibration still holds when the destination space is an
    # order of magnitude larger in both dimension and capacity.
    "minilm-e5large": ("sentence-transformers/all-MiniLM-L6-v2", "intfloat/e5-large-v2"),
}

# ------------------------------------------------------------------- scales
# Caps on corpus size. The dataset itself is chosen independently (--dataset);
# a cap larger than the corpus means "use the full corpus".
SCALE_MAX_DOCS: dict[str, int] = {
    "smoke": 1000,      # minutes end-to-end; CI-style sanity runs
    "small": 6000,      # full scifact / nfcorpus
    "full": 60000,      # full fiqa / a 60k quora subset
    # large scales are prepared by experiments/prepare_large.py as memory-mapped
    # *bundles*; the scale ladder sub-samples ONE bundle so corpus size varies
    # with dataset, encoder pair and workload held exactly fixed.
    "xl": 1_000_000,
    "xxl": 10_000_000,
}

DEFAULT_DATASET_FOR_SCALE = {"smoke": "scifact", "small": "scifact", "full": "fiqa"}


def add_common_args(ap: argparse.ArgumentParser, default_pair: str = "minilm-e5") -> None:
    ap.add_argument("--dataset", default=None,
                    help="BEIR dataset (default depends on --scale)")
    ap.add_argument("--pair", default=default_pair, choices=sorted(MODEL_PAIRS))
    ap.add_argument("--scale", default="smoke", choices=sorted(SCALE_MAX_DOCS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="output CSV path (default: results/<auto>.csv)")


def resolve_dataset(args) -> str:
    return args.dataset or DEFAULT_DATASET_FOR_SCALE[args.scale]


def prepared_path(dataset: str, pair: str, scale: str) -> Path:
    return PREPARED_DIR / f"{dataset}_{pair}_{scale}.npz"


def bundle_path(dataset: str, pair: str, tag: str) -> Path:
    return PREPARED_DIR / f"{dataset}_{pair}_{tag}"


def load_bundle(dataset: str, pair: str, tag: str, n_docs: int | None = None,
                seed: int = 42, max_queries: int | None = None) -> MigrationData:
    """Load a memory-mapped large bundle, optionally sub-sampled to `n_docs`.

    Sub-sampling keeps every qrel-positive document plus a seeded random fill,
    so relevance judgements stay meaningful at every rung of the scale ladder.
    The returned arrays are materialised in RAM (the engine indexes them
    randomly, so mmap page-faulting would dominate); at 10M x 384 float32 that
    is ~15 GB per space.
    """
    root = bundle_path(dataset, pair, tag)
    if not root.exists():
        raise SystemExit(
            f"bundle not found: {root}\nrun: python experiments/prepare_large.py "
            f"--dataset {dataset} --pair {pair} --tag {tag}")
    meta = json.loads((root / "meta.json").read_text())
    doc_ids = [str(x) for x in np.load(root / "doc_ids.npy", allow_pickle=False)]
    q_ids = [str(x) for x in np.load(root / "q_ids.npy", allow_pickle=False)]
    qrels_raw = json.loads((root / "qrels.json").read_text())
    qrels = {q: {int(d): float(g) for d, g in rel.items()} for q, rel in qrels_raw.items()}

    old = np.load(root / "old.npy", mmap_mode="r")
    new = np.load(root / "new.npy", mmap_mode="r")
    q_old = np.load(root / "q_old.npy", mmap_mode="r")
    q_new = np.load(root / "q_new.npy", mmap_mode="r")

    N = old.shape[0]
    if n_docs is not None and n_docs < N:
        rng = np.random.default_rng(seed)
        positives = np.unique(np.concatenate(
            [np.fromiter(rel.keys(), dtype=np.int64) for rel in qrels.values()]
        )) if qrels else np.empty(0, dtype=np.int64)
        fill = max(0, n_docs - positives.size)
        rest = np.setdiff1d(np.arange(N, dtype=np.int64), positives,
                            assume_unique=False)
        take = rng.choice(rest, size=min(fill, rest.size), replace=False)
        keep = np.sort(np.concatenate([positives, take]))
        remap = {int(o): i for i, o in enumerate(keep)}
        old = np.ascontiguousarray(old[keep])
        new = np.ascontiguousarray(new[keep])
        doc_ids = [doc_ids[i] for i in keep]
        qrels = {q: {remap[d]: g for d, g in rel.items() if d in remap}
                 for q, rel in qrels.items()}
        qrels = {q: rel for q, rel in qrels.items() if rel}
        keep_q = [i for i, q in enumerate(q_ids) if q in qrels]
        q_ids = [q_ids[i] for i in keep_q]
        q_old = np.ascontiguousarray(q_old[keep_q])
        q_new = np.ascontiguousarray(q_new[keep_q])
    else:
        old, new = np.ascontiguousarray(old), np.ascontiguousarray(new)
        q_old, q_new = np.ascontiguousarray(q_old), np.ascontiguousarray(q_new)

    if max_queries is not None and len(q_ids) > max_queries:
        rng = np.random.default_rng(seed + 1)
        sel = np.sort(rng.choice(len(q_ids), size=max_queries, replace=False))
        q_ids = [q_ids[i] for i in sel]
        q_old, q_new = q_old[sel], q_new[sel]
        qrels = {q: qrels[q] for q in q_ids if q in qrels}

    meta = dict(meta)
    meta.update({"n_docs": old.shape[0], "n_queries": len(q_ids),
                 "bundle_n_docs": N, "subsampled": bool(n_docs and n_docs < N)})
    return MigrationData(doc_ids=doc_ids, old=old, new=new, q_ids=q_ids,
                         q_old=q_old, q_new=q_new, qrels=qrels, meta=meta)


def load_any(dataset: str, pair: str, scale: str, n_docs: int | None = None,
             seed: int = 42, max_queries: int | None = None) -> MigrationData:
    """Load either a compressed .npz (small scales) or a bundle (large scales)."""
    if prepared_path(dataset, pair, scale).exists():
        return load_prepared(dataset, pair, scale)
    return load_bundle(dataset, pair, scale, n_docs=n_docs, seed=seed,
                       max_queries=max_queries)


def load_prepared(dataset: str, pair: str, scale: str) -> MigrationData:
    path = prepared_path(dataset, pair, scale)
    if not path.exists():
        raise SystemExit(
            f"prepared data not found: {path}\n"
            f"run: python experiments/prepare_data.py --dataset {dataset} "
            f"--pair {pair} --scale {scale}")
    z = np.load(path, allow_pickle=False)
    qrels_raw = json.loads(str(z["qrels_json"]))
    qrels = {q: {int(d): float(g) for d, g in rel.items()} for q, rel in qrels_raw.items()}
    meta = json.loads(str(z["meta_json"]))
    return MigrationData(
        doc_ids=[str(d) for d in z["doc_ids"]],
        old=z["old"], new=z["new"],
        q_ids=[str(q) for q in z["q_ids"]],
        q_old=z["q_old"], q_new=z["q_new"],
        qrels=qrels, meta=meta)


# --------------------------------------------------------------- engine config
def default_engine_config(n_docs: int, scale: str, seed: int) -> EngineConfig:
    """Engine defaults scaled to the corpus so every scale yields a timeline
    with a comparable number of steps and useful certification cadence."""
    # certifier sample sizes scale with the corpus: a certification round's
    # reveals must stay a small fraction of N (they are recycled but still
    # spend budget; at deployment scale m*pool/N is negligible by design)
    # certifier rounds are COST-CAPPED (cap = fraction of N revealed per
    # round); sample_queries is a per-round max, reached only when pools are
    # cheap (late in migration). The anytime decision bound pools recent rounds.
    # Round cost caps. A certification round costs O(m * |pool|) reveals, so its
    # cost AS A FRACTION OF THE CORPUS falls like 1/N: the certifier is
    # intrinsically a large-corpus mechanism, and on a 5k-document corpus a
    # single pool is already 1% of everything. Small scales therefore get a
    # larger cap (reveals are recycled as migration progress, so a larger cap
    # is redirected migration rather than waste) purely so the mechanism is
    # observable there at all; see the scale ladder for the real trend.
    presets = {
        "smoke": dict(batch=50, parts=8, wl=16, ev=1, m=40, pool=10, period=3,
                      cap=0.02, seed_frac=0.05),
        "small": dict(batch=100, parts=16, wl=32, ev=2, m=100, pool=20, period=4,
                      cap=0.012, seed_frac=0.02),
        "full": dict(batch=500, parts=32, wl=32, ev=2, m=150, pool=30, period=5,
                     cap=0.008, seed_frac=0.02),
        # large scales: batch is set as a FRACTION of N below so the timeline
        # has a comparable number of points (~120) at every rung of the ladder.
        "xl": dict(batch=0, parts=64, wl=64, ev=2, m=200, pool=30, period=5,
                   cap=0.002, seed_frac=0.02),
        # at 10^7 the evaluation harness dominates wall-clock, so timeline
        # points are spaced further apart (the migration itself is unchanged)
        "xxl": dict(batch=0, parts=128, wl=64, ev=4, m=200, pool=30, period=5,
                    cap=0.001, seed_frac=0.02),
    }[scale]
    if presets["batch"] == 0:                       # ~120 timeline steps
        presets["batch"] = max(1, int(round(n_docs / 120.0)))
    return EngineConfig(
        seed=seed,
        seed_fraction=presets["seed_frac"],
        batch_size=presets["batch"],
        n_partitions=presets["parts"],
        workload_per_step=presets["wl"],
        eval_every_steps=presets["ev"],
        adapter=AdapterConfig(kinds=("procrustes", "ridge", "mlp"), primary="auto",
                              mlp_epochs=60, train_reverse=True),
        scheduler=SchedulerConfig(),
        certifier=CertifierConfig(sample_queries=presets["m"], pool_extra=presets["pool"],
                                  period_steps=presets["period"],
                                  max_round_cost_frac=presets["cap"],
                                  target_recall=0.90),
        serve=ServeConfig(k=10, overfetch=4, ef_search=128),
    )


# -------------------------------------------------------------------- outputs
def _git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=5).stdout.strip() or "n/a"
    except Exception:
        return "n/a"


def write_results(rows: list[dict], out_path: Path, meta: dict) -> Path:
    """Write CSV + .meta.json sidecar. Every row is stamped with the meta's
    scalar identity columns (dataset, pair, scale, seed, n_docs)."""
    import pandas as pd

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for col in ("dataset", "pair", "scale", "seed", "n_docs"):
        if col in meta and col not in df.columns:
            df[col] = meta[col]
    df.to_csv(out_path, index=False)
    sidecar = dict(meta)
    sidecar.update({
        "argv": sys.argv,
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "git_rev": _git_rev(),
        "n_rows": len(df),
    })
    with open(out_path.with_suffix(out_path.suffix + ".meta.json"), "w") as f:
        json.dump(sidecar, f, indent=2, default=str)
    print(f"wrote {len(df)} rows -> {out_path}")
    return out_path


def base_meta(args, data: MigrationData | None = None, **extra) -> dict:
    meta = {"dataset": resolve_dataset(args), "pair": args.pair, "scale": args.scale,
            "seed": args.seed}
    if data is not None:
        meta.update({"n_docs": data.n_docs, "n_queries": len(data.q_ids),
                     "d_old": data.old.shape[1], "d_new": data.new.shape[1],
                     "old_model": data.meta.get("old_model"),
                     "new_model": data.meta.get("new_model")})
    meta.update(extra)
    return meta


def default_out(args, exp_name: str, suffix: str = "") -> Path:
    ds = resolve_dataset(args)
    tag = f"{exp_name}_{ds}_{args.pair}_{args.scale}{suffix}"
    return RESULTS_DIR / f"{tag}.csv"


# ------------------------------------------------------------ derived metrics
def budget_to_reach(timeline_rows: list[dict], metric: str, target_value: float
                    ) -> float:
    """Smallest budget_frac at which `metric` reaches target_value (nan if never)."""
    for r in timeline_rows:
        v = r.get(metric)
        if v is not None and not (isinstance(v, float) and np.isnan(v)) and v >= target_value:
            return float(r["budget_frac"])
    return float("nan")
