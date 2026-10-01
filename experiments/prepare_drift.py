"""Prepare embeddings for the faithful Drift-Adapter reproduction (exp10).

Drift-Adapter (Vejendla, EMNLP 2025, Sec. 4) evaluates document-to-document
search on three MTEB text datasets (AG-News, DBpedia-14, Emotion) with
all-MiniLM-L6-v2 -> all-mpnet-base-v2: the database is 1M items sampled from
the training split, queries are 10,000 test-split items. A 1M database cannot
be drawn from these training splits without duplication (120,000 / 560,000 /
15,956 items; MTEB's own DBpediaClassification split has 2,048), so we use the
ENTIRE original training split as the database and up to 10,000 test items as
queries (all of them when the test split is smaller). This is the only
deviation from the stated protocol and it is recorded in meta.json.

The same script embeds, with the Drift-Adapter destination model
(all-mpnet-base-v2), the two Metamorph retrieval corpora used for the factor
analysis: full fiqa (57,638 docs, 648 test queries, same order as
data/prepared/fiqa_minilm-e5_full.npz) and the 1M-passage MS MARCO ladder rung
(the exact subsample exp9 uses: load_bundle(n_docs=1_000_000, seed=42)).

Layout: data/prepared/drift/<dataset>/{db,q}_<tag>.npy plus meta.json.

  python experiments/prepare_drift.py --datasets emotion agnews dbpedia14 \
      --models minilm mpnet e5s gtes
  python experiments/prepare_drift.py --datasets fiqa msmarco1m --models mpnet
"""
from __future__ import annotations

import argparse
import io
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from common import DATA_DIR, PREPARED_DIR
from prepare_large import encode_into

from metamorph.embed import pick_device

DRIFT_DIR = PREPARED_DIR / "drift"

MODELS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", 384),
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", 768),
    "e5s": ("intfloat/e5-small-v2", 384),
    "gtes": ("thenlper/gte-small", 384),
}

HF = "https://huggingface.co/api/datasets"
CLASSIFICATION = {
    # name: (hf dataset, config, text builder)
    "agnews": ("fancyzhx/ag_news", "default", lambda r: str(r["text"])),
    "dbpedia14": ("fancyzhx/dbpedia_14", "dbpedia_14",
                  lambda r: (str(r["title"]).strip() + " " + str(r["content"]).strip()).strip()),
    "emotion": ("mteb/emotion", "default", lambda r: str(r["text"])),
}
MAX_QUERIES = 10_000


def _read_parquet_split(repo: str, config: str, split: str):
    import pandas as pd

    listing = json.loads(urllib.request.urlopen(f"{HF}/{repo}/parquet").read())
    urls = listing[config][split]
    frames = []
    for u in urls:
        frames.append(pd.read_parquet(io.BytesIO(urllib.request.urlopen(u).read())))
    return pd.concat(frames, ignore_index=True)


def save_labels(name: str, seed: int) -> None:
    """Class labels, aligned with db_*.npy (train order) and q_*.npy (q_index)."""
    repo, config, _ = CLASSIFICATION[name]
    train = _read_parquet_split(repo, config, "train")
    test = _read_parquet_split(repo, config, "test")
    q_idx = np.load(DRIFT_DIR / name / "q_index.npy")
    np.save(DRIFT_DIR / name / "db_labels.npy", train["label"].to_numpy().astype(np.int64))
    np.save(DRIFT_DIR / name / "q_labels.npy", test["label"].to_numpy()[q_idx].astype(np.int64))
    print(f"[{name}] labels: {train['label'].nunique()} classes", flush=True)


def load_classification(name: str, seed: int):
    repo, config, text_of = CLASSIFICATION[name]
    train = _read_parquet_split(repo, config, "train")
    test = _read_parquet_split(repo, config, "test")
    db_texts = [text_of(r) for r in train.to_dict("records")]
    q_all = [text_of(r) for r in test.to_dict("records")]
    q_idx = np.arange(len(q_all))
    if len(q_all) > MAX_QUERIES:
        q_idx = np.sort(np.random.default_rng(seed).choice(len(q_all), MAX_QUERIES,
                                                           replace=False))
    q_texts = [q_all[i] for i in q_idx]
    meta = {"source": repo, "config": config, "db_split": "train", "q_split": "test",
            "n_train_split": len(train), "n_test_split": len(test),
            "n_db": len(db_texts), "n_queries": len(q_texts),
            "task": "symmetric document-to-document search (Drift-Adapter Sec. 4)",
            "deviation": ("database = entire training split (paper states 1M items, "
                          "which exceeds the split size); queries = up to 10,000 "
                          "test items, sampled with the given seed when larger")}
    return db_texts, q_texts, q_idx, meta


def load_fiqa():
    z = np.load(PREPARED_DIR / "fiqa_minilm-e5_full.npz", allow_pickle=False)
    doc_ids = [str(d) for d in z["doc_ids"]]
    q_ids = [str(q) for q in z["q_ids"]]
    root = DATA_DIR / "beir" / "fiqa"
    docs, queries = {}, {}
    with open(root / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            t, b = (r.get("title") or "").strip(), (r.get("text") or "").strip()
            docs[str(r["_id"])] = (t + " " + b).strip() if t else b
    with open(root / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            queries[str(r["_id"])] = r["text"]
    meta = {"source": "BEIR fiqa (order of fiqa_minilm-e5_full.npz)",
            "n_db": len(doc_ids), "n_queries": len(q_ids),
            "task": "asymmetric passage retrieval"}
    return [docs[d] for d in doc_ids], [queries[q] for q in q_ids], np.arange(len(q_ids)), meta


def msmarco_rung_indices(n_docs: int = 1_000_000, seed: int = 42):
    """Exactly the selection of common.load_bundle(n_docs, seed) on the xxl bundle."""
    root = PREPARED_DIR / "msmarco_minilm-e5_xxl"
    qrels_raw = json.loads((root / "qrels.json").read_text())
    qrels = {q: {int(d): float(g) for d, g in rel.items()} for q, rel in qrels_raw.items()}
    N = np.load(root / "old.npy", mmap_mode="r").shape[0]
    rng = np.random.default_rng(seed)
    positives = np.unique(np.concatenate(
        [np.fromiter(rel.keys(), dtype=np.int64) for rel in qrels.values()]))
    fill = max(0, n_docs - positives.size)
    rest = np.setdiff1d(np.arange(N, dtype=np.int64), positives, assume_unique=False)
    take = rng.choice(rest, size=min(fill, rest.size), replace=False)
    keep = np.sort(np.concatenate([positives, take]))
    keep_set = set(keep.tolist())
    q_ids = [str(x) for x in np.load(root / "q_ids.npy", allow_pickle=False)]
    keep_q = [i for i, q in enumerate(q_ids)
              if any(d in keep_set for d in qrels.get(q, {}))]
    return keep, np.asarray(keep_q, dtype=np.int64), q_ids


def load_msmarco1m():
    root = PREPARED_DIR / "msmarco_minilm-e5_xxl"
    keep, keep_q, q_ids = msmarco_rung_indices()
    want = np.zeros(int(keep.max()) + 1, dtype=bool)
    want[keep] = True
    texts = []
    with open(root / "doc_texts.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > keep[-1]:
                break
            if want[i]:
                texts.append(json.loads(line))
    assert len(texts) == keep.size, (len(texts), keep.size)
    queries = {}
    with open(DATA_DIR / "beir" / "msmarco" / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            queries[str(r["_id"])] = r["text"]
    q_texts = [queries[q_ids[i]] for i in keep_q]
    meta = {"source": "MS MARCO passage, 1M ladder rung (load_bundle n_docs=1e6 seed=42)",
            "n_db": int(keep.size), "n_queries": len(q_texts),
            "task": "asymmetric passage retrieval"}
    return texts, q_texts, keep_q, meta, keep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", required=True,
                    choices=sorted(CLASSIFICATION) + ["fiqa", "msmarco1m"])
    ap.add_argument("--models", nargs="+", default=["minilm", "mpnet"], choices=sorted(MODELS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=100_000)
    ap.add_argument("--labels-only", action="store_true",
                    help="only (re)write class labels for already-embedded datasets")
    args = ap.parse_args()
    device = pick_device()
    if args.labels_only:
        for ds in args.datasets:
            save_labels(ds, args.seed)
        return

    for ds in args.datasets:
        out = DRIFT_DIR / ds
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        extra = None
        if ds in CLASSIFICATION:
            db_texts, q_texts, q_idx, meta = load_classification(ds, args.seed)
            symmetric = True
        elif ds == "fiqa":
            db_texts, q_texts, q_idx, meta = load_fiqa()
            symmetric = False
        else:
            db_texts, q_texts, q_idx, meta, extra = load_msmarco1m()
            symmetric = False
        np.save(out / "q_index.npy", np.asarray(q_idx, dtype=np.int64))
        if extra is not None:
            np.save(out / "db_index.npy", extra)
        print(f"[{ds}] db={len(db_texts)} q={len(q_texts)} (loaded in {time.time()-t0:.0f}s)",
              flush=True)
        for tag in args.models:
            name, dim = MODELS[tag]
            # E5 on a symmetric task uses the "query: " prefix on both sides
            # (E5 model card); asymmetric retrieval uses query/passage.
            db_kind = "query" if symmetric else "passage"
            encode_into(name, db_texts, db_kind, out / f"db_{tag}.npy", dim,
                        args.batch_size, args.chunk, out / f".progress_db_{tag}.json", device)
            encode_into(name, q_texts, "query", out / f"q_{tag}.npy", dim,
                        args.batch_size, max(args.chunk, len(q_texts)),
                        out / f".progress_q_{tag}.json", device)
        meta.update({"dataset": ds, "seed": args.seed, "symmetric": symmetric,
                     "models": {t: MODELS[t][0] for t in args.models},
                     "prepared_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        old = json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else {}
        old_models = old.get("models", {})
        old_models.update(meta["models"])
        meta["models"] = old_models
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[{ds}] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
