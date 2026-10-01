"""Build a valid prefix bundle from a partially-encoded large bundle.

`prepare_large.py` encodes into memory-mapped arrays in file order and
checkpoints every chunk, so at any moment the first `rows_done` rows of both
`old.npy` and `new.npy` are complete.  This script materialises those rows as a
self-contained bundle: the prefix of the corpus, with qrels restricted to the
documents it contains and queries with no surviving judgement dropped.

    python experiments/truncate_bundle.py --src msmarco_minilm-e5_xxl \
        --dst msmarco_minilm-e5_xl --rows 4000000

Use this to stop a multi-hour encode early without losing the work, or to pin a
scale-ladder rung to an exact corpus size.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from common import PREPARED_DIR


def _rebuild_queries(src: Path) -> None:
    """Recreate q_ids / qrels / q_old / q_new for a bundle whose encode is
    still in flight, using exactly the selection prepare_large.py would make."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import DATA_DIR, MODEL_PAIRS
    from prepare_large import load_queries_qrels
    from metamorph.data import DEFAULT_SPLIT, download_beir
    from metamorph.embed import embed_texts, pick_device
    from metamorph.data import qrels_to_indices

    meta = json.loads((src / "meta.json").read_text()) if (src / "meta.json").exists() else {}
    name = src.name                       # <dataset>_<pair>_<tag>
    dataset = meta.get("dataset") or name.split("_")[0]
    pair = meta.get("pair") or name.split("_")[1]
    seed = int(meta.get("seed", 42))
    max_queries = int(meta.get("max_queries", 8000))
    split = meta.get("split") or DEFAULT_SPLIT.get(dataset, "test")
    root = download_beir(dataset, DATA_DIR)
    queries, qrels = load_queries_qrels(root, split)
    rng = np.random.default_rng(seed)
    q_ids = sorted(queries)
    if len(q_ids) > max_queries:
        keep = rng.choice(len(q_ids), size=max_queries, replace=False)
        q_ids = sorted(q_ids[i] for i in keep)
    qrels = {q: qrels[q] for q in q_ids}
    doc_ids = [str(x) for x in np.load(src / "doc_ids.npy", allow_pickle=False)]
    qrels_idx = qrels_to_indices(qrels, doc_ids)
    old_model, new_model = MODEL_PAIRS[pair]
    texts = [queries[q] for q in q_ids]
    dev = pick_device()
    for role, model in (("q_old", old_model), ("q_new", new_model)):
        if (src / f"{role}.npy").exists():
            continue
        emb = embed_texts(model, texts, "query", cache_dir=DATA_DIR / "emb_cache",
                          batch_size=128, device=dev)
        np.save(src / f"{role}.npy", np.asarray(emb, dtype=np.float32))
        print(f"  encoded {role}.npy for {len(texts)} queries")
    np.save(src / "q_ids.npy", np.array(q_ids))
    (src / "qrels.json").write_text(json.dumps(
        {q: {str(d): g for d, g in rel.items()} for q, rel in qrels_idx.items()}))
    print(f"  rebuilt query side: {len(q_ids)} queries")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="bundle directory name")
    ap.add_argument("--dst", required=True, help="output bundle directory name")
    ap.add_argument("--rows", type=int, default=None,
                    help="prefix length (default: min of the two progress files)")
    ap.add_argument("--copy-texts", action="store_true",
                    help="also copy the prefix of doc_texts.jsonl (needed by exp7)")
    args = ap.parse_args()

    src, dst = PREPARED_DIR / args.src, PREPARED_DIR / args.dst
    if not src.exists():
        raise SystemExit(f"no such bundle: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    done = []
    for role in ("old", "new"):
        p = src / f".progress_{role}.json"
        if p.exists():
            done.append(int(json.loads(p.read_text())["rows_done"]))
        elif (src / f"{role}.npy").exists():
            done.append(int(np.load(src / f"{role}.npy", mmap_mode="r").shape[0]))
    rows = args.rows or (min(done) if done else 0)
    if rows <= 0:
        raise SystemExit("nothing encoded yet")
    print(f"prefix rows: {rows:,} (available: {done})")
    if any(rows > d for d in done):
        raise SystemExit(f"requested {rows} rows but only {min(done)} are encoded")

    doc_ids = [str(x) for x in np.load(src / "doc_ids.npy", allow_pickle=False)][:rows]
    for role in ("old", "new"):
        a = np.load(src / f"{role}.npy", mmap_mode="r")[:rows]
        out = np.lib.format.open_memmap(dst / f"{role}.npy", mode="w+",
                                        dtype=np.float32, shape=(rows, a.shape[1]))
        step = 500_000
        for i in range(0, rows, step):
            out[i:i + step] = a[i:i + step]
        out.flush()
        del out
        print(f"  wrote {role}.npy {rows:,} x {a.shape[1]}")
        norms = np.linalg.norm(np.load(dst / f"{role}.npy", mmap_mode="r")[-1024:], axis=1)
        if not np.all(norms > 0.5):
            raise SystemExit(f"{role}.npy tail contains unencoded (zero) rows")

    if not (src / "q_ids.npy").exists():
        # The source encode is still running: prepare_large writes queries only
        # at the end. Reconstruct the same query selection deterministically from
        # the BEIR files and encode it here (a few thousand queries, seconds).
        _rebuild_queries(src)
    q_ids = [str(x) for x in np.load(src / "q_ids.npy", allow_pickle=False)]
    qrels = json.loads((src / "qrels.json").read_text())
    keep = {int(d) for d in range(rows)}
    qrels = {q: {d: g for d, g in rel.items() if int(d) in keep} for q, rel in qrels.items()}
    qrels = {q: rel for q, rel in qrels.items() if rel}
    sel = [i for i, q in enumerate(q_ids) if q in qrels]
    for role in ("q_old", "q_new"):
        a = np.load(src / f"{role}.npy", mmap_mode="r")
        np.save(dst / f"{role}.npy", np.ascontiguousarray(a[sel]))
    np.save(dst / "doc_ids.npy", np.array(doc_ids))
    np.save(dst / "q_ids.npy", np.array([q_ids[i] for i in sel]))
    (dst / "qrels.json").write_text(json.dumps(qrels))

    meta = json.loads((src / "meta.json").read_text()) if (src / "meta.json").exists() else {}
    meta.update({"n_docs": rows, "n_queries": len(sel), "format": "bundle-v1",
                 "truncated_from": args.src, "scale": args.dst.split("_")[-1]})
    meta.setdefault("d_old", int(np.load(dst / "old.npy", mmap_mode="r").shape[1]))
    meta.setdefault("d_new", int(np.load(dst / "new.npy", mmap_mode="r").shape[1]))
    (dst / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"queries kept: {len(sel)} / {len(q_ids)}")

    if args.copy_texts and (src / "doc_texts.jsonl").exists():
        with open(src / "doc_texts.jsonl", encoding="utf-8") as fi, \
             open(dst / "doc_texts.jsonl", "w", encoding="utf-8") as fo:
            for i, line in enumerate(fi):
                if i >= rows:
                    break
                fo.write(line)
        print("  copied doc_texts.jsonl prefix")
    print(f"prepared -> {dst}")


if __name__ == "__main__":
    main()
