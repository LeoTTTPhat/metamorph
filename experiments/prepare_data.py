"""Download a BEIR dataset, embed it with an OLD/NEW model pair, cache as .npz.

Usage:
  python experiments/prepare_data.py --dataset scifact --pair minilm-e5 --scale smoke

Downloads are cached in data/beir/, embeddings in data/emb_cache/, and the
final prepared bundle in data/prepared/<dataset>_<pair>_<scale>.npz.

Budget notes: scifact ~3MB, nfcorpus ~2.5MB, fiqa ~18MB, quora ~35MB zips;
models are 45-130MB each (HF cache). Embedding runs on MPS if available.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from common import (MODEL_PAIRS, PREPARED_DIR, SCALE_MAX_DOCS, DATA_DIR,
                    add_common_args, prepared_path, resolve_dataset)

from metamorph.data import load_beir, qrels_to_indices
from metamorph.embed import embed_texts, pick_device


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--max-docs", type=int, default=None,
                    help="override the scale's corpus cap")
    ap.add_argument("--max-queries", type=int, default=2000,
                    help="cap on queries (BEIR test split is usually smaller)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = ap.parse_args()

    dataset = resolve_dataset(args)
    old_model, new_model = MODEL_PAIRS[args.pair]
    max_docs = args.max_docs if args.max_docs is not None else SCALE_MAX_DOCS[args.scale]
    out = prepared_path(dataset, args.pair, args.scale)
    if out.exists() and not args.force:
        print(f"already prepared: {out}")
        return

    t0 = time.time()
    print(f"loading BEIR {dataset} (max_docs={max_docs}) ...")
    doc_ids, doc_texts, q_ids, q_texts, qrels_str = load_beir(
        dataset, DATA_DIR, split="test", max_docs=max_docs, seed=args.seed)
    if len(q_ids) > args.max_queries:
        keep = set(np.random.default_rng(args.seed).choice(
            len(q_ids), size=args.max_queries, replace=False).tolist())
        q_ids = [q for i, q in enumerate(q_ids) if i in keep]
        q_texts = [t for i, t in enumerate(q_texts) if i in keep]
        qrels_str = {q: rel for q, rel in qrels_str.items() if q in set(q_ids)}
    print(f"  {len(doc_ids)} docs, {len(q_ids)} queries "
          f"(device={pick_device()})")

    cache = DATA_DIR / "emb_cache"
    embs = {}
    for role, model in (("old", old_model), ("new", new_model)):
        print(f"embedding docs with {role.upper()} model {model} ...")
        embs[role] = embed_texts(model, doc_texts, "passage", cache_dir=cache,
                                 batch_size=args.batch_size)
        print(f"embedding queries with {role.upper()} model {model} ...")
        embs[f"q_{role}"] = embed_texts(model, q_texts, "query", cache_dir=cache,
                                        batch_size=args.batch_size)

    qrels_idx = qrels_to_indices(qrels_str, doc_ids)
    meta = {
        "dataset": dataset, "pair": args.pair, "scale": args.scale,
        "old_model": old_model, "new_model": new_model,
        "n_docs": len(doc_ids), "n_queries": len(q_ids),
        "d_old": int(embs["old"].shape[1]), "d_new": int(embs["new"].shape[1]),
        "max_docs_cap": max_docs, "seed": args.seed,
        "prepared_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prepare_seconds": round(time.time() - t0, 1),
    }
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        doc_ids=np.array(doc_ids), q_ids=np.array(q_ids),
        old=embs["old"], new=embs["new"],
        q_old=embs["q_old"], q_new=embs["q_new"],
        qrels_json=json.dumps({q: {str(d): g for d, g in rel.items()}
                               for q, rel in qrels_idx.items()}),
        meta_json=json.dumps(meta))
    print(f"prepared -> {out}  ({time.time() - t0:.1f}s)")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
