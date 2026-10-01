"""Prepare a LARGE BEIR corpus (10^5-10^7 docs) as a memory-mapped bundle.

Unlike prepare_data.py (which compresses everything into a single .npz and is
fine to ~100k docs), this writes a *directory bundle* of raw .npy files that the
engine can memory-map, and embeds the corpus in resumable chunks so a multi-hour
encode survives interruption:

    data/prepared/<dataset>_<pair>_<tag>/
        old.npy  new.npy  q_old.npy  q_new.npy   (float32, row-major)
        doc_ids.npy  q_ids.npy  qrels.json  meta.json
        .progress_<role>.json                     (resume checkpoints)

Usage:
    python experiments/prepare_large.py --dataset msmarco --pair minilm-e5 \
        --tag xxl [--max-docs N] [--batch-size 512]

The bundle is prepared ONCE at full size; the scale ladder in the experiments is
built by deterministically sub-sampling this bundle (see common.load_bundle),
so scale is varied with dataset, pair and encoder held exactly fixed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from common import MODEL_PAIRS, PREPARED_DIR, DATA_DIR

from metamorph.data import DEFAULT_SPLIT, download_beir, qrels_to_indices
from metamorph.embed import PREFIXES, pick_device


# --------------------------------------------------------------------------
def stream_corpus(path: Path):
    """Yield (doc_id, text) from a BEIR corpus.jsonl without holding the file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            title = (r.get("title") or "").strip()
            body = (r.get("text") or "").strip()
            yield str(r["_id"]), (title + " " + body).strip() if title else body


def load_queries_qrels(root: Path, split: str):
    qrels: dict[str, dict[str, float]] = {}
    with open(root / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        f.readline()
        for line in f:
            qid, did, score = line.rstrip("\n").split("\t")
            if float(score) > 0:
                qrels.setdefault(qid, {})[did] = float(score)
    queries: dict[str, str] = {}
    with open(root / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if str(r["_id"]) in qrels:
                queries[str(r["_id"])] = r["text"]
    qrels = {q: rel for q, rel in qrels.items() if q in queries}
    return queries, qrels


def encode_into(model_name: str, texts: list[str], kind: str, out_path: Path,
                dim: int, batch_size: int, chunk: int, progress_path: Path,
                device: str) -> np.ndarray:
    """Encode `texts` into a memmapped float32 (n, dim) .npy, resumably."""
    from sentence_transformers import SentenceTransformer

    n = len(texts)
    done = 0
    if progress_path.exists() and out_path.exists():
        done = int(json.loads(progress_path.read_text())["rows_done"])
        done = min(done, n)
    if done >= n:
        print(f"  [{out_path.name}] already complete ({n} rows)")
        return np.load(out_path, mmap_mode="r")

    if not out_path.exists():
        # allocate the full array on disk up front
        arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32,
                                        shape=(n, dim))
    else:
        arr = np.lib.format.open_memmap(out_path, mode="r+")
        if arr.shape != (n, dim):
            arr = np.lib.format.open_memmap(out_path, mode="w+",
                                            dtype=np.float32, shape=(n, dim))
            done = 0

    qp, pp = PREFIXES.get(model_name, ("", ""))
    prefix = qp if kind == "query" else pp
    model = SentenceTransformer(model_name, device=device)
    t0, t_start_rows = time.time(), done
    while done < n:
        hi = min(n, done + chunk)
        emb = model.encode([prefix + t for t in texts[done:hi]],
                           batch_size=batch_size, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False)
        arr[done:hi] = np.asarray(emb, dtype=np.float32)
        done = hi
        arr.flush()
        progress_path.write_text(json.dumps({"rows_done": done, "n": n}))
        el = time.time() - t0
        rate = (done - t_start_rows) / max(el, 1e-9)
        eta = (n - done) / max(rate, 1e-9)
        print(f"  [{out_path.name}] {done}/{n} ({100*done/n:.1f}%) "
              f"{rate:.0f} docs/s  eta {eta/60:.1f} min", flush=True)
    del model
    return np.load(out_path, mmap_mode="r")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pair", default="minilm-e5", choices=sorted(MODEL_PAIRS))
    ap.add_argument("--tag", default="xxl", help="bundle name suffix")
    ap.add_argument("--split", default=None)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--max-queries", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=200_000,
                    help="rows per checkpointed encode chunk")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    old_model, new_model = MODEL_PAIRS[args.pair]
    split = args.split or DEFAULT_SPLIT.get(args.dataset, "test")
    out = PREPARED_DIR / f"{args.dataset}_{args.pair}_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    t0 = time.time()

    root = download_beir(args.dataset, DATA_DIR)
    print(f"corpus: {root}  split={split}  device={device}")

    queries, qrels = load_queries_qrels(root, split)
    rng = np.random.default_rng(args.seed)
    q_ids = sorted(queries)
    if len(q_ids) > args.max_queries:
        keep = rng.choice(len(q_ids), size=args.max_queries, replace=False)
        q_ids = sorted(q_ids[i] for i in keep)
    qrels = {q: qrels[q] for q in q_ids}
    positives = {d for rel in qrels.values() for d in rel}
    print(f"  {len(q_ids)} judged queries, {len(positives)} judged docs")

    # ---- stream the corpus once, keeping all positives + a random fill ----
    doc_ids: list[str] = []
    doc_texts: list[str] = []
    ids_path, texts_cache = out / "doc_ids.npy", out / "doc_texts.jsonl"
    if texts_cache.exists() and ids_path.exists():
        doc_ids = [str(x) for x in np.load(ids_path, allow_pickle=False)]
        with open(texts_cache, encoding="utf-8") as f:
            doc_texts = [json.loads(line) for line in f]
        print(f"  reusing cached corpus subset: {len(doc_ids)} docs")
    else:
        cap = args.max_docs
        n_seen = 0
        for did, text in stream_corpus(root / "corpus.jsonl"):
            n_seen += 1
            if cap is None or did in positives:
                doc_ids.append(did)
                doc_texts.append(text)
            elif len(doc_ids) < cap:
                doc_ids.append(did)
                doc_texts.append(text)
            if n_seen % 1_000_000 == 0:
                print(f"    scanned {n_seen:,} corpus lines, kept {len(doc_ids):,}")
        print(f"  corpus lines: {n_seen:,}; kept {len(doc_ids):,}")
        np.save(ids_path, np.array(doc_ids))
        with open(texts_cache, "w", encoding="utf-8") as f:
            for t in doc_texts:
                f.write(json.dumps(t) + "\n")

    kept = set(doc_ids)
    qrels = {q: {d: g for d, g in rel.items() if d in kept} for q, rel in qrels.items()}
    qrels = {q: rel for q, rel in qrels.items() if rel}
    q_ids = [q for q in q_ids if q in qrels]
    q_texts = [queries[q] for q in q_ids]
    print(f"  final: {len(doc_ids):,} docs, {len(q_ids)} queries")

    # write the query side up front: it is cheap, and it lets a long doc encode
    # be truncated into a usable bundle while it is still running
    np.save(out / "q_ids.npy", np.array(q_ids))
    qrels_idx_early = qrels_to_indices(qrels, doc_ids)
    with open(out / "qrels.json", "w") as f:
        json.dump({q: {str(d): g for d, g in rel.items()}
                   for q, rel in qrels_idx_early.items()}, f)

    dims = {}
    for role, model in (("old", old_model), ("new", new_model)):
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(model, device="cpu")
        dims[role] = int(m.get_sentence_embedding_dimension())
        del m

    for role, model in (("old", old_model), ("new", new_model)):
        print(f"encoding docs with {role.upper()} = {model} ...", flush=True)
        encode_into(model, doc_texts, "passage", out / f"{role}.npy", dims[role],
                    args.batch_size, args.chunk, out / f".progress_{role}.json",
                    device)
        print(f"encoding queries with {role.upper()} ...", flush=True)
        encode_into(model, q_texts, "query", out / f"q_{role}.npy", dims[role],
                    args.batch_size, max(args.chunk, len(q_texts)),
                    out / f".progress_q_{role}.json", device)

    np.save(out / "q_ids.npy", np.array(q_ids))
    qrels_idx = qrels_to_indices(qrels, doc_ids)
    with open(out / "qrels.json", "w") as f:
        json.dump({q: {str(d): g for d, g in rel.items()} for q, rel in qrels_idx.items()}, f)
    meta = {
        "dataset": args.dataset, "pair": args.pair, "scale": args.tag,
        "split": split, "old_model": old_model, "new_model": new_model,
        "n_docs": len(doc_ids), "n_queries": len(q_ids),
        "d_old": dims["old"], "d_new": dims["new"], "seed": args.seed,
        "format": "bundle-v1",
        "prepared_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prepare_seconds": round(time.time() - t0, 1),
        "device": device,
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"prepared -> {out}  ({(time.time()-t0)/60:.1f} min)")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
