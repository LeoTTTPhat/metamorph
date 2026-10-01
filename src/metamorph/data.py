"""BEIR dataset download and parsing (no `beir` dependency — direct zip parse).

Datasets are cached under data/beir/<name>/ as corpus.jsonl, queries.jsonl,
qrels/*.tsv. Subsetting keeps all qrel-positive docs plus a random fill to the
requested size (so relevance judgments stay meaningful at reduced scale).
"""
from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

BEIR_BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
KNOWN = {"scifact", "nfcorpus", "fiqa", "quora", "arguana", "scidocs",
         "nq", "msmarco", "hotpotqa", "dbpedia-entity", "fever", "climate-fever"}

# BEIR datasets whose judged query split is not called "test"
DEFAULT_SPLIT = {"msmarco": "dev"}


def download_beir(name: str, data_dir: str | Path) -> Path:
    if name not in KNOWN:
        raise ValueError(f"unknown BEIR dataset {name}; known: {sorted(KNOWN)}")
    root = Path(data_dir) / "beir"
    out = root / name
    if (out / "corpus.jsonl").exists():
        return out
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / f"{name}.zip"
    if not zip_path.exists():
        url = f"{BEIR_BASE}/{name}.zip"
        print(f"downloading {url} ...")
        urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(root)
    if not (out / "corpus.jsonl").exists():
        raise RuntimeError(f"extraction of {name} did not produce corpus.jsonl")
    return out


def load_beir(name: str, data_dir: str | Path, split: str = "test",
              max_docs: int | None = None, seed: int = 42):
    """Returns (doc_ids, doc_texts, query_ids, query_texts, qrels).

    qrels: {qid: {doc_id_str: gain}} using string doc ids (mapped to indices later).
    """
    path = download_beir(name, data_dir)
    docs: dict[str, str] = {}
    with open(path / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            text = (r.get("title") or "").strip()
            body = (r.get("text") or "").strip()
            docs[str(r["_id"])] = (text + " " + body).strip() if text else body
    queries: dict[str, str] = {}
    with open(path / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            queries[str(r["_id"])] = r["text"]
    qrels: dict[str, dict[str, float]] = {}
    qrels_file = path / "qrels" / f"{split}.tsv"
    with open(qrels_file, encoding="utf-8") as f:
        header = f.readline()  # query-id  corpus-id  score
        for line in f:
            qid, did, score = line.rstrip("\n").split("\t")
            if float(score) > 0:
                qrels.setdefault(qid, {})[did] = float(score)
    # keep only queries with qrels in this split
    queries = {q: t for q, t in queries.items() if q in qrels}

    doc_ids = sorted(docs)
    if max_docs is not None and len(doc_ids) > max_docs:
        rng = np.random.default_rng(seed)
        positives = sorted({d for rel in qrels.values() for d in rel if d in docs})
        rest = sorted(set(doc_ids) - set(positives))
        fill = max(0, max_docs - len(positives))
        keep = set(positives) | set(rng.choice(rest, size=min(fill, len(rest)),
                                               replace=False).tolist())
        doc_ids = sorted(keep)
    doc_texts = [docs[d] for d in doc_ids]
    kept = set(doc_ids)
    qrels = {q: {d: g for d, g in rel.items() if d in kept} for q, rel in qrels.items()}
    qrels = {q: rel for q, rel in qrels.items() if rel}
    queries = {q: t for q, t in queries.items() if q in qrels}
    q_ids = sorted(queries)
    return doc_ids, doc_texts, q_ids, [queries[q] for q in q_ids], qrels


def qrels_to_indices(qrels: dict[str, dict[str, float]],
                     doc_ids: list[str]) -> dict[str, dict[int, float]]:
    pos = {d: i for i, d in enumerate(doc_ids)}
    return {q: {pos[d]: g for d, g in rel.items() if d in pos}
            for q, rel in qrels.items()}
