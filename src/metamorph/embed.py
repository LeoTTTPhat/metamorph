"""Sentence-transformers model wrappers with per-model prefix handling and caching."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .utils import l2_normalize

# per-model text prefixes (E5 family requires them; others use none)
PREFIXES: dict[str, tuple[str, str]] = {  # model -> (query_prefix, passage_prefix)
    "intfloat/e5-small-v2": ("query: ", "passage: "),
    "intfloat/e5-base-v2": ("query: ", "passage: "),
    "intfloat/e5-large-v2": ("query: ", "passage: "),
}


def _cache_key(model_name: str, texts_fingerprint: str, kind: str) -> str:
    h = hashlib.sha1(f"{model_name}|{kind}|{texts_fingerprint}".encode()).hexdigest()[:16]
    return f"{model_name.replace('/', '__')}_{kind}_{h}.npy"


def fingerprint_texts(texts: list[str]) -> str:
    h = hashlib.sha1()
    h.update(str(len(texts)).encode())
    for t in texts[:50]:
        h.update(t[:200].encode("utf-8", "ignore"))
    for t in texts[-50:]:
        h.update(t[:200].encode("utf-8", "ignore"))
    return h.hexdigest()


def pick_device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def embed_texts(model_name: str, texts: list[str], kind: str,
                cache_dir: str | Path | None = None, batch_size: int = 64,
                device: str | None = None) -> np.ndarray:
    """Embed texts ('query' or 'passage'), with disk caching."""
    assert kind in ("query", "passage")
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / _cache_key(model_name, fingerprint_texts(texts), kind)
        if cache_path.exists():
            return np.load(cache_path)
    from sentence_transformers import SentenceTransformer

    qp, pp = PREFIXES.get(model_name, ("", ""))
    prefix = qp if kind == "query" else pp
    inputs = [prefix + t for t in texts]
    model = SentenceTransformer(model_name, device=device or pick_device())
    emb = model.encode(inputs, batch_size=batch_size, convert_to_numpy=True,
                       show_progress_bar=len(inputs) > 500, normalize_embeddings=True)
    emb = l2_normalize(np.asarray(emb, dtype=np.float32))
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, emb)
    return emb
