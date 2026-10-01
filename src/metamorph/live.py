"""Live-encoder store: budget units become real encoder forward passes.

Everywhere else in this codebase the new-model embeddings are precomputed into
an oracle that the engine may only read by paying the budget meter.  That gives
exact cost accounting and lets a full migration timeline replay in minutes, but
it means the meter is an *abstraction*: nothing in the loop measures what a
budget unit actually costs.

:class:`LiveEmbeddingStore` closes that gap.  It keeps the same interface and
the same accounting, but ``reveal`` genuinely runs the new encoder over the
documents' text, so a migration's wall-clock includes the encoder work it is
trying to save.  The precomputed oracle is retained for *evaluation only* and
is used to assert that the live encoder reproduces it, so a live run and a
replayed run are the same experiment.
"""
from __future__ import annotations

import time

import numpy as np

from .store import EmbeddingStore
from .utils import CostMeter, l2_normalize


class LiveEmbeddingStore(EmbeddingStore):
    """EmbeddingStore whose reveals invoke the real encoder.

    Parameters
    ----------
    old, new_oracle : the usual embedding matrices (the oracle is evaluation-only)
    texts           : document texts, aligned with the embedding rows
    model_name      : the NEW model to run
    """

    def __init__(self, old: np.ndarray, new_oracle: np.ndarray, texts: list[str],
                 model_name: str, meter: CostMeter | None = None,
                 batch_size: int = 256, device: str | None = None,
                 verify_against_oracle: int = 64):
        super().__init__(old, new_oracle, meter)
        from sentence_transformers import SentenceTransformer

        from .embed import PREFIXES, pick_device

        self.texts = texts
        self.model_name = model_name
        self.batch_size = batch_size
        self._prefix = PREFIXES.get(model_name, ("", ""))[1]
        self._device = device or pick_device()
        self._model = SentenceTransformer(model_name, device=self._device)
        # live vectors; rows are filled in as documents are revealed
        self._live = np.zeros_like(new_oracle)
        self.encode_seconds = 0.0
        self.encode_calls = 0
        self.encode_docs = 0
        self.encode_chars = 0
        self._verify_left = verify_against_oracle
        self.max_oracle_gap = 0.0

    # -- the metered path ---------------------------------------------------
    def reveal(self, ids: np.ndarray, reason: str) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        fresh = np.unique(ids[~self.revealed[ids]])
        if fresh.size:
            batch = [self._prefix + self.texts[int(i)] for i in fresh]
            t0 = time.perf_counter()
            emb = self._model.encode(batch, batch_size=self.batch_size,
                                     convert_to_numpy=True,
                                     normalize_embeddings=True,
                                     show_progress_bar=False)
            self.encode_seconds += time.perf_counter() - t0
            self.encode_calls += 1
            self.encode_docs += int(fresh.size)
            self.encode_chars += sum(len(t) for t in batch)
            emb = l2_normalize(np.asarray(emb, dtype=np.float32))
            self._live[fresh] = emb
            if self._verify_left > 0:
                take = min(self._verify_left, fresh.size)
                gap = 1.0 - np.sum(emb[:take] * self._new[fresh[:take]], axis=1)
                self.max_oracle_gap = max(self.max_oracle_gap, float(np.max(gap)))
                self._verify_left -= take
            self.meter.charge(int(fresh.size), reason)
            self.revealed[fresh] = True
        return self._live[ids]

    def peek_revealed(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        if not np.all(self.revealed[ids]):
            raise RuntimeError("peek_revealed on unrevealed ids — budget violation")
        return self._live[ids]

    # -- reporting ----------------------------------------------------------
    def encoder_stats(self) -> dict:
        return {
            "encode_seconds": round(self.encode_seconds, 3),
            "encode_calls": self.encode_calls,
            "encode_docs": self.encode_docs,
            "encode_docs_per_s": (self.encode_docs / self.encode_seconds
                                  if self.encode_seconds > 0 else float("nan")),
            "encode_chars_per_doc": (self.encode_chars / self.encode_docs
                                     if self.encode_docs else float("nan")),
            "device": self._device,
            "max_oracle_cosine_gap": self.max_oracle_gap,
        }
