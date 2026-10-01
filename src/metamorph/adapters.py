"""Cross-space adapters: generalized Procrustes, ridge linear map, small MLP.

All adapters map R^{d_in} -> R^{d_out} (dimensions may differ) and L2-normalize
their output (cosine geometry everywhere). An AdapterBank fits an ensemble in
both directions and exposes ensemble disagreement, a scheduler signal that
requires no NEW-model calls.
"""
from __future__ import annotations

import numpy as np

from .config import AdapterConfig
from .utils import l2_normalize


class BaseAdapter:
    name = "base"

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "BaseAdapter":
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, X: np.ndarray) -> np.ndarray:
        one = X.ndim == 1
        out = self.transform(np.atleast_2d(np.asarray(X, dtype=np.float32)))
        return out[0] if one else out


class ProcrustesAdapter(BaseAdapter):
    """Semi-orthogonal W (W^T W = I) minimizing ||X W - Y||_F.

    For d_in == d_out this is classical orthogonal Procrustes; for rectangular
    shapes it is the polar-factor solution W = U V^T of SVD(X^T Y).
    """

    name = "procrustes"

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        U, _, Vt = np.linalg.svd(X.T @ Y, full_matrices=False)
        self.W = (U @ Vt).astype(np.float32)  # (d_in, d_out), semi-orthogonal
        return self

    def transform(self, X):
        return l2_normalize(X @ self.W)


class RidgeAdapter(BaseAdapter):
    """Closed-form ridge regression map with bias: Y ~ X W + b."""

    name = "ridge"

    def __init__(self, lam: float = 1.0):
        self.lam = lam

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        mx, my = X.mean(0), Y.mean(0)
        Xc, Yc = X - mx, Y - my
        d = X.shape[1]
        A = Xc.T @ Xc + self.lam * np.eye(d)
        self.W = np.linalg.solve(A, Xc.T @ Yc).astype(np.float32)
        self.b = (my - mx @ self.W).astype(np.float32)
        return self

    def transform(self, X):
        return l2_normalize(X @ self.W + self.b)


class MLPAdapter(BaseAdapter):
    """Small residual-free MLP trained with cosine loss (torch, CPU/MPS)."""

    name = "mlp"

    def __init__(self, hidden: int = 512, epochs: int = 30, lr: float = 1e-3,
                 batch: int = 256, seed: int = 0, device: str | None = None):
        self.hidden, self.epochs, self.lr, self.batch, self.seed = hidden, epochs, lr, batch, seed
        self.device = device

    def _pick_device(self):
        import torch
        if self.device:
            return torch.device(self.device)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def fit(self, X, Y):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)
        dev = self._pick_device()
        X_t = torch.as_tensor(np.asarray(X, dtype=np.float32), device=dev)
        Y_t = torch.as_tensor(np.asarray(Y, dtype=np.float32), device=dev)
        d_in, d_out = X_t.shape[1], Y_t.shape[1]
        net = nn.Sequential(
            nn.Linear(d_in, self.hidden), nn.GELU(), nn.Linear(self.hidden, d_out)
        ).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr)
        n = X_t.shape[0]
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, self.batch):
                idx = perm[s:s + self.batch]
                pred = torch.nn.functional.normalize(net(X_t[idx]), dim=-1)
                tgt = torch.nn.functional.normalize(Y_t[idx], dim=-1)
                loss = (1.0 - (pred * tgt).sum(-1)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        net.eval()
        self._net, self._dev = net, dev
        return self

    def transform(self, X):
        import torch
        with torch.no_grad():
            out = self._net(torch.as_tensor(np.asarray(X, dtype=np.float32), device=self._dev))
        return l2_normalize(out.cpu().numpy())


def make_adapter(kind: str, cfg: AdapterConfig, seed: int = 0) -> BaseAdapter:
    if kind == "procrustes":
        return ProcrustesAdapter()
    if kind == "ridge":
        return RidgeAdapter(lam=cfg.ridge_lambda)
    if kind == "mlp":
        return MLPAdapter(hidden=cfg.mlp_hidden, epochs=cfg.mlp_epochs,
                          lr=cfg.mlp_lr, batch=cfg.mlp_batch, seed=seed)
    raise ValueError(f"unknown adapter kind: {kind}")


class AdapterBank:
    """Ensemble of adapters fitted on the seed pairs, in both directions."""

    def __init__(self, cfg: AdapterConfig, seed: int = 0):
        self.cfg = cfg
        self.seed = seed
        self.forward: dict[str, BaseAdapter] = {}
        self.reverse: dict[str, BaseAdapter] = {}
        self.primary_kind: str | None = None
        # global score-fusion weights: exact ~ theta . [old_score, adapted_score, 1]
        self.calib_theta = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    def fit(self, X_old_seed: np.ndarray, X_new_seed: np.ndarray) -> "AdapterBank":
        X_old_seed = l2_normalize(X_old_seed)
        X_new_seed = l2_normalize(X_new_seed)
        for kind in self.cfg.kinds:
            self.forward[kind] = make_adapter(kind, self.cfg, self.seed).fit(X_old_seed, X_new_seed)
        self.primary_kind = (self._select_primary(X_old_seed, X_new_seed)
                             if self.cfg.primary == "auto" else self.cfg.primary)
        if self.cfg.train_reverse:
            rev_kind = (self.primary_kind if self.primary_kind in self.cfg.kinds
                        else self.cfg.kinds[0])
            self.reverse[rev_kind] = make_adapter(rev_kind, self.cfg, self.seed).fit(
                X_new_seed, X_old_seed)
        self._fit_score_fusion(X_old_seed, X_new_seed)
        return self

    def _select_primary(self, X_old_seed: np.ndarray, X_new_seed: np.ndarray) -> str:
        """Pick the serving adapter by held-out cosine fidelity on the seed.

        The best adapter family depends on the seed size (Procrustes is far
        more robust than ridge/MLP at small n); an 80/20 split decides,
        then all adapters stay fitted on the full seed (refit).
        """
        n = X_old_seed.shape[0]
        if n < 20 or len(self.cfg.kinds) == 1:
            return self.cfg.kinds[0]
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n)
        n_val = max(4, n // 5)
        val, tr = perm[:n_val], perm[n_val:]
        best_kind, best_fid = self.cfg.kinds[0], -np.inf
        for kind in self.cfg.kinds:
            a = make_adapter(kind, self.cfg, self.seed).fit(X_old_seed[tr], X_new_seed[tr])
            fid = float(np.mean(np.sum(a(X_old_seed[val]) * X_new_seed[val], axis=-1)))
            if fid > best_fid:
                best_kind, best_fid = kind, fid
        return best_kind

    def _fit_score_fusion(self, X_old_seed: np.ndarray, X_new_seed: np.ndarray,
                          max_pairs: int = 4000, lam: float = 1e-3) -> None:
        """Global fusion of the two score sources onto the exact-score scale.

        For an unmigrated doc x and query q, TWO relevance signals exist:
          u1 = cos(x_old, q_old)      exact score in the OLD space
          u2 = cos(f(x_old), q_new)   adapted score in the NEW space
        Neither is on the exact new-score scale (e.g. E5 cosines sit in a
        narrow high band), so raw scores are not comparable with exact scores
        of migrated candidates. Fit  v ~ theta . [u1, u2, 1]  by least squares
        over seed-pair pseudo-queries (seed docs play both roles). Serving
        re-fits theta per query on dual-scored migrated candidates, shrinking
        toward this global fit.
        """
        Y = X_new_seed
        Xo = X_old_seed
        n = Y.shape[0]
        rng = np.random.default_rng(self.seed)
        # CROSS-FITTED adapter scores: the fusion must see the adapter's
        # out-of-sample behavior — scoring the seed with the adapter trained
        # on that same seed would leak (u2 looks near-perfect and the fusion
        # would zero out the old-space evidence). Two folds, each scored by
        # the adapter fitted on the other fold.
        kind = self.primary_kind or self.cfg.kinds[0]
        if n >= 24:
            F = np.empty_like(Y)
            perm = rng.permutation(n)
            half = n // 2
            folds = (perm[:half], perm[half:])
            for h, fold in enumerate(folds):
                other = folds[1 - h]
                a = make_adapter(kind, self.cfg, self.seed).fit(Xo[other], Y[other])
                F[fold] = a(Xo[fold])
        else:
            F = self.f(Xo)
        # kept for serving: per-query fusion refits use these LEAK-FREE
        # adapted embeddings for seed docs (aligned with the seed order)
        self.cv_adapted_seed = F
        m = min(max_pairs, n * max(n - 1, 1))
        i = rng.integers(0, n, size=m)
        j = rng.integers(0, n, size=m)
        keep = i != j
        i, j = i[keep], j[keep]
        u1 = np.sum(Xo[i] * Xo[j], axis=-1)   # old-space exact score
        u2 = np.sum(F[i] * Y[j], axis=-1)     # adapted score
        v = np.sum(Y[i] * Y[j], axis=-1)      # exact new-space score
        A = np.stack([u1, u2, np.ones_like(u1)], axis=1).astype(np.float64)
        AtA = A.T @ A + lam * np.eye(3)
        self.calib_theta = np.linalg.solve(AtA, A.T @ v.astype(np.float64))

    @property
    def f(self) -> BaseAdapter:
        """Primary forward adapter old -> new (serving scores)."""
        return self.forward[self.primary_kind or self.cfg.kinds[0]]

    @property
    def g(self) -> BaseAdapter | None:
        """Reverse adapter new -> old (query routing in new_only mode)."""
        return next(iter(self.reverse.values()), None)

    def disagreement(self, X_old: np.ndarray, batch: int = 8192) -> np.ndarray:
        """Per-row ensemble disagreement: mean pairwise (1 - cos) of adapted outputs."""
        kinds = list(self.forward)
        if len(kinds) < 2:
            return np.zeros(X_old.shape[0], dtype=np.float32)
        n = X_old.shape[0]
        out = np.zeros(n, dtype=np.float32)
        for s in range(0, n, batch):
            chunk = X_old[s:s + batch]
            outs = [self.forward[k](chunk) for k in kinds]
            acc, pairs = np.zeros(chunk.shape[0], dtype=np.float32), 0
            for i in range(len(outs)):
                for j in range(i + 1, len(outs)):
                    acc += 1.0 - np.sum(outs[i] * outs[j], axis=-1)
                    pairs += 1
            out[s:s + batch] = acc / pairs
        return out
