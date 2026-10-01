"""Drift-Adapter training, isolated in a torch-only worker process.

PyTorch and FAISS each link their own libomp on macOS and abort when both are
initialised in one process ("OMP: Error #15"). exp10 therefore trains adapters
here, in a spawned process that never imports FAISS, and receives plain NumPy
weights back; inference is re-implemented in NumPy (`apply_weights`) and
checked against the torch forward pass on every fit (`check` field).

Adapter definitions follow Drift-Adapter (Vejendla, EMNLP 2025), Sec. 3-4 and
App. A: OP (closed form), LA (U V^T x + t, rank 64, DSM), residual MLP
(x + W2 GELU(W1 x + b1) + b2, 256 hidden, dropout 0.1, DSM); MSE loss, AdamW
lr 3e-4, weight decay 0.01, batch 256, <= 50 epochs, early stopping on
validation MSE with patience 5, 80/20 train/validation split.
"""
from __future__ import annotations

import copy
import time

import numpy as np
from scipy.special import erf


def procrustes(X_in: np.ndarray, Y_out: np.ndarray) -> np.ndarray:
    """R (d_out x d_in) minimizing ||X_in R^T - Y_out||_F, orthonormal rows or
    columns: R = U V^T from SVD(A B^T) with A = Y_out^T, B = X_in^T."""
    U, _, Vt = np.linalg.svd(Y_out.T.astype(np.float64) @ X_in.astype(np.float64),
                             full_matrices=False)
    return (U @ Vt).astype(np.float32)


def _gelu(x: np.ndarray) -> np.ndarray:  # exact (erf) GELU, torch's default
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def apply_weights(w: dict, Z: np.ndarray, chunk: int = 131072) -> np.ndarray:
    """NumPy inference for every adapter kind (dropout is inactive at eval)."""
    out = []
    for s in range(0, Z.shape[0], chunk):
        z = np.asarray(Z[s:s + chunk], dtype=np.float32)
        if w["kind"] == "OP":
            y = z @ w["R"].T
        elif w["kind"] == "OP+DSM":
            y = (z @ w["R"].T) * w["s"]
        elif w["kind"] == "LA":
            y = ((z @ w["V"].T) @ w["U"].T + w["t"]) * w["s"]
        elif w["kind"] == "MLP":
            base = z if w["P"] is None else z @ w["P"].T
            h = _gelu(z @ w["W1"].T + w["b1"])
            y = (base + h @ w["W2"].T + w["b2"]) * w["s"]
        else:
            raise ValueError(w["kind"])
        out.append(y.astype(np.float32))
    return np.concatenate(out)


def _train_mse(torch, model, X, Y, seed, lr=3e-4, wd=0.01, batch=256, epochs=50,
               patience=5, val_frac=0.2):
    g = torch.Generator().manual_seed(seed)
    n = X.shape[0]
    perm = torch.randperm(n, generator=g)
    n_val = int(round(val_frac * n)) if val_frac > 0 else 0
    va, tr = perm[:n_val], perm[n_val:]
    Xt, Yt = torch.as_tensor(X), torch.as_tensor(Y)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best, best_state, bad, ep_run, best_ep = float("inf"), None, 0, 0, 0
    for ep in range(epochs):
        ep_run = ep + 1
        model.train()
        order = tr[torch.randperm(tr.numel(), generator=g)]
        for s in range(0, order.numel(), batch):
            b = order[s:s + batch]
            loss = ((model(Xt[b]) - Yt[b]) ** 2).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if n_val == 0:
            continue
        model.eval()
        with torch.no_grad():
            v = float(((model(Xt[va]) - Yt[va]) ** 2).sum(-1).mean())
        if v < best - 1e-9:
            best, bad, best_ep = v, 0, ep_run
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return ep_run, best_ep, best


RECIPES = {
    # Drift-Adapter as published (Sec. 4, App. A.2)
    "paper": [dict(lr=3e-4, epochs=50, patience=5, rank=64, hidden=256)],
    # same hyperparameters, but trained until validation MSE stops improving
    "converged": [dict(lr=3e-4, epochs=1000, patience=10, rank=64, hidden=256)],
    # best of the paper's own sensitivity ranges (App. A.2), chosen on validation MSE
    "grid": [dict(lr=lr, epochs=500, patience=10, rank=r, hidden=h)
             for lr in (1e-4, 3e-4, 1e-3) for r, h in ((64, 256), (128, 512))],
}


def train_adapter_recipe(kind: str, X_in: np.ndarray, Y_out: np.ndarray, seed: int,
                         recipe: str = "paper", threads: int = 4) -> dict:
    """Fit under a recipe; for several configurations keep the lowest validation MSE."""
    best = None
    configs = RECIPES[recipe] if kind in ("LA", "MLP") else RECIPES["paper"][:1]
    for cfg in configs:
        w = train_adapter(kind, X_in, Y_out, seed, threads=threads, **cfg)
        if best is None or (w["info"]["val_mse"] < best["info"]["val_mse"]):
            best = w
    best["info"]["recipe"] = recipe
    return best


def train_adapter(kind: str, X_in: np.ndarray, Y_out: np.ndarray, seed: int,
                  threads: int = 4, lr: float = 3e-4, epochs: int = 50, patience: int = 5,
                  rank: int = 64, hidden: int = 256) -> dict:
    """Fit one adapter X_in -> Y_out (rows L2-normalized). Returns NumPy weights."""
    import torch
    import torch.nn as nn

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    X_in = np.ascontiguousarray(X_in, dtype=np.float32)
    Y_out = np.ascontiguousarray(Y_out, dtype=np.float32)
    d_in, d_out = X_in.shape[1], Y_out.shape[1]
    t0 = time.time()

    class LowRankAffine(nn.Module):
        def __init__(self, r=rank):
            super().__init__()
            self.V = nn.Linear(d_in, r, bias=False)
            self.U = nn.Linear(r, d_out, bias=True)  # bias = t
            self.s = nn.Parameter(torch.ones(d_out))  # DSM

        def forward(self, x):
            return self.U(self.V(x)) * self.s

    class ResidualMLP(nn.Module):
        def __init__(self, shortcut, hidden=hidden, dropout=0.1):
            super().__init__()
            self.fc1 = nn.Linear(d_in, hidden)
            self.act = nn.GELU()
            self.drop = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden, d_out)
            self.P = None if shortcut is None else torch.as_tensor(shortcut)  # fixed
            self.s = nn.Parameter(torch.ones(d_out))  # DSM

        def forward(self, x):
            base = x if self.P is None else x @ self.P.T
            return (base + self.fc2(self.drop(self.act(self.fc1(x))))) * self.s

    class DiagScale(nn.Module):
        def __init__(self):
            super().__init__()
            self.s = nn.Parameter(torch.ones(d_out))

        def forward(self, x):
            return x * self.s

    info = {"kind": kind, "epochs": 0, "best_epoch": 0, "val_mse": float("nan"),
            "lr": lr, "max_epochs": epochs, "patience": patience,
            "rank": rank if kind == "LA" else 0, "hidden": hidden if kind == "MLP" else 0}
    if kind == "OP":
        w = {"kind": kind, "R": procrustes(X_in, Y_out)}
    elif kind == "OP+DSM":
        R = procrustes(X_in, Y_out)
        m = DiagScale()
        ep, be, v = _train_mse(torch, m, X_in @ R.T, Y_out, seed, epochs=20,
                               patience=10 ** 9, val_frac=0.0)
        w = {"kind": kind, "R": R, "s": m.s.detach().numpy().copy()}
        info.update(epochs=ep)
    elif kind == "LA":
        m = LowRankAffine()
        ep, be, v = _train_mse(torch, m, X_in, Y_out, seed, lr=lr, epochs=epochs,
                               patience=patience)
        w = {"kind": kind, "V": m.V.weight.detach().numpy().copy(),
             "U": m.U.weight.detach().numpy().copy(), "t": m.U.bias.detach().numpy().copy(),
             "s": m.s.detach().numpy().copy()}
        info.update(epochs=ep, best_epoch=be, val_mse=v)
    elif kind == "MLP":
        P = None if d_in == d_out else procrustes(X_in, Y_out)
        m = ResidualMLP(P)
        ep, be, v = _train_mse(torch, m, X_in, Y_out, seed, lr=lr, epochs=epochs,
                               patience=patience)
        w = {"kind": kind, "P": P, "W1": m.fc1.weight.detach().numpy().copy(),
             "b1": m.fc1.bias.detach().numpy().copy(),
             "W2": m.fc2.weight.detach().numpy().copy(),
             "b2": m.fc2.bias.detach().numpy().copy(), "s": m.s.detach().numpy().copy()}
        info.update(epochs=ep, best_epoch=be, val_mse=v)
    else:
        raise ValueError(kind)
    # parity check: NumPy inference must match the torch forward pass
    probe = X_in[:256]
    if kind in ("LA", "MLP"):
        with torch.no_grad():
            ref = m(torch.as_tensor(probe)).numpy()
        info["check_max_abs"] = float(np.abs(apply_weights(w, probe) - ref).max())
    else:
        info["check_max_abs"] = 0.0
    info["fit_s"] = time.time() - t0
    w["info"] = info
    return w
