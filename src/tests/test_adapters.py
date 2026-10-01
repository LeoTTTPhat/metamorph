import numpy as np
import pytest

from metamorph.adapters import AdapterBank, MLPAdapter, ProcrustesAdapter, RidgeAdapter
from metamorph.config import AdapterConfig
from metamorph.synth import random_semi_orthogonal
from metamorph.utils import l2_normalize


@pytest.fixture(scope="module")
def rotated_pair():
    rng = np.random.default_rng(0)
    X = l2_normalize(rng.normal(size=(400, 32)))
    W = random_semi_orthogonal(32, 48, rng)
    Y = l2_normalize(X @ W + 0.01 * rng.normal(size=(400, 48)))
    return X, Y, W


def test_procrustes_recovers_rotation(rotated_pair):
    X, Y, _ = rotated_pair
    ad = ProcrustesAdapter().fit(X[:300], Y[:300])
    # semi-orthogonality of W: orthonormal on the smaller-dimension side
    WWt = ad.W @ ad.W.T  # (32, 32) since d_in=32 < d_out=48
    assert np.allclose(WWt, np.eye(32), atol=1e-4)
    # held-out mapping is near-exact
    cos = np.sum(ad(X[300:]) * Y[300:], axis=-1)
    assert cos.mean() > 0.98


def test_ridge_and_mlp_handle_dim_change(rotated_pair):
    X, Y, _ = rotated_pair
    for ad in (RidgeAdapter(lam=1e-2), MLPAdapter(hidden=64, epochs=100, batch=64, seed=0, device="cpu")):
        ad.fit(X[:300], Y[:300])
        out = ad(X[300:])
        assert out.shape == (100, 48)
        assert np.allclose(np.linalg.norm(out, axis=-1), 1.0, atol=1e-4)
        cos = np.sum(out * Y[300:], axis=-1)
        assert cos.mean() > 0.9, f"{ad.name} mean cos {cos.mean():.3f}"


def test_single_vector_call(rotated_pair):
    X, Y, _ = rotated_pair
    ad = RidgeAdapter().fit(X, Y)
    v = ad(X[0])
    assert v.shape == (48,)


def test_bank_disagreement_higher_off_manifold():
    rng = np.random.default_rng(1)
    X = l2_normalize(rng.normal(size=(500, 16)))
    W = random_semi_orthogonal(16, 16, rng)
    Y = l2_normalize(X @ W + 0.02 * rng.normal(size=(500, 16)))
    cfg = AdapterConfig(kinds=("procrustes", "ridge"), primary="ridge", train_reverse=True)
    bank = AdapterBank(cfg, seed=0).fit(X[:200], Y[:200])
    assert bank.g is not None
    dis_in = bank.disagreement(X)                      # in-distribution
    X_out = l2_normalize(rng.normal(size=(500, 16)) * 5 + 3)  # shifted region
    dis_out = bank.disagreement(X_out)
    assert dis_out.mean() > dis_in.mean()  # adapters disagree more off-manifold
