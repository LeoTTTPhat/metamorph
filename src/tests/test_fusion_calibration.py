"""Tests for dual-evidence score fusion and its leakage-safe calibration."""
import numpy as np
import pytest

from metamorph.adapters import AdapterBank
from metamorph.baselines import make_engine
from metamorph.config import AdapterConfig, EngineConfig, CertifierConfig, SchedulerConfig, ServeConfig
from metamorph.synth import make_synthetic
from metamorph.utils import l2_normalize


def _bank(n=200, d_old=24, d_new=32, seed=3):
    rng = np.random.default_rng(seed)
    X = l2_normalize(rng.normal(size=(n, d_old)))
    W = np.linalg.svd(rng.normal(size=(d_old, d_new)), full_matrices=False)
    W = (W[0] @ W[2]).astype(np.float32)
    Y = l2_normalize(X @ W + 0.1 * rng.normal(size=(n, d_new)))
    bank = AdapterBank(AdapterConfig(kinds=("procrustes", "ridge"), primary="auto"),
                       seed=seed).fit(X, Y)
    return bank, X, Y


def test_fusion_theta_is_finite_and_uses_both_features():
    bank, _, _ = _bank()
    th = bank.calib_theta
    assert th.shape == (3,)
    assert np.all(np.isfinite(th))
    # in a learnable-but-noisy space, the adapted score must carry weight
    assert abs(th[1]) > 1e-3


def test_primary_auto_selection_picks_a_fitted_kind():
    bank, _, _ = _bank()
    assert bank.primary_kind in ("procrustes", "ridge")
    assert bank.f is bank.forward[bank.primary_kind]


def test_cross_fit_prevents_leak_dominated_theta():
    # With leakage, the adapted feature would look near-perfect and take all
    # the weight while the old-score feature is zeroed/negated. Cross-fitting
    # must keep the old-score weight non-negative in a regime where the
    # old space is genuinely informative.
    rng = np.random.default_rng(0)
    n, d = 300, 24
    X = l2_normalize(rng.normal(size=(n, d)))
    # new space = same space (old score is a perfect predictor), adapter noisy
    Y = l2_normalize(X + 0.4 * rng.normal(size=(n, d)))
    bank = AdapterBank(AdapterConfig(kinds=("ridge",), primary="ridge"),
                       seed=0).fit(X, Y)
    assert bank.calib_theta[0] > 0.0


def test_mid_migration_quality_not_below_endpoints_floor():
    """Serving NDCG during migration must never crater far below the WORSE
    endpoint (the regression the fusion fix addresses)."""
    data = make_synthetic(n_docs=800, n_queries=80, d_old=24, d_new=32, seed=5)
    cfg = EngineConfig(
        seed=5, seed_fraction=0.05, batch_size=80, n_partitions=6,
        workload_per_step=8, eval_every_steps=2,
        adapter=AdapterConfig(kinds=("procrustes", "ridge"), primary="auto"),
        scheduler=SchedulerConfig(),
        certifier=CertifierConfig(enabled=False),
        serve=ServeConfig(k=10, overfetch=3),
    )
    from metamorph.baselines import endpoint_quality
    from metamorph.workload import ZipfStream, split_queries

    wq, eq = split_queries(data.q_ids, seed=5)
    old_q = endpoint_quality(data, eq, 10, space="old")["ndcg@10"]
    new_q = endpoint_quality(data, eq, 10, space="new")["ndcg@10"]
    floor = 0.85 * min(old_q, new_q)
    engine = make_engine(data, cfg, "metamorph")
    timeline = engine.run(ZipfStream(wq, 1.1, seed=5), eq, max_steps=50)
    ndcgs = [r["ndcg@10"] for r in timeline if np.isfinite(r["ndcg@10"])]
    assert len(ndcgs) >= 3
    assert min(ndcgs) >= floor, (min(ndcgs), old_q, new_q)
