"""Tests for cost-capped certification rounds and the pooled anytime bound."""
import numpy as np

from metamorph.baselines import make_engine
from metamorph.certify import empirical_bernstein_lb
from metamorph.config import (AdapterConfig, CertifierConfig, EngineConfig,
                              SchedulerConfig, ServeConfig)
from metamorph.synth import make_synthetic
from metamorph.workload import ZipfStream, split_queries


def _cfg(**cert_over):
    cert = dict(sample_queries=50, pool_extra=10, period_steps=2,
                target_recall=0.8, min_budget_frac=0.0,
                max_round_cost_frac=0.01, history_window=6)
    cert.update(cert_over)
    return EngineConfig(
        seed=9, seed_fraction=0.05, batch_size=60, n_partitions=6,
        workload_per_step=8, eval_every_steps=2,
        adapter=AdapterConfig(kinds=("procrustes", "ridge"), primary="auto"),
        scheduler=SchedulerConfig(),
        certifier=CertifierConfig(**cert),
        serve=ServeConfig(k=10, overfetch=3),
    )


def test_round_cost_respects_cap_up_to_one_pool():
    data = make_synthetic(n_docs=1500, n_queries=90, seed=2)
    cfg = _cfg()
    engine = make_engine(data, cfg, "metamorph")
    wq, eq = split_queries(data.q_ids, seed=9)
    engine.run(ZipfStream(wq, 1.1, seed=9), eq, max_steps=8)
    cap = int(round(cfg.certifier.max_round_cost_frac * data.n_docs))
    # one query's pool may overshoot the cap; that query may also carry the
    # deeper pool-truncation probe, so the bound is the cap plus one served
    # list plus one PROBE-sized pool.
    cc = cfg.certifier
    max_pool = cfg.serve.k + cc.pool_extra * max(1, cc.probe_multiplier)
    for rep in engine.certifier.history:
        assert rep.cost_units <= cap + max_pool
        assert rep.n_queries >= 1  # at least one query always processed


def test_pooled_bound_uses_growing_history():
    data = make_synthetic(n_docs=1500, n_queries=90, seed=2)
    engine = make_engine(data, _cfg(), "metamorph")
    wq, eq = split_queries(data.q_ids, seed=9)
    engine.run(ZipfStream(wq, 1.1, seed=9), eq, max_steps=12)
    hist = engine.certifier.history
    assert len(hist) >= 3
    n_pooled = [h.n_pooled for h in hist]
    assert n_pooled == sorted(n_pooled)          # windowed pool never shrinks here
    assert n_pooled[-1] >= hist[-1].n_queries    # pooled >= this round's m
    for h in hist:
        assert h.pooled_lb <= h.recall_hat + 1e-9 or h.n_pooled > h.n_queries


def test_certifier_fires_with_pooled_samples_in_easy_space():
    easy = make_synthetic(n_docs=2000, n_queries=100, d_old=24, d_new=32,
                          noise=0.05, cluster_distortion=0.05, seed=13)
    cfg = _cfg(target_recall=0.7, max_round_cost_frac=0.02)
    engine = make_engine(easy, cfg, "metamorph")
    wq, eq = split_queries(easy.q_ids, seed=7)
    engine.run(ZipfStream(wq, 1.1, seed=7), eq, max_steps=100)
    assert engine.certified_at is not None
    assert engine.certified_at <= 1.0


def test_bernstein_pooling_tightens_bound():
    rng = np.random.default_rng(0)
    small = rng.choice([0.9, 1.0], size=20)
    large = rng.choice([0.9, 1.0], size=400)
    assert empirical_bernstein_lb(large, 0.05) > empirical_bernstein_lb(small, 0.05)
