"""Tests for the certifier's score-mass target and truncation debiasing."""
import numpy as np

from metamorph.baselines import make_engine
from metamorph.config import (AdapterConfig, CertifierConfig, EngineConfig,
                              SchedulerConfig, ServeConfig)
from metamorph.synth import make_synthetic
from metamorph.workload import ZipfStream, split_queries


def _cfg(**cert_over):
    cert = dict(sample_queries=40, pool_extra=10, period_steps=2,
                target_recall=0.8, min_budget_frac=0.0,
                max_round_cost_frac=0.02, history_window=0)
    cert.update(cert_over)
    return EngineConfig(
        seed=5, seed_fraction=0.05, batch_size=60, n_partitions=6,
        workload_per_step=8, eval_every_steps=2,
        adapter=AdapterConfig(kinds=("procrustes", "ridge"), primary="auto"),
        scheduler=SchedulerConfig(), certifier=CertifierConfig(**cert),
        serve=ServeConfig(k=10, overfetch=3))


def _run(cfg, steps=10):
    data = make_synthetic(n_docs=1500, n_queries=90, seed=2)
    engine = make_engine(data, cfg, "metamorph")
    wq, eq = split_queries(data.q_ids, seed=5)
    engine.run(ZipfStream(wq, 1.1, seed=5), eq, max_steps=steps)
    return engine


def test_score_mass_is_bounded_and_at_least_agreement():
    """Score mass is a [0,1] quantity and is never below strict agreement:
    a served list that contains j of the true top-k captures at least the mass
    of those j, and the swapped-in documents carry non-negative shifted score."""
    engine = _run(_cfg())
    assert engine.certifier.history
    for rep in engine.certifier.history:
        assert 0.0 <= rep.score_mass_hat <= 1.0
        assert rep.score_mass_hat >= rep.recall_hat - 1e-6


def test_truncation_probe_produces_non_negative_bias_and_a_bound():
    engine = _run(_cfg(probe_frac=0.5, probe_multiplier=4))
    last = engine.certifier.history[-1]
    assert last.n_probe > 0
    assert last.bias_hat >= 0.0
    assert last.bias_ub >= last.bias_hat - 1e-9


def test_bias_correction_only_lowers_the_decision_bound():
    engine = _run(_cfg(probe_frac=0.5))
    for rep in engine.certifier.history:
        target = rep.score_mass_lb if rep.metric == "score_mass" else rep.raw_pooled_lb
        assert rep.pooled_lb <= target + 1e-9


def test_probe_disabled_means_no_correction():
    engine = _run(_cfg(probe_frac=0.0))
    for rep in engine.certifier.history:
        assert rep.n_probe == 0
        assert rep.bias_ub == 0.0
        assert abs(rep.pooled_lb - rep.raw_pooled_lb) < 1e-9


def test_score_mass_metric_selects_the_score_mass_bound():
    engine = _run(_cfg(metric="score_mass", probe_frac=0.0))
    for rep in engine.certifier.history:
        assert rep.metric == "score_mass"
        assert abs(rep.pooled_lb - rep.score_mass_lb) < 1e-9


def test_anytime_bound_pools_the_whole_stream_when_window_is_zero():
    engine = _run(_cfg(history_window=0), steps=14)
    ns = [r.n_pooled for r in engine.certifier.history]
    assert ns == sorted(ns) and ns[-1] > ns[0]


def test_windowed_bound_caps_the_pool():
    engine = _run(_cfg(history_window=2, sample_queries=20), steps=14)
    hist = engine.certifier.history
    assert len(hist) >= 3
    per_round = [r.n_queries for r in hist]
    for i, rep in enumerate(hist):
        assert rep.n_pooled <= sum(per_round[max(0, i - 1):i + 1])


def test_bias_correction_shrinks_with_more_probes():
    """The truncation correction is an upper confidence bound on the bias, so
    it must decay as probe samples accumulate — otherwise a sound bound could
    never certify anything."""
    from metamorph.bounds import upper_bound_from_lb

    prev = 2.0
    for n in (20, 50, 100, 400, 1600):
        ub = upper_bound_from_lb(np.zeros(n), 0.025)
        assert ub < prev
        prev = ub
    assert prev < 0.01


def test_pool_includes_new_space_candidates():
    """Migrated documents are free to score, so the pool is widened with the
    new index's top-C as well as the old index's — closing the commonest
    source of upward truncation bias."""
    engine = _run(_cfg(pool_extra=10, probe_frac=0.0), steps=10)
    router = engine.router
    q = engine.data.q_new[0]
    q_old = engine.data.q_old[0]
    pool_wide = engine.certifier._pool_for(np.array([0, 1]), q_old, q, 10)
    pool_none = engine.certifier._pool_for(np.array([0, 1]), q_old, q, 0)
    assert pool_none.size == 2
    assert pool_wide.size > pool_none.size
    if router.new_index.count:
        new_ids, _ = router.new_index.search(q, 10)
        assert set(new_ids.tolist()).issubset(set(pool_wide.tolist()))
