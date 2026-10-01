"""End-to-end integration: full migration on synthetic data, all methods."""
import numpy as np
import pytest

from metamorph.baselines import endpoint_quality, full_swap_timeline, make_engine
from metamorph.config import AdapterConfig, CertifierConfig, EngineConfig, SchedulerConfig, ServeConfig
from metamorph.state import Phase
from metamorph.synth import make_synthetic
from metamorph.workload import ZipfStream, split_queries


def small_cfg(**over) -> EngineConfig:
    cfg = EngineConfig(
        seed=7, seed_fraction=0.05, batch_size=40, n_partitions=6,
        workload_per_step=8, eval_every_steps=2,
        adapter=AdapterConfig(kinds=("procrustes", "ridge"), primary="ridge",
                              train_reverse=True),
        scheduler=SchedulerConfig(),
        certifier=CertifierConfig(sample_queries=8, pool_extra=20, period_steps=3,
                                  target_recall=0.9),
        serve=ServeConfig(k=10, overfetch=3, ef_search=150),
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture(scope="module")
def data():
    return make_synthetic(n_docs=500, n_queries=60, d_old=24, d_new=32, seed=11)


def run_method(data, method, **cfg_over):
    engine = make_engine(data, small_cfg(**cfg_over), method)
    workload_qids, eval_qids = split_queries(data.q_ids, seed=7)
    stream = ZipfStream(workload_qids, alpha=1.1, seed=7)
    timeline = engine.run(stream, eval_qids, max_steps=100)
    return engine, timeline


def test_metamorph_full_run(data):
    engine, timeline = run_method(data, "metamorph")
    assert engine.state.phase == Phase.FULLY_MIGRATED
    assert engine.state.migrated.all()
    # budget conservation (I3): meter total == revealed docs
    assert engine.meter.total == int(engine.store.revealed.sum())
    # quality at the end matches destination quality
    final = timeline[-1]
    assert final["recall@10_vs_dest"] > 0.9
    # monotone budget along the timeline
    budgets = [r["budget_frac"] for r in timeline]
    assert all(b2 >= b1 for b1, b2 in zip(budgets, budgets[1:]))
    # cost breakdown columns present
    assert "cost_seed" in final and "cost_scheduler" in final


def test_static_adapter_spends_only_seed(data):
    engine, timeline = run_method(data, "static_adapter", max_budget_fraction=0.05)
    assert engine.meter.total <= int(np.ceil(0.06 * data.n_docs)) + 1
    assert engine.state.phase != Phase.FULLY_MIGRATED
    # still delivers useful quality via the adapter (random top-10 would be 0.02;
    # the synthetic space is deliberately distorted so adapter-only saturates low)
    assert timeline[-1]["recall@10_vs_dest"] > 0.15


def test_random_vs_priority_reproducible(data):
    _, t1 = run_method(data, "random_order")
    _, t2 = run_method(data, "random_order")
    assert [r["budget_units"] for r in t1] == [r["budget_units"] for r in t2]
    assert t1[-1]["recall@10_vs_dest"] == t2[-1]["recall@10_vs_dest"]


def test_full_swap_baseline(data):
    _, eval_qids = split_queries(data.q_ids, seed=7)
    rows = full_swap_timeline(data, eval_qids, k=10)
    assert rows[0]["budget_frac"] == 0.0 and rows[-1]["budget_frac"] == 1.0
    assert rows[-1]["recall@10_vs_dest"] == 1.0  # destination system vs itself
    assert rows[0]["recall@10_vs_dest"] < 1.0


def test_endpoint_quality_orders(data):
    _, eval_qids = split_queries(data.q_ids, seed=7)
    q_new = endpoint_quality(data, eval_qids, 10, space="new")
    q_old = endpoint_quality(data, eval_qids, 10, space="old")
    assert q_new["recall@10_vs_dest"] == 1.0
    assert q_old["recall@10_vs_dest"] < q_new["recall@10_vs_dest"]


def test_certifier_fires_before_full_budget():
    # well-behaved space (low distortion) + moderate target: certification
    # should fire strictly before the full budget is spent. The corpus must be
    # large enough that one certification round's pool reveals do not cover it
    # (at 500 docs a single round touches ~70% of the corpus and the race with
    # full migration makes the property vacuous).
    easy = make_synthetic(n_docs=3000, n_queries=100, d_old=24, d_new=32,
                          noise=0.05, cluster_distortion=0.05, seed=13)
    cfg = small_cfg(batch_size=60)
    # probe_frac=0: this test covers the certifier's core firing behaviour.
    # The pool-truncation correction is a separate, deliberately conservative
    # layer whose subtraction is bounded by an upper confidence bound on the
    # bias; with only a handful of probe samples that UCB dominates, which is
    # correct behaviour and is covered by test_certify_targets /
    # test_bias_correction_shrinks_with_more_probes.
    cfg.certifier = CertifierConfig(sample_queries=50, pool_extra=10,
                                    period_steps=2, target_recall=0.7,
                                    probe_frac=0.0, max_round_cost_frac=0.02)
    engine = make_engine(easy, cfg, "metamorph")
    workload_qids, eval_qids = split_queries(easy.q_ids, seed=7)
    stream = ZipfStream(workload_qids, alpha=1.1, seed=7)
    engine.run(stream, eval_qids, max_steps=100)
    assert engine.certified_at is not None
    assert engine.certified_at < 1.0
    assert engine.state.phase in (Phase.QUALITY_COMPLETE, Phase.FULLY_MIGRATED)
