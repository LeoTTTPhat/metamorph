import numpy as np
import pytest

from metamorph.adapters import AdapterBank
from metamorph.certify import Certifier, empirical_bernstein_lb
from metamorph.config import AdapterConfig, CertifierConfig, ServeConfig
from metamorph.index import BruteForceIndex, HnswIndex
from metamorph.serve import QueryRouter
from metamorph.state import MigrationState, Phase
from metamorph.store import EmbeddingStore
from metamorph.synth import make_synthetic


@pytest.fixture(scope="module")
def world():
    data = make_synthetic(n_docs=400, n_queries=40, d_old=24, d_new=32, seed=3,
                          noise=0.08, cluster_distortion=0.1)
    store = EmbeddingStore(data.old, data.new)
    part_of = np.arange(400) % 8
    state = MigrationState(400, part_of)
    state.to_phase(Phase.SEEDING)
    rng = np.random.default_rng(0)
    seed_ids = rng.choice(400, size=60, replace=False)
    seed_new = store.reveal(seed_ids, "seed")
    bank = AdapterBank(AdapterConfig(kinds=("procrustes", "ridge"), primary="ridge"),
                       seed=0).fit(store.old[seed_ids], seed_new)
    old_index = HnswIndex(24, 400, ef_search=200)
    old_index.add(store.old, np.arange(400))
    new_index = HnswIndex(32, 400, ef_search=200)
    state.mark_migrated(seed_ids)
    new_index.add(store.peek_revealed(seed_ids), seed_ids)
    state.to_phase(Phase.ADAPTIVE_SERVING)
    router = QueryRouter(store, state, old_index, new_index, bank,
                         ServeConfig(k=10, overfetch=4))
    return data, store, state, router, bank


def test_serve_exactly_once(world):
    """I1: candidates from the two spaces never overlap; both spaces contribute."""
    data, store, state, router, _ = world
    res = router.serve(data.q_old[0], data.q_new[0])
    old_c = res.candidates[:res.n_from_old]
    new_c = res.candidates[res.n_from_old:]
    assert np.intersect1d(old_c, new_c).size == 0
    assert not state.migrated[old_c].any()
    assert state.migrated[new_c].all()
    assert res.ids.size == 10
    assert np.all(np.diff(res.scores) <= 1e-6)


def test_serve_quality_beats_random(world):
    data, store, state, router, _ = world
    dest = BruteForceIndex(store.oracle_full())
    hits = 0
    for i in range(20):
        res = router.serve(data.q_old[i], data.q_new[i])
        true_ids, _ = dest.search(data.q_new[i], 10)
        hits += np.intersect1d(res.ids, true_ids).size
    # exact top-10 agreement with the destination system, with only ~15% of
    # docs migrated and seed-trained adapters; random would be 10/400 = 0.025
    assert hits / (20 * 10) > 0.35


def test_doc_filter_respected(world):
    data, store, state, router, _ = world
    doc_filter = np.zeros(400, dtype=bool)
    doc_filter[:100] = True
    res = router.serve(data.q_old[1], data.q_new[1], doc_filter=doc_filter)
    assert (res.ids < 100).all()


def test_verification_spends_budget_and_migrates(world):
    data, store, state, router, _ = world
    router.cfg.verify_per_query = 3
    before = store.meter.by_reason.get("verify", 0)
    res = router.serve(data.q_old[2], data.q_new[2])
    router.cfg.verify_per_query = 0
    spent = store.meter.by_reason.get("verify", 0) - before
    assert spent == res.verified_ids.size > 0
    assert store.revealed[res.verified_ids].all()


def test_bernstein_bound_sane():
    rng = np.random.default_rng(0)
    samples = rng.uniform(0.9, 1.0, size=50)
    lb = empirical_bernstein_lb(samples, delta=0.05)
    assert 0.0 <= lb <= samples.mean()
    assert lb > 0.7  # tight for low variance
    assert empirical_bernstein_lb(np.array([]), 0.05) == 0.0


def test_certifier_estimates_recall(world):
    data, store, state, router, _ = world
    cfg = CertifierConfig(sample_queries=15, pool_extra=30, recycle=True,
                          target_recall=0.99)
    cert = Certifier(store, state, router, cfg, np.random.default_rng(0))
    before = store.meter.by_reason.get("certify", 0)
    report = cert.certify(data.q_old, data.q_new)
    assert 0.0 <= report.lower_bound <= report.recall_hat <= 1.0
    assert report.cost_units == store.meter.by_reason.get("certify", 0) - before
    assert report.cost_units > 0
    # recycled docs are revealed (paid for) and handed to the engine to absorb
    recycled = cert.pop_recycled()
    assert store.revealed[recycled].all()
    assert not state.migrated[recycled].any()
    assert cert.pop_recycled().size == 0  # pop is one-shot
    assert report.miss_mass_by_partition.shape == (8,)
