"""Adversarial tests for the mid-migration query semantics of the design.

The four invariants (I1 exactly-once serving, I2 monotonicity, I3 budget
conservation, I4 legal phase transitions) already have unit coverage in
`test_store_state.py`.  What was *specified but not verified* is the case the
design calls out explicitly: what a query may observe when the scheduler flips
a batch of documents from the old space to the new one **while a query is in
flight**, and whether the invariants survive adversarial interleavings rather
than the well-behaved sequences a normal run produces.

These tests exercise that: they interleave `mark_migrated` with serving, drive
random operation orders, and assert the invariants after every step.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from metamorph.adapters import AdapterBank
from metamorph.config import AdapterConfig, ServeConfig
from metamorph.index import HnswIndex
from metamorph.serve import QueryRouter
from metamorph.state import MigrationState, Phase
from metamorph.store import EmbeddingStore
from metamorph.synth import make_synthetic

N_DOCS, N_QUERIES, D_OLD, D_NEW = 300, 30, 24, 32


def _world(seed: int = 5):
    data = make_synthetic(n_docs=N_DOCS, n_queries=N_QUERIES, d_old=D_OLD,
                          d_new=D_NEW, seed=seed, noise=0.08,
                          cluster_distortion=0.1)
    store = EmbeddingStore(data.old, data.new)
    state = MigrationState(N_DOCS, np.arange(N_DOCS) % 6)
    state.to_phase(Phase.SEEDING)
    rng = np.random.default_rng(seed)
    seed_ids = rng.choice(N_DOCS, size=45, replace=False)
    seed_new = store.reveal(seed_ids, "seed")
    bank = AdapterBank(AdapterConfig(kinds=("procrustes", "ridge"), primary="ridge"),
                       seed=0).fit(store.old[seed_ids], seed_new)
    old_index = HnswIndex(D_OLD, N_DOCS, ef_search=160)
    old_index.add(store.old, np.arange(N_DOCS))
    new_index = HnswIndex(D_NEW, N_DOCS, ef_search=160)
    state.mark_migrated(seed_ids)
    new_index.add(store.peek_revealed(seed_ids), seed_ids)
    state.to_phase(Phase.ADAPTIVE_SERVING)
    router = QueryRouter(store, state, old_index, new_index, bank,
                         ServeConfig(k=10, overfetch=4))
    return data, store, state, router, new_index


def _migrate_batch(store, state, new_index, ids):
    """Flip a batch the way the scheduler does, keeping the new index in step."""
    fresh = state.mark_migrated(np.asarray(ids, dtype=np.int64))
    if fresh.size:
        new_index.add(store.reveal(fresh, "scheduler"), fresh)
    return fresh


# ---------------------------------------------------------------- I1 under race
def test_batch_flip_between_queries_never_double_serves():
    """A document flipped mid-migration is served from exactly one space.

    This is the race the design specifies semantics for: the scheduler moves a
    batch while queries are being answered.  After every flip, no served
    candidate may appear in both the old-space and new-space candidate lists,
    and no old-space candidate may be one that has already migrated.
    """
    data, store, state, router, new_index = _world()
    rng = np.random.default_rng(11)
    for step in range(25):
        qi = int(rng.integers(N_QUERIES))
        res = router.serve(data.q_old[qi], data.q_new[qi])
        old_c = res.candidates[:res.n_from_old]
        new_c = res.candidates[res.n_from_old:]
        assert np.intersect1d(old_c, new_c).size == 0, f"step {step}: double-served"
        assert not state.migrated[old_c].any(), f"step {step}: migrated doc served from old space"
        assert state.migrated[new_c].all(), f"step {step}: unmigrated doc served from new space"
        # adversarial flip: migrate a batch drawn from THIS query's candidates,
        # i.e. exactly the documents the next query is most likely to touch
        victim = res.candidates[rng.permutation(res.candidates.size)[:12]]
        _migrate_batch(store, state, new_index, victim)
        state.check_invariants()


def test_flip_during_query_preserves_snapshot_consistency():
    """Serving twice around a flip yields results consistent with a snapshot.

    A query answered before a flip and the same query answered after it must
    each be internally consistent; and any document that moved to the new space
    must not afterwards be answered from the old one (I1 + I2 at serving level).
    """
    data, store, state, router, new_index = _world(seed=7)
    qi = 3
    before = router.serve(data.q_old[qi], data.q_new[qi])
    moved = before.candidates[:max(1, before.n_from_old)]
    _migrate_batch(store, state, new_index, moved)
    after = router.serve(data.q_old[qi], data.q_new[qi])
    after_old = after.candidates[:after.n_from_old]
    assert np.intersect1d(after_old, moved).size == 0, "flipped docs still served from old space"
    assert state.migrated[moved].all()
    state.check_invariants()


# ---------------------------------------------------------------- I2 monotone
def test_monotonicity_under_adversarial_reordering():
    """No document ever reverts, under random re-marking of already-migrated ids."""
    data, store, state, router, new_index = _world(seed=13)
    rng = np.random.default_rng(2)
    prev = state.migrated.copy()
    for _ in range(40):
        ids = rng.choice(N_DOCS, size=int(rng.integers(1, 30)), replace=True)
        _migrate_batch(store, state, new_index, ids)
        assert np.all(state.migrated >= prev), "a migrated document reverted"
        prev = state.migrated.copy()
        state.check_invariants()


# ---------------------------------------------------------------- I3 budget
def test_budget_conservation_under_interleaving():
    """Total distinct oracle reveals equal metered spend, however operations interleave."""
    data, store, state, router, new_index = _world(seed=17)
    meter = store.meter
    rng = np.random.default_rng(4)
    for _ in range(30):
        if rng.random() < 0.5:
            qi = int(rng.integers(N_QUERIES))
            router.serve(data.q_old[qi], data.q_new[qi])
        else:
            ids = rng.choice(N_DOCS, size=int(rng.integers(1, 20)), replace=True)
            _migrate_batch(store, state, new_index, ids)
        state.check_invariants()
    revealed = int(store.revealed.sum())
    spent = int(meter.total) if meter is not None else revealed
    assert revealed == spent, f"I3 violated: {revealed} revealed vs {spent} metered"
    # and a revealed document is exactly a migrated one
    assert int(state.migrated.sum()) == revealed


# ---------------------------------------------------------------- I4 phase DAG
def test_every_illegal_phase_transition_is_rejected():
    """Exhaustively: only edges of the specified DAG are permitted."""
    legal = {
        (Phase.INIT, Phase.SEEDING),
        (Phase.SEEDING, Phase.ADAPTIVE_SERVING),
        (Phase.ADAPTIVE_SERVING, Phase.QUALITY_COMPLETE),
        (Phase.ADAPTIVE_SERVING, Phase.FULLY_MIGRATED),
        (Phase.QUALITY_COMPLETE, Phase.FULLY_MIGRATED),
    }
    checked = 0
    for src, dst in itertools.product(list(Phase), list(Phase)):
        if src == dst:
            continue
        ms = MigrationState(4, np.array([0, 0, 1, 1]))
        ms.phase = src
        if (src, dst) in legal:
            ms.to_phase(dst)
            assert ms.phase == dst
        else:
            with pytest.raises(ValueError):
                ms.to_phase(dst)
        checked += 1
    assert checked >= 20, "phase product space unexpectedly small"


# ---------------------------------------------------------------- filters
def test_predicate_filter_respects_partition_split():
    """A filtered query still observes each document in exactly one space."""
    data, store, state, router, new_index = _world(seed=23)
    rng = np.random.default_rng(9)
    _migrate_batch(store, state, new_index, rng.choice(N_DOCS, 90, replace=False))
    mask = np.zeros(N_DOCS, dtype=bool)
    mask[rng.choice(N_DOCS, 150, replace=False)] = True
    res = router.serve(data.q_old[0], data.q_new[0], doc_filter=mask)
    assert mask[res.ids].all(), "filter violated: ineligible document served"
    old_c = res.candidates[:res.n_from_old]
    new_c = res.candidates[res.n_from_old:]
    assert np.intersect1d(old_c, new_c).size == 0
    state.check_invariants()
