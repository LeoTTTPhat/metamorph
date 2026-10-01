import numpy as np
import pytest

from metamorph.state import MigrationState, PartitionState, Phase
from metamorph.store import EmbeddingStore
from metamorph.utils import CostMeter, l2_normalize


def make_store(n=50, d1=8, d2=12, seed=0):
    rng = np.random.default_rng(seed)
    return EmbeddingStore(rng.normal(size=(n, d1)), rng.normal(size=(n, d2)))


def test_reveal_charges_once_per_doc():
    st = make_store()
    st.reveal(np.array([1, 2, 3]), "seed")
    assert st.meter.total == 3
    st.reveal(np.array([2, 3, 4, 4]), "scheduler")  # dup ids + already revealed
    assert st.meter.total == 4
    assert st.meter.by_reason == {"seed": 3, "scheduler": 1}


def test_peek_unrevealed_raises():
    st = make_store()
    st.reveal(np.array([0]), "seed")
    st.peek_revealed(np.array([0]))
    with pytest.raises(RuntimeError):
        st.peek_revealed(np.array([1]))


def test_reveal_returns_normalized_new_embeddings():
    st = make_store()
    out = st.reveal(np.array([5]), "seed")
    assert np.allclose(np.linalg.norm(out, axis=-1), 1.0, atol=1e-5)


def test_state_machine_transitions_and_invariants():
    part_of = np.array([0, 0, 0, 1, 1, 2])
    ms = MigrationState(6, part_of)
    ms.to_phase(Phase.SEEDING)
    ms.to_phase(Phase.ADAPTIVE_SERVING)
    assert ms.part_state[0] == PartitionState.OLD_ONLY
    fresh = ms.mark_migrated(np.array([0, 3]))
    assert set(fresh.tolist()) == {0, 3}
    assert ms.part_state[0] == PartitionState.DUAL
    ms.check_invariants()
    # re-marking is a no-op (I2 monotone)
    assert ms.mark_migrated(np.array([0])).size == 0
    ms.mark_migrated(np.array([1, 2]))
    assert ms.part_state[0] == PartitionState.NEW_ONLY
    # full migration flips the global phase
    ms.mark_migrated(np.array([4, 5]))
    assert ms.phase == Phase.FULLY_MIGRATED
    ms.check_invariants()


def test_illegal_phase_transition_rejected():
    ms = MigrationState(2, np.array([0, 0]))
    with pytest.raises(ValueError):
        ms.to_phase(Phase.QUALITY_COMPLETE)


def test_deprioritize_cycle():
    ms = MigrationState(3, np.array([0, 0, 0]))
    ms.mark_migrated(np.array([0]))
    ms.deprioritize(0)
    assert ms.part_state[0] == PartitionState.DUAL_DEPRIORITIZED
    ms.reprioritize(0)
    assert ms.part_state[0] == PartitionState.DUAL
