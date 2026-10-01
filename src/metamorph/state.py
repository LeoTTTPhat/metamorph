"""Migration state machine: per-partition serving states and global phases.

Invariants (enforced here, checked again in tests):
  I1 exactly-once serving: each doc is searchable in exactly one space
     (encoded by the boolean `migrated` mask owned by this class).
  I2 monotonicity: a migrated doc never reverts.
  I3 budget conservation: every reveal goes through EmbeddingStore's meter.
  I4 phase/partition transitions follow the declared DAGs.
"""
from __future__ import annotations

from enum import Enum

import numpy as np


class PartitionState(str, Enum):
    OLD_ONLY = "OLD_ONLY"          # no doc of the partition migrated yet
    DUAL = "DUAL"                  # some docs migrated: served across both spaces
    DUAL_DEPRIORITIZED = "DUAL_DEPRIORITIZED"  # certified adapter-adequate
    NEW_ONLY = "NEW_ONLY"          # all docs migrated: old-space serving dropped


class Phase(str, Enum):
    INIT = "INIT"
    SEEDING = "SEEDING"
    ADAPTIVE_SERVING = "ADAPTIVE_SERVING"
    QUALITY_COMPLETE = "QUALITY_COMPLETE"
    FULLY_MIGRATED = "FULLY_MIGRATED"


_PHASE_EDGES = {
    Phase.INIT: {Phase.SEEDING},
    Phase.SEEDING: {Phase.ADAPTIVE_SERVING},
    Phase.ADAPTIVE_SERVING: {Phase.QUALITY_COMPLETE, Phase.FULLY_MIGRATED},
    Phase.QUALITY_COMPLETE: {Phase.FULLY_MIGRATED},
    Phase.FULLY_MIGRATED: set(),
}

_PART_EDGES = {
    PartitionState.OLD_ONLY: {PartitionState.DUAL, PartitionState.NEW_ONLY},
    PartitionState.DUAL: {PartitionState.DUAL_DEPRIORITIZED, PartitionState.NEW_ONLY},
    PartitionState.DUAL_DEPRIORITIZED: {PartitionState.DUAL, PartitionState.NEW_ONLY},
    PartitionState.NEW_ONLY: set(),
}


class MigrationState:
    def __init__(self, n_docs: int, part_of: np.ndarray):
        self.n = n_docs
        self.part_of = np.asarray(part_of, dtype=np.int32)
        self.n_partitions = int(self.part_of.max()) + 1 if n_docs else 0
        self.migrated = np.zeros(n_docs, dtype=bool)
        self._part_migrated = np.zeros(self.n_partitions, dtype=np.int64)
        self._part_size = np.bincount(self.part_of, minlength=self.n_partitions)
        self.part_state = [PartitionState.OLD_ONLY] * self.n_partitions
        self.phase = Phase.INIT
        self.events: list[tuple[str, str]] = []

    # -- global phase -------------------------------------------------------
    def to_phase(self, phase: Phase) -> None:
        if phase == self.phase:
            return
        if phase not in _PHASE_EDGES[self.phase]:
            raise ValueError(f"illegal phase transition {self.phase} -> {phase}")
        self.events.append(("phase", f"{self.phase.value}->{phase.value}"))
        self.phase = phase

    # -- partition states ---------------------------------------------------
    def _set_part_state(self, p: int, s: PartitionState) -> None:
        cur = self.part_state[p]
        if s == cur:
            return
        if s not in _PART_EDGES[cur]:
            raise ValueError(f"illegal partition transition {cur} -> {s} (p={p})")
        self.events.append((f"part{p}", f"{cur.value}->{s.value}"))
        self.part_state[p] = s

    def mark_migrated(self, ids: np.ndarray) -> np.ndarray:
        """Flip docs to the new space; returns the ids that were newly flipped."""
        ids = np.asarray(ids, dtype=np.int64)
        fresh = ids[~self.migrated[ids]]
        fresh = np.unique(fresh)
        if fresh.size == 0:
            return fresh
        self.migrated[fresh] = True   # I2: set-only, never cleared
        for p, cnt in zip(*np.unique(self.part_of[fresh], return_counts=True)):
            self._part_migrated[p] += cnt
            if self._part_migrated[p] == self._part_size[p]:
                self._set_part_state(int(p), PartitionState.NEW_ONLY)
            elif self.part_state[p] == PartitionState.OLD_ONLY:
                self._set_part_state(int(p), PartitionState.DUAL)
        if bool(self.migrated.all()) and self.phase in (
                Phase.ADAPTIVE_SERVING, Phase.QUALITY_COMPLETE):
            self.to_phase(Phase.FULLY_MIGRATED)
        return fresh

    def deprioritize(self, p: int) -> None:
        if self.part_state[p] == PartitionState.DUAL:
            self._set_part_state(p, PartitionState.DUAL_DEPRIORITIZED)

    def reprioritize(self, p: int) -> None:
        if self.part_state[p] == PartitionState.DUAL_DEPRIORITIZED:
            self._set_part_state(p, PartitionState.DUAL)

    # -- invariant checks ---------------------------------------------------
    def check_invariants(self) -> None:
        for p in range(self.n_partitions):
            frac = self._part_migrated[p]
            assert frac == int(self.migrated[self.part_of == p].sum())
            if self.part_state[p] == PartitionState.NEW_ONLY:
                assert frac == self._part_size[p]
            if self.part_state[p] == PartitionState.OLD_ONLY:
                assert frac == 0

    @property
    def migrated_fraction(self) -> float:
        return float(self.migrated.mean()) if self.n else 0.0
