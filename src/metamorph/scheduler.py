"""Prioritized re-embedding scheduler.

Signals (per doc):
  traffic(i)   — exponentially decayed count of appearances in served candidates
  hardness(i)  — decayed sum of 1/(margin+eps) over queries that retrieved i
  dis(i)       — adapter-ensemble disagreement (corpus-wide, no NEW-model calls)
  cert_boost(p)— multiplicative boost for partitions flagged by the certifier

prio(i) = (w_t*traffic~ + w_h*hardness~ + w_d*dis~ + w_0) * (1 + boost(part(i)))
with ~ = rank-normalization over unmigrated docs. w_0 > 0 -> no starvation.
"""
from __future__ import annotations

import numpy as np

from .config import SchedulerConfig
from .utils import rank_normalize


class Scheduler:
    def __init__(self, n_docs: int, part_of: np.ndarray, cfg: SchedulerConfig,
                 rng: np.random.Generator):
        self.cfg = cfg
        self.n = n_docs
        self.part_of = np.asarray(part_of, dtype=np.int32)
        n_parts = int(self.part_of.max()) + 1 if n_docs else 0
        self.traffic = np.zeros(n_docs, dtype=np.float32)
        self.hardness = np.zeros(n_docs, dtype=np.float32)
        self.dis = np.zeros(n_docs, dtype=np.float32)
        self.boost = np.zeros(n_parts, dtype=np.float32)
        self.rng = rng

    # -- signal updates -----------------------------------------------------
    def observe_query(self, candidate_ids: np.ndarray, margin: float) -> None:
        if candidate_ids.size == 0:
            return
        self.traffic[candidate_ids] += 1.0
        self.hardness[candidate_ids] += 1.0 / (max(margin, 0.0) + 1e-3)

    def decay(self) -> None:
        self.traffic *= self.cfg.decay
        self.hardness *= self.cfg.decay

    def set_disagreement(self, dis: np.ndarray) -> None:
        self.dis = np.asarray(dis, dtype=np.float32)

    def set_cert_boost(self, boost_by_partition: np.ndarray) -> None:
        self.boost = np.asarray(boost_by_partition, dtype=np.float32)

    # -- priorities ---------------------------------------------------------
    def priorities(self, unmigrated: np.ndarray) -> np.ndarray:
        """Priority scores for the given unmigrated doc ids.

        Adapter disagreement is a *cold-start* prior, not a steady-state signal:
        it is the only non-uniform ordering available before any traffic has
        been observed, and it costs no new-model calls, but once queries have
        touched a document traffic strictly dominates it (measured in exp2).
        Its weight is therefore annealed by the fraction of still-unmigrated
        documents that traffic has already reached, so the mixture starts as
        pure disagreement and ends as pure traffic without a tuned schedule.
        """
        c = self.cfg
        traf = self.traffic[unmigrated]
        dis_term = rank_normalize(self.dis[unmigrated])
        w_dis = c.w_disagreement
        if c.anneal_disagreement == "per_doc":
            # the principle applied where it belongs: disagreement is a prior
            # for documents traffic has not reached YET, so gate it per
            # document rather than globally. Parameter-free, and it does not
            # dilute the ordering of the documents traffic has spoken about.
            dis_term = np.where(traf > 0, 0.0, dis_term).astype(np.float32)
        elif c.anneal_disagreement:
            # global form: decay by corpus-wide traffic coverage. Too slow when
            # the hot set is a small fraction of a large corpus (exp2).
            coverage = float((traf > 0).mean()) if traf.size else 0.0
            w_dis = c.w_disagreement * (1.0 - coverage)
        p = (c.w_traffic * rank_normalize(traf)
             + c.w_hardness * rank_normalize(self.hardness[unmigrated])
             + w_dis * dis_term
             + c.w_base)
        p = p * (1.0 + c.cert_boost * self.boost[self.part_of[unmigrated]])
        return p.astype(np.float32)

    def next_batch(self, migrated_mask: np.ndarray, batch_size: int) -> np.ndarray:
        unmigrated = np.where(~migrated_mask)[0]
        if unmigrated.size == 0:
            return unmigrated
        prio = self.priorities(unmigrated)
        if self.cfg.partition_major:
            # rank partitions by mean member priority; take whole partitions first
            parts = self.part_of[unmigrated]
            order = np.argsort(-np.bincount(parts, weights=prio, minlength=self.boost.size + 1)
                               / np.maximum(np.bincount(parts, minlength=self.boost.size + 1), 1))
            chosen: list[np.ndarray] = []
            remaining = batch_size
            for p in order:
                members = unmigrated[parts == p]
                if members.size == 0 or remaining <= 0:
                    continue
                mp = prio[parts == p]
                take = members[np.argsort(-mp, kind="stable")[:remaining]]
                chosen.append(take)
                remaining -= take.size
            return np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)
        # doc-major: global top-B with random tie-breaking for reproducible
        # fairness. argpartition (O(N)) rather than a full sort: at 10^6-10^7
        # unmigrated docs a full argsort per step dominates the step cost.
        jitter = self.rng.random(unmigrated.size).astype(np.float32) * 1e-6
        keyed = -(prio + jitter)
        b = min(batch_size, unmigrated.size)
        if b >= unmigrated.size:
            order = np.argsort(keyed, kind="stable")
        else:
            part = np.argpartition(keyed, b - 1)[:b]
            order = part[np.argsort(keyed[part], kind="stable")]
        return unmigrated[order]


class StaticPriorScheduler(Scheduler):
    """Baseline B5: order by a traffic prior computed OFFLINE from a historical
    query log, then frozen — no online adaptation.

    This is the strongest ordering a deployment can build without touching the
    serving path: last month's logs, replayed against the existing old-space
    index, cost no new-model calls and need no instrumentation. It isolates
    what the *online* half of the scheduler actually buys.
    """

    def observe_query(self, candidate_ids: np.ndarray, margin: float) -> None:
        return  # frozen prior: the live stream does not update priorities

    def decay(self) -> None:
        return


class RandomScheduler(Scheduler):
    """Baseline B2: uniformly random order, same machinery otherwise."""

    def next_batch(self, migrated_mask: np.ndarray, batch_size: int) -> np.ndarray:
        unmigrated = np.where(~migrated_mask)[0]
        if unmigrated.size == 0:
            return unmigrated
        take = min(batch_size, unmigrated.size)
        return self.rng.choice(unmigrated, size=take, replace=False)
