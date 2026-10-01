"""Configuration dataclasses for the migration engine and experiments."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AdapterConfig:
    kinds: tuple[str, ...] = ("procrustes", "ridge", "mlp")  # ensemble members
    primary: str = "auto"           # serving adapter; "auto" = held-out selection
    ridge_lambda: float = 1.0
    mlp_hidden: int = 512
    mlp_epochs: int = 100
    mlp_lr: float = 1e-3
    mlp_batch: int = 128
    train_reverse: bool = True      # also fit g: new -> old


@dataclass
class SchedulerConfig:
    w_traffic: float = 1.0
    w_hardness: float = 0.5
    w_disagreement: float = 1.0
    w_base: float = 0.05            # starvation-free floor
    decay: float = 0.99             # per-step EMA decay of traffic/hardness
    # how the cold-start disagreement prior is retired: True = decay the weight
    # by corpus-wide traffic coverage; "per_doc" = drop it for any document
    # traffic has already reached; False = never (static mixture)
    anneal_disagreement: bool | str = True
    partition_major: bool = False   # complete whole partitions first
    cert_boost: float = 1.0         # multiplier added for cert-flagged partitions


@dataclass
class CertifierConfig:
    enabled: bool = True
    sample_queries: int = 100       # m: max queries per round (cost cap may stop earlier)
    pool_extra: int = 20            # C: adapter-space extra candidates in the pool
    delta: float = 0.05             # 1-delta confidence
    target_recall: float = 0.90     # tau: cutover threshold on the pooled lower bound
    period_steps: int = 5           # certify every this many scheduler steps
    min_budget_frac: float = 0.10   # no rounds before this budget (cost gate)
    max_round_cost_frac: float = 0.01  # reveal budget per round, fraction of N
    history_window: int = 0         # rounds pooled for the LB (0 = the whole
                                    # stream, which the anytime bound allows)
    recycle: bool = True            # revealed docs count as migrated
    bound: str = "betting"          # "betting" (anytime-valid) | "eb" (fixed-sample)
    metric: str = "recall"          # certification target: "recall" | "score_mass"
    probe_frac: float = 1.0         # every certification query also yields a
                                    # truncation-bias sample: the bias bound is
                                    # sample-count limited, and a cheap probe on
                                    # every query beats a deep probe on a few
    probe_multiplier: int = 2       # deeper pool = probe_multiplier * pool_extra
    suffix_bound: bool = True       # tightest valid bound over all suffixes of
                                    # the round sequence (union-bounded), rather
                                    # than one bound over the whole stream
    min_probe: int = 30             # no certification decision until the
                                    # truncation probe can bound its own bias


@dataclass
class ServeConfig:
    k: int = 10
    overfetch: int = 4              # search K = overfetch * k per space
    query_mode: str = "dual"        # "dual" (embed query w/ both models) | "new_only"
    verify_per_query: int = 0       # v: true re-embeds of top unmigrated candidates
    ef_search: int = 128
    eps_quantile: float = 0.95      # 1-alpha for the split-conformal residual
                                    # quantile that bounds the fusion error


@dataclass
class EngineConfig:
    seed: int = 42
    seed_fraction: float = 0.02     # docs re-embedded up front to train adapters
    batch_size: int = 200           # docs migrated per scheduler step
    n_partitions: int = 16
    workload_per_step: int = 32     # queries processed between scheduler steps
    max_budget_fraction: float = 1.0
    eval_every_steps: int = 1
    hnsw_M: int = 16
    hnsw_ef_construction: int = 200
    # Old-index compaction: rebuild the old-space index over the surviving
    # unmigrated documents once they fall below this fraction of the size at
    # the last build (0 disables). Masked ANN search degrades as the live set
    # shrinks -- the overfetch escalates -- and geometric rebuilds bound the
    # total rebuild work at O(N) inserts while keeping search on a dense index.
    compact_old_index_ratio: float = 0.5
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    certifier: CertifierConfig = field(default_factory=CertifierConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)
