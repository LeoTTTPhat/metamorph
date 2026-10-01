import numpy as np

from metamorph.config import SchedulerConfig
from metamorph.scheduler import RandomScheduler, Scheduler


def make_sched(n=100, cfg=None, seed=0):
    part_of = np.arange(n) % 4
    rng = np.random.default_rng(seed)
    return Scheduler(n, part_of, cfg or SchedulerConfig(), rng), part_of


def test_traffic_drives_priority():
    sched, _ = make_sched()
    hot = np.array([7, 8, 9])
    for _ in range(10):
        sched.observe_query(hot, margin=0.5)
    batch = sched.next_batch(np.zeros(100, dtype=bool), 3)
    assert set(batch.tolist()) == {7, 8, 9}


def test_hardness_breaks_traffic_ties():
    cfg = SchedulerConfig(w_traffic=1.0, w_hardness=1.0, w_disagreement=0.0)
    sched, _ = make_sched(cfg=cfg)
    sched.observe_query(np.array([1]), margin=0.5)   # easy query
    sched.observe_query(np.array([2]), margin=0.001)  # hard query
    batch = sched.next_batch(np.zeros(100, dtype=bool), 1)
    assert batch[0] == 2


def test_disagreement_signal():
    cfg = SchedulerConfig(w_traffic=0.0, w_hardness=0.0, w_disagreement=1.0, w_base=0.0)
    sched, _ = make_sched(cfg=cfg)
    dis = np.zeros(100, dtype=np.float32)
    dis[42] = 1.0
    sched.set_disagreement(dis)
    assert sched.next_batch(np.zeros(100, dtype=bool), 1)[0] == 42


def test_migrated_docs_never_selected():
    sched, _ = make_sched()
    migrated = np.zeros(100, dtype=bool)
    migrated[:90] = True
    batch = sched.next_batch(migrated, 20)
    assert batch.size == 10
    assert not migrated[batch].any()


def test_cert_boost_prioritizes_partition():
    cfg = SchedulerConfig(w_traffic=0.0, w_hardness=0.0, w_disagreement=0.0,
                          w_base=1.0, cert_boost=5.0)
    sched, part_of = make_sched(cfg=cfg)
    boost = np.zeros(4, dtype=np.float32)
    boost[2] = 1.0
    sched.set_cert_boost(boost)
    batch = sched.next_batch(np.zeros(100, dtype=bool), 10)
    assert (part_of[batch] == 2).all()


def test_partition_major_completes_partitions():
    cfg = SchedulerConfig(partition_major=True, w_traffic=1.0, w_base=0.01)
    n = 40
    part_of = np.repeat(np.arange(4), 10)
    sched = Scheduler(n, part_of, cfg, np.random.default_rng(0))
    sched.observe_query(np.where(part_of == 3)[0][:5], margin=0.01)
    batch = sched.next_batch(np.zeros(n, dtype=bool), 10)
    assert (part_of[batch] == 3).all() and batch.size == 10


def test_random_scheduler_reproducible():
    part_of = np.zeros(100, dtype=np.int32)
    a = RandomScheduler(100, part_of, SchedulerConfig(), np.random.default_rng(7))
    b = RandomScheduler(100, part_of, SchedulerConfig(), np.random.default_rng(7))
    m = np.zeros(100, dtype=bool)
    assert np.array_equal(a.next_batch(m, 10), b.next_batch(m, 10))
