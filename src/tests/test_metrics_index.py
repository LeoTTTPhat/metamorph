import numpy as np

from metamorph.index import BruteForceIndex, HnswIndex
from metamorph.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from metamorph.utils import l2_normalize, rank_normalize, topk_indices


def test_recall_at_k():
    assert recall_at_k(np.array([1, 2, 3]), np.array([2, 3, 4]), 3) == 2 / 3
    assert recall_at_k(np.array([1]), np.array([]), 5) == 1.0


def test_ndcg_perfect_and_worst():
    rels = {1: 1.0, 2: 1.0}
    assert ndcg_at_k(np.array([1, 2, 9]), rels, 3) == 1.0
    assert ndcg_at_k(np.array([7, 8, 9]), rels, 3) == 0.0


def test_mrr():
    assert mrr_at_k(np.array([5, 1, 2]), {1: 1.0}, 10) == 0.5
    assert mrr_at_k(np.array([5, 6]), {1: 1.0}, 10) == 0.0


def test_topk_and_ranknorm():
    s = np.array([0.1, 0.9, 0.5])
    assert topk_indices(s, 2).tolist() == [1, 2]
    rn = rank_normalize(np.array([10.0, 30.0, 20.0]))
    assert rn.tolist() == [0.0, 1.0, 0.5]
    assert rank_normalize(np.ones(5)).max() == 0.0


def test_hnsw_matches_bruteforce():
    rng = np.random.default_rng(0)
    X = l2_normalize(rng.normal(size=(500, 24)))
    q = l2_normalize(rng.normal(size=24))
    bf_ids, _ = BruteForceIndex(X).search(q, 10)
    idx = HnswIndex(24, 500, ef_search=200)
    idx.add(X, np.arange(500))
    hn_ids, hn_scores = idx.search(q, 10)
    assert recall_at_k(hn_ids, bf_ids, 10) >= 0.9
    assert np.all(np.diff(hn_scores) <= 1e-6)  # descending similarity


def test_hnsw_masked_search_excludes_ids():
    rng = np.random.default_rng(1)
    X = l2_normalize(rng.normal(size=(300, 16)))
    idx = HnswIndex(16, 300, ef_search=200)
    idx.add(X, np.arange(300))
    mask = np.ones(300, dtype=bool)
    q = X[5]
    banned = BruteForceIndex(X).search(q, 5)[0]
    mask[banned] = False
    ids, _ = idx.search(q, 10, mask=mask)
    assert ids.size == 10
    assert not np.isin(ids, banned).any()


def test_incremental_add():
    rng = np.random.default_rng(2)
    X = l2_normalize(rng.normal(size=(100, 8)))
    idx = HnswIndex(8, 100)
    assert idx.search(X[0], 5)[0].size == 0  # empty index
    idx.add(X[:50], np.arange(50))
    idx.add(X[50:], np.arange(50, 100))
    ids, _ = idx.search(X[7], 1)
    assert ids[0] == 7
