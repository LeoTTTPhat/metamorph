import numpy as np
import pytest

from metamorph.synth import make_synthetic, random_semi_orthogonal
from metamorph.utils import l2_normalize
from metamorph.warmstart import (KnnGraph, beam_search, exact_knn, lloyd_kmeans,
                                 nn_descent, warm_kmeans_init)


@pytest.fixture(scope="module")
def spaces():
    data = make_synthetic(n_docs=500, n_queries=20, d_old=24, d_new=32,
                          noise=0.1, seed=5)
    return data


def test_exact_knn_no_self():
    rng = np.random.default_rng(0)
    X = l2_normalize(rng.normal(size=(100, 8)))
    nn = exact_knn(X, 5)
    for i in range(100):
        assert i not in nn[i]


def test_warm_init_beats_random_init(spaces):
    X_old, X_new = spaces.old, spaces.new
    k = 10
    true_nn = exact_knn(X_new, k)
    old_nn = exact_knn(X_old, k)
    rng = np.random.default_rng(0)
    cold = KnnGraph.random_init(X_new, k, rng)
    warm = KnnGraph.from_old_graph(X_new, old_nn)
    assert warm.recall_vs_truth(true_nn) > cold.recall_vs_truth(true_nn) + 0.2


def test_nn_descent_improves_and_warm_needs_less_work(spaces):
    X_old, X_new = spaces.old, spaces.new
    k = 10
    true_nn = exact_knn(X_new, k)
    old_nn = exact_knn(X_old, k)

    def run(graph_init, seed):
        counter = [0]
        rec = []
        nn_descent(X_new, graph_init, n_iters=6, rng=np.random.default_rng(seed),
                   counter=counter,
                   callback=lambda it, g: rec.append((counter[0], g.recall_vs_truth(true_nn))))
        return rec

    rng = np.random.default_rng(0)
    cold_curve = run(KnnGraph.random_init(X_new, k, rng), 0)
    warm_curve = run(KnnGraph.from_old_graph(X_new, old_nn), 0)
    assert warm_curve[-1][1] > 0.8               # converges to a good graph
    assert warm_curve[0][1] > cold_curve[0][1]   # warm start ahead early
    # distance comps to reach recall 0.8: warm start needs fewer
    def comps_to(curve, r):
        for c, rec in curve:
            if rec >= r:
                return c
        return float("inf")
    assert comps_to(warm_curve, 0.8) < comps_to(cold_curve, 0.8)


def test_beam_search_finds_neighbors(spaces):
    X_new = spaces.new
    k = 10
    true_nn = exact_knn(X_new, k)
    graph = KnnGraph.from_old_graph(X_new, exact_knn(spaces.old, k))
    nn_descent(X_new, graph, n_iters=5)
    q = X_new[3]
    found = beam_search(X_new, graph, q, k, ef=64, rng=np.random.default_rng(0))
    truth = np.concatenate([[3], true_nn[3]])[:k]
    assert np.intersect1d(found, truth).size / k >= 0.6


def test_warm_kmeans_converges_faster(spaces):
    X_old, X_new = spaces.old, spaces.new
    k = 8
    from sklearn.cluster import KMeans
    old_assign = KMeans(n_clusters=k, random_state=0, n_init=3).fit_predict(X_old)
    warm_C = warm_kmeans_init(X_new, old_assign, k)
    rng = np.random.default_rng(0)
    rand_C = X_new[rng.choice(X_new.shape[0], size=k, replace=False)]
    warm = lloyd_kmeans(X_new, warm_C)
    cold = lloyd_kmeans(X_new, rand_C)
    assert warm.n_iter <= cold.n_iter
    assert warm.inertia <= cold.inertia * 1.05
