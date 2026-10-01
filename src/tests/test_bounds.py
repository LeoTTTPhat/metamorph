"""Tests for the confidence bounds used by the certifier."""
import numpy as np
import pytest

from metamorph.bounds import (betting_lb, empirical_bernstein_lb, lower_bound,
                              upper_bound_from_lb)


def _bernoulli(mu, n, seed):
    return (np.random.default_rng(seed).random(n) < mu).astype(np.float64)


def test_betting_lb_is_a_lower_bound_on_the_sample_mean():
    for seed in range(20):
        x = _bernoulli(0.9, 200, seed)
        assert 0.0 <= betting_lb(x, 0.05) <= x.mean() + 1e-9


def test_betting_lb_covers_at_the_nominal_rate():
    mu, trials = 0.9, 600
    viol = sum(betting_lb(_bernoulli(mu, 150, s), 0.05) > mu for s in range(trials))
    assert viol / trials <= 0.05


def test_betting_lb_is_anytime_valid_under_repeated_peeking():
    """The certifier looks after every round; a fixed-sample bound would not
    be licensed to do that.  Peeking at many prefixes must still respect delta."""
    mu, trials, viol = 0.9, 300, 0
    for s in range(trials):
        x = _bernoulli(mu, 400, 10_000 + s)
        if any(betting_lb(x[:n], 0.05) > mu for n in range(5, 400, 20)):
            viol += 1
    assert viol / trials <= 0.05


def test_betting_lb_beats_empirical_bernstein_in_the_certifier_regime():
    """High-mean, low-variance samples with m in the low hundreds: exactly the
    regime where the empirical-Bernstein 1/m term made the bound inoperative."""
    for mu in (0.95, 0.99):
        for m in (50, 100, 300):
            x = _bernoulli(mu, m, seed=int(1000 * mu) + m)
            assert betting_lb(x, 0.05) > empirical_bernstein_lb(x, 0.05)


def test_betting_lb_tightens_with_more_samples():
    prev = -1.0
    for m in (25, 50, 100, 400, 1600):
        lb = betting_lb(_bernoulli(0.95, m, seed=7), 0.05)
        assert lb >= prev - 0.05     # monotone up to sampling noise
        prev = lb
    assert prev > 0.90


def test_upper_bound_mirrors_lower_bound():
    x = _bernoulli(0.2, 300, seed=3)
    ub = upper_bound_from_lb(x, 0.05)
    assert x.mean() - 1e-9 <= ub <= 1.0


def test_upper_bound_covers_at_the_nominal_rate():
    mu, trials = 0.1, 400
    viol = sum(upper_bound_from_lb(_bernoulli(mu, 200, s), 0.05) < mu for s in range(trials))
    assert viol / trials <= 0.05


def test_degenerate_inputs():
    assert betting_lb(np.array([]), 0.05) == 0.0
    assert empirical_bernstein_lb(np.array([]), 0.05) == 0.0
    assert betting_lb(np.zeros(50), 0.05) == 0.0
    assert betting_lb(np.ones(50), 0.05) > 0.5


def test_lower_bound_dispatch():
    x = _bernoulli(0.8, 100, seed=1)
    assert lower_bound(x, 0.05, "eb") == empirical_bernstein_lb(x, 0.05)
    assert lower_bound(x, 0.05, "betting") == betting_lb(x, 0.05)
    with pytest.raises(ValueError):
        lower_bound(x, 0.05, "nope")


# --------------------------------------------------------------- Prop. 2'
def _lb(scores, top, eps, n_old):
    """Call the router's certificate on a synthetic candidate list."""
    from metamorph.serve import QueryRouter
    return QueryRouter._certified_recall_lb(None, np.arange(len(scores)),
                                            np.asarray(scores, float),
                                            np.asarray(top), eps, n_old)


def test_certificate_is_one_when_the_gap_is_wide():
    # served = the two clear leaders; competitors far below; small eps
    scores = np.array([0.9, 0.8, 0.1, 0.05])
    assert _lb(scores, np.array([0, 1]), eps=0.01, n_old=4) == 1.0


def test_certificate_is_zero_when_everything_is_contested():
    scores = np.array([0.50, 0.49, 0.48, 0.47])
    assert _lb(scores, np.array([0, 1]), eps=0.5, n_old=4) == 0.0


def test_certificate_is_graded_not_all_or_nothing():
    # one served item is safe, the other is contested -> exactly 0.5
    scores = np.array([0.90, 0.50, 0.49, 0.10])
    lb = _lb(scores, np.array([0, 1]), eps=0.02, n_old=4)
    assert lb == 0.5


def test_migrated_candidates_carry_no_error():
    # identical scores, but all candidates exact (n_old=0) -> eps irrelevant
    scores = np.array([0.90, 0.80, 0.10, 0.05])
    assert _lb(scores, np.array([0, 1]), eps=0.4, n_old=0) == 1.0
    # same list with everything unmigrated and a large eps: nothing provable
    assert _lb(scores, np.array([0, 1]), eps=0.4, n_old=4) < 1.0


def test_certificate_never_exceeds_truth_on_random_instances():
    rng = np.random.default_rng(0)
    for t in range(300):
        n, k = 40, 10
        n_old = int(rng.integers(0, n + 1))
        true = rng.random(n)
        err = np.zeros(n)
        eps = float(rng.uniform(0.0, 0.3))
        err[:n_old] = rng.uniform(-eps, eps, size=n_old)
        served_scores = true + err
        top = np.argsort(-served_scores)[:k]
        lb = _lb(served_scores, top, eps, n_old)
        true_top = set(np.argsort(-true)[:k].tolist())
        truth = len(true_top & set(top.tolist())) / k
        assert lb <= truth + 1e-9, (t, lb, truth)


# ------------------------------------------------------- suffix-max bound
def test_suffix_bound_is_never_far_below_the_pooled_bound():
    from metamorph.bounds import suffix_lower_bound
    chunks = [_bernoulli(0.9, 40, s) for s in range(12)]
    pooled = betting_lb(np.concatenate(chunks), 0.05)
    suffix, _ = suffix_lower_bound(chunks, 0.05)
    assert suffix >= betting_lb(np.concatenate(chunks), 0.025) - 1e-9
    assert suffix >= pooled - 0.05


def test_suffix_bound_recovers_from_a_weak_prefix():
    """Rising quality: early rounds are poor, recent rounds are good. Pooling
    everything is dragged down; the suffix bound is not."""
    from metamorph.bounds import suffix_lower_bound
    chunks = [_bernoulli(0.4, 200, s) for s in range(6)] + \
             [_bernoulli(0.99, 200, 100 + s) for s in range(6)]
    pooled = betting_lb(np.concatenate(chunks), 0.05)
    suffix, nrounds = suffix_lower_bound(chunks, 0.05)
    assert suffix > pooled + 0.2
    assert nrounds <= 6


def test_suffix_bound_covers_under_a_stationary_stream():
    from metamorph.bounds import suffix_lower_bound
    mu, trials, viol = 0.9, 300, 0
    for t in range(trials):
        chunks = [_bernoulli(mu, 25, 5000 + 20 * t + j) for j in range(10)]
        if any(suffix_lower_bound(chunks[:j + 1], 0.05)[0] > mu for j in range(10)):
            viol += 1
    assert viol / trials <= 0.05


def test_conformal_quantile_covers_a_held_out_residual():
    """The split-conformal index ceil((n+1)(1-alpha)) must cover an exchangeable
    further draw at least (1-alpha) of the time, for any residual distribution."""
    rng = np.random.default_rng(11)
    for dist in ("normal", "heavy", "uniform"):
        hits, trials = 0, 3000
        for t in range(trials):
            n = int(rng.integers(12, 60))
            if dist == "normal":
                r = np.abs(rng.normal(size=n + 1))
            elif dist == "heavy":
                r = np.abs(rng.standard_t(2, size=n + 1))
            else:
                r = rng.random(n + 1)
            cal, new = np.sort(r[:n]), r[n]
            alpha = 0.05
            rank = int(np.ceil((n + 1) * (1.0 - alpha)))
            eps = cal[-1] if rank > n else cal[rank - 1]
            hits += new <= eps
        assert hits / trials >= 0.94, (dist, hits / trials)
