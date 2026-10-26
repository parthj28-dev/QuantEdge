"""Tests for quantedge.frontier."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import quantedge.frontier as frontier_module
from quantedge.exceptions import OptimizationError
from quantedge.frontier import efficient_frontier
from quantedge.optimize import WEIGHT_TOLERANCE, achievable_return_range, max_sharpe, min_variance


def make_problem(seed: int = 0, n: int = 5):
    rng = np.random.default_rng(seed)
    names = [f"A{i}" for i in range(n)]
    factors = rng.normal(size=(n, n))
    cov = 0.02 * factors @ factors.T / n + np.diag(rng.uniform(0.01, 0.05, n))
    return pd.Series(rng.uniform(0.03, 0.15, n), index=names), pd.DataFrame(cov, index=names, columns=names)


@pytest.fixture(params=[1.0, 0.4])
def capped_frontier(request):
    mu, cov = make_problem()
    return mu, cov, request.param, efficient_frontier(mu, cov, n_points=20, max_weight=request.param,
                                                      rf_annual=0.02)


def test_frontier_spans_gmv_to_max_return(capped_frontier):
    mu, cov, cap, result = capped_frontier
    gmv = min_variance(cov, mu=mu, max_weight=cap)
    _, r_max = achievable_return_range(mu, max_weight=cap)
    assert len(result.points) == 20 and result.failed_targets == ()
    assert result.points["expected_return"].iloc[0] == pytest.approx(gmv.expected_return)
    assert result.points["volatility"].iloc[0] == pytest.approx(gmv.volatility, rel=1e-6)
    assert result.points["expected_return"].iloc[-1] == pytest.approx(r_max, abs=1e-8)
    np.testing.assert_allclose(result.points["expected_return"], result.points["target_return"], atol=1e-8)


def test_frontier_is_increasing_in_both_risk_and_return(capped_frontier):
    *_, result = capped_frontier
    assert (np.diff(result.points["expected_return"]) > 0).all()
    assert (np.diff(result.points["volatility"]) >= -1e-9).all()
    assert not any("Warning" in n for n in result.notes)


def test_every_frontier_portfolio_is_feasible(capped_frontier):
    _, cov, cap, result = capped_frontier
    w = result.weights
    assert list(w.columns) == list(cov.index)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=WEIGHT_TOLERANCE)
    assert (w.to_numpy() >= 0).all()
    assert (w.to_numpy() <= cap + WEIGHT_TOLERANCE).all()


def test_max_sharpe_dominates_the_frontier(capped_frontier):
    # The tangency portfolio is the frontier point with the best Sharpe ratio, so no
    # traced point may beat it, and the best traced point should be close to it.
    mu, cov, cap, result = capped_frontier
    tangency = max_sharpe(mu, cov, rf_annual=0.02, max_weight=cap)
    assert result.points["sharpe"].max() <= tangency.sharpe + 1e-9
    assert result.points["sharpe"].max() >= tangency.sharpe - 0.05
    # And the tangency portfolio is itself (to solver accuracy) on the frontier:
    frontier_vol = np.interp(tangency.expected_return, result.points["expected_return"],
                             result.points["volatility"])
    assert tangency.volatility <= frontier_vol + 1e-6


def test_two_asset_frontier_matches_closed_form():
    # With two assets the weight is pinned by the target: w = (R - mu_B) / (mu_A - mu_B).
    mu = pd.Series({"A": 0.12, "B": 0.06})
    s_a, s_b, rho = 0.25, 0.15, 0.3
    cov = pd.DataFrame([[s_a**2, rho * s_a * s_b], [rho * s_a * s_b, s_b**2]], index=["A", "B"],
                       columns=["A", "B"])
    result = efficient_frontier(mu, cov, n_points=10)
    for r, vol in zip(result.points["expected_return"], result.points["volatility"]):
        w = (r - 0.06) / 0.06
        expected = math.sqrt(w**2 * s_a**2 + (1 - w) ** 2 * s_b**2 + 2 * w * (1 - w) * rho * s_a * s_b)
        assert vol == pytest.approx(expected, rel=1e-6)


def test_single_point_frontier_when_all_returns_are_equal():
    # Every portfolio has the same expected return, so only the least risky one is efficient.
    equal = pd.Series({"A": 0.08, "B": 0.08})
    cov2 = pd.DataFrame(np.diag([0.04, 0.09]), index=equal.index, columns=equal.index)
    flat = efficient_frontier(equal, cov2, n_points=10)
    assert len(flat.points) == 1
    assert any("single point" in n for n in flat.notes)


def test_failed_points_are_reported_not_hidden(monkeypatch):
    mu, cov = make_problem()
    real = frontier_module.min_variance_for_target
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OptimizationError("simulated failure")
        return real(*args, **kwargs)

    monkeypatch.setattr(frontier_module, "min_variance_for_target", flaky)
    result = efficient_frontier(mu, cov, n_points=10)
    assert len(result.failed_targets) == 1
    assert len(result.points) == 9
    assert any("could not be solved" in n for n in result.notes)


def test_all_points_failing_raises(monkeypatch):
    mu, cov = make_problem()

    def always_fail(*args, **kwargs):
        raise OptimizationError("simulated failure")

    monkeypatch.setattr(frontier_module, "min_variance_for_target", always_fail)
    with pytest.raises(OptimizationError, match="no target-return problem"):
        efficient_frontier(mu, cov, n_points=5)


def test_invalid_point_count():
    mu, cov = make_problem()
    with pytest.raises(ValueError):
        efficient_frontier(mu, cov, n_points=1)


def test_infeasible_caps_propagate():
    mu, cov = make_problem(n=3)
    with pytest.raises(OptimizationError, match="Infeasible"):
        efficient_frontier(mu, cov, max_weight=0.2)
