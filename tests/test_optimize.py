"""Tests for quantedge.optimize.

Three kinds of evidence that the optimiser is right:
1. Closed-form answers (2-asset min-variance, inverse-variance, tangency portfolio).
2. Brute force: no point on a fine grid of allowed weights beats the optimiser.
3. Constraint checks on many random problems.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeResult

import quantedge.optimize as opt
from quantedge.exceptions import OptimizationError
from quantedge.metrics import annual_arithmetic_risk_free, annualized_volatility, sharpe_ratio, simple_returns
from quantedge.optimize import (
    WEIGHT_TOLERANCE,
    achievable_return_range,
    historical_inputs,
    max_sharpe,
    min_variance,
    min_variance_for_target,
    portfolio_stats,
)

TOL = 1e-6  # tolerance for comparing optimiser output with exact answers


def cov_from(vols, corr) -> np.ndarray:
    vols = np.asarray(vols, dtype=float)
    return np.asarray(corr, dtype=float) * np.outer(vols, vols)


def frame(matrix, names) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=names, columns=names)


def random_problem(seed: int, n: int = 6) -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    names = [f"A{i}" for i in range(n)]
    factors = rng.normal(size=(n, n))
    cov = 0.02 * factors @ factors.T / n + np.diag(rng.uniform(0.01, 0.05, n))
    mu = pd.Series(rng.uniform(0.02, 0.15, n), index=names)
    return mu, frame(cov, names)


def assert_feasible(weights: pd.Series, cap) -> None:
    caps = cap if isinstance(cap, pd.Series) else pd.Series(cap, index=weights.index)
    assert weights.sum() == pytest.approx(1.0, abs=WEIGHT_TOLERANCE)
    assert (weights >= 0).all()
    assert (weights <= caps.reindex(weights.index) + WEIGHT_TOLERANCE).all()


# --------------------------------------------------------------------------- #
# Closed-form answers
# --------------------------------------------------------------------------- #
def test_two_asset_min_variance_matches_formula():
    # w1 = (s2^2 - s12) / (s1^2 + s2^2 - 2 s12) = (0.09 - 0.012) / (0.04 + 0.09 - 0.024)
    cov = frame(cov_from([0.2, 0.3], [[1, 0.2], [0.2, 1]]), ["X", "Y"])
    w = min_variance(cov).weights
    assert w["X"] == pytest.approx(0.078 / 0.106, abs=TOL)
    assert w["Y"] == pytest.approx(1 - 0.078 / 0.106, abs=TOL)


def test_uncorrelated_min_variance_is_inverse_variance():
    vols = np.array([0.1, 0.2, 0.4])
    cov = frame(np.diag(vols**2), ["A", "B", "C"])
    expected = (1 / vols**2) / (1 / vols**2).sum()
    np.testing.assert_allclose(min_variance(cov).weights, expected, atol=TOL)


def test_unconstrained_tangency_portfolio_matches_formula():
    # When every weight of Sigma^-1 (mu - rf) is positive, long-only constraints don't bind
    # and the max-Sharpe weights are Sigma^-1 (mu - rf), normalised to sum to 1.
    names = ["A", "B", "C"]
    sigma = cov_from([0.15, 0.20, 0.25], [[1, 0.3, 0.3], [0.3, 1, 0.3], [0.3, 0.3, 1]])
    mu = pd.Series([0.08, 0.10, 0.12], index=names)
    rf = 0.02
    raw = np.linalg.solve(sigma, mu.to_numpy() - annual_arithmetic_risk_free(rf))
    assert (raw > 0).all()  # precondition for the formula to apply
    result = max_sharpe(mu, frame(sigma, names), rf_annual=rf)
    np.testing.assert_allclose(result.weights, raw / raw.sum(), atol=1e-5)


def test_cap_binds_when_unconstrained_answer_exceeds_it():
    vols = np.array([0.1, 0.2, 0.4])  # unconstrained GMV puts ~76 % in A
    cov = frame(np.diag(vols**2), ["A", "B", "C"])
    w = min_variance(cov, max_weight=0.5).weights
    assert w["A"] == pytest.approx(0.5, abs=TOL)
    # Remaining 50 % is split between B and C in inverse-variance proportion (4:1)
    assert w["B"] / w["C"] == pytest.approx(4.0, rel=1e-4)


# --------------------------------------------------------------------------- #
# Brute force on a grid of weights
# --------------------------------------------------------------------------- #
def grid_weights(n_assets: int, step: float, cap: float):
    ticks = np.round(np.arange(0, cap + 1e-12, step), 10)
    for combo in itertools.product(ticks, repeat=n_assets - 1):
        last = 1 - sum(combo)
        if -1e-12 <= last <= cap + 1e-12:
            yield np.array([*combo, max(last, 0.0)])


@pytest.mark.parametrize("cap", [1.0, 0.6, 0.4])
def test_no_grid_portfolio_beats_the_optimisers(cap):
    names = ["A", "B", "C"]
    sigma = cov_from([0.18, 0.25, 0.30], [[1, 0.5, -0.2], [0.5, 1, 0.1], [-0.2, 0.1, 1]])
    mu = pd.Series([0.07, 0.11, 0.09], index=names)
    rf = 0.03
    excess = mu.to_numpy() - annual_arithmetic_risk_free(rf)
    grid = list(grid_weights(3, 0.005, cap))
    best_sharpe = max((w @ excess) / math.sqrt(w @ sigma @ w) for w in grid)
    lowest_var = min(w @ sigma @ w for w in grid)

    ms = max_sharpe(mu, frame(sigma, names), rf_annual=rf, max_weight=cap)
    mv = min_variance(frame(sigma, names), max_weight=cap)
    assert ms.sharpe >= best_sharpe - 1e-9
    assert mv.volatility**2 <= lowest_var + 1e-12
    assert_feasible(ms.weights, cap)
    assert_feasible(mv.weights, cap)


# --------------------------------------------------------------------------- #
# Constraints on random problems
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("cap", [1.0, 0.35, 0.2])
def test_constraints_hold_on_random_problems(seed, cap):
    mu, cov = random_problem(seed)
    for result in (min_variance(cov, mu=mu, max_weight=cap),
                   max_sharpe(mu, cov, rf_annual=0.02, max_weight=cap)):
        assert_feasible(result.weights, cap)
        assert list(result.weights.index) == list(cov.index)


def test_max_sharpe_is_at_least_as_good_as_min_variance_and_equal_weight():
    mu, cov = random_problem(42)
    ms = max_sharpe(mu, cov, rf_annual=0.02, max_weight=0.4)
    mv = min_variance(cov, mu=mu, max_weight=0.4, rf_annual=0.02)
    ew = portfolio_stats(pd.Series(1 / len(mu), index=mu.index), mu, cov, 0.02)
    assert ms.sharpe >= mv.sharpe - 1e-9
    assert ms.sharpe >= ew.sharpe - 1e-9


def test_per_asset_caps():
    mu, cov = random_problem(3, n=4)
    caps = pd.Series({"A0": 0.1, "A1": 0.5, "A2": 0.5, "A3": 0.0})  # A3 excluded entirely
    result = max_sharpe(mu, cov, max_weight=caps)
    assert_feasible(result.weights, caps)
    assert result.weights["A3"] == 0.0


def test_results_are_deterministic():
    mu, cov = random_problem(5)
    first = max_sharpe(mu, cov, rf_annual=0.01, max_weight=0.3).weights
    second = max_sharpe(mu, cov, rf_annual=0.01, max_weight=0.3).weights
    pd.testing.assert_series_equal(first, second)


# --------------------------------------------------------------------------- #
# Consistency with metrics.py (units)
# --------------------------------------------------------------------------- #
def test_ex_ante_stats_match_metrics_on_the_same_history(synthetic_prices):
    # With historical mu and Sigma, the optimiser's Sharpe/vol must equal what metrics.py
    # computes from the (daily-rebalanced) portfolio's own return series. This proves
    # both modules use identical annualisation and risk-free conventions.
    returns = simple_returns(synthetic_prices)
    mu, cov = historical_inputs(returns)
    result = max_sharpe(mu, cov, rf_annual=0.05, max_weight=0.6)
    portfolio = returns @ result.weights
    assert result.sharpe == pytest.approx(sharpe_ratio(portfolio, rf_annual=0.05), rel=1e-10)
    assert result.volatility == pytest.approx(annualized_volatility(portfolio), rel=1e-10)


# --------------------------------------------------------------------------- #
# Feasibility and input validation
# --------------------------------------------------------------------------- #
def test_infeasible_caps_raise_before_solving():
    mu, cov = random_problem(1, n=3)
    with pytest.raises(OptimizationError, match="Infeasible.*33.3%"):
        min_variance(cov, max_weight=0.3)


@pytest.mark.parametrize("cap", [0.0, -0.1, 1.5, math.nan])
def test_invalid_cap_values(cap):
    _, cov = random_problem(1, n=3)
    with pytest.raises(OptimizationError, match="Maximum weight"):
        min_variance(cov, max_weight=cap)


def test_single_asset():
    cov = frame([[0.04]], ["ONLY"])
    assert min_variance(cov).weights["ONLY"] == pytest.approx(1.0)
    with pytest.raises(OptimizationError, match="Infeasible"):
        min_variance(cov, max_weight=0.9)


def test_non_psd_covariance_rejected():
    bad = frame([[0.04, 0.09], [0.09, 0.04]], ["A", "B"])  # implies correlation 2.25
    with pytest.raises(OptimizationError, match="positive semi-definite"):
        min_variance(bad)


def test_asymmetric_covariance_rejected():
    with pytest.raises(OptimizationError, match="symmetric"):
        min_variance(frame([[0.04, 0.01], [0.02, 0.09]], ["A", "B"]))


def test_nan_inputs_rejected():
    mu, cov = random_problem(2, n=3)
    bad_cov = cov.copy()
    bad_cov.iloc[0, 1] = bad_cov.iloc[1, 0] = np.nan
    with pytest.raises(OptimizationError, match="NaN"):
        min_variance(bad_cov)
    with pytest.raises(OptimizationError, match="NaN"):
        max_sharpe(mu.where(mu.index != "A1"), cov)


def test_mismatched_labels_rejected_and_order_is_irrelevant():
    mu, cov = random_problem(4, n=3)
    with pytest.raises(OptimizationError, match="same assets"):
        max_sharpe(mu.rename({"A0": "ZZZ"}), cov)
    reordered = max_sharpe(mu.iloc[::-1], cov).weights
    pd.testing.assert_series_equal(reordered, max_sharpe(mu, cov).weights)


# --------------------------------------------------------------------------- #
# Zero volatility and degenerate cases
# --------------------------------------------------------------------------- #
def test_min_variance_with_cash_like_asset_uses_its_cap():
    cov = frame(np.diag([0.0, 0.04, 0.09]), ["CASH", "B", "C"])
    result = min_variance(cov, max_weight=0.5)
    assert result.weights["CASH"] == pytest.approx(0.5, abs=TOL)
    assert_feasible(result.weights, 0.5)


def test_min_variance_zero_risk_portfolio_reports_undefined_sharpe():
    cov = frame(np.diag([0.0, 0.04]), ["CASH", "B"])
    mu = pd.Series({"CASH": 0.03, "B": 0.08})
    result = min_variance(cov, mu=mu)
    assert result.weights["CASH"] == pytest.approx(1.0, abs=TOL)
    assert math.isnan(result.sharpe)
    assert any("zero volatility" in n for n in result.notes)


def test_max_sharpe_rejects_zero_volatility_asset():
    cov = frame(np.diag([0.0, 0.04, 0.09]), ["CASH", "B", "C"])
    mu = pd.Series({"CASH": 0.03, "B": 0.08, "C": 0.10})
    with pytest.raises(OptimizationError, match="zero-volatility asset.*CASH"):
        max_sharpe(mu, cov)


def test_max_sharpe_rejects_perfect_hedge():
    # Two assets with correlation -1 and equal volatility combine to zero risk.
    cov = frame(cov_from([0.2, 0.2], [[1, -1], [-1, 1]]), ["L", "S"])
    mu = pd.Series({"L": 0.08, "S": 0.06})
    with pytest.raises(OptimizationError, match="zero-risk combination"):
        max_sharpe(mu, cov, rf_annual=0.01)


def test_max_sharpe_when_nothing_beats_risk_free_is_flagged():
    mu, cov = random_problem(6, n=4)
    result = max_sharpe(mu * 0 + 0.01, cov, rf_annual=0.08)
    assert result.sharpe < 0
    assert any("non-positive Sharpe" in n for n in result.notes)
    assert_feasible(result.weights, 1.0)


# --------------------------------------------------------------------------- #
# Target return
# --------------------------------------------------------------------------- #
def test_target_return_is_hit_exactly():
    mu, cov = random_problem(7)
    lo, hi = achievable_return_range(mu, max_weight=0.4)
    for target in np.linspace(lo, hi, 7):
        result = min_variance_for_target(mu, cov, target, max_weight=0.4)
        assert result.expected_return == pytest.approx(target, abs=1e-8)
        assert_feasible(result.weights, 0.4)


def test_achievable_range_is_exact():
    mu = pd.Series({"A": 0.05, "B": 0.10, "C": 0.20})
    # cap 40 %: max = 0.4*0.20 + 0.4*0.10 + 0.2*0.05 ; min = 0.4*0.05 + 0.4*0.10 + 0.2*0.20
    assert achievable_return_range(mu, max_weight=0.4) == pytest.approx((0.10, 0.13))


def test_target_outside_range_raises():
    mu, cov = random_problem(7)
    _, hi = achievable_return_range(mu, max_weight=0.4)
    with pytest.raises(OptimizationError, match="outside the achievable range"):
        min_variance_for_target(mu, cov, hi + 0.01, max_weight=0.4)


# --------------------------------------------------------------------------- #
# Solver failures are surfaced, not hidden
# --------------------------------------------------------------------------- #
def test_solver_failure_raises_with_scipy_message(monkeypatch):
    def failing(*args, **kwargs):
        return OptimizeResult(x=args[1], success=False, message="Iteration limit reached", nit=1000)

    monkeypatch.setattr(opt, "minimize", failing)
    _, cov = random_problem(0, n=3)
    with pytest.raises(OptimizationError, match="Iteration limit reached"):
        min_variance(cov)


def test_solver_claiming_success_with_bad_weights_is_caught(monkeypatch):
    def lying(*args, **kwargs):
        return OptimizeResult(x=np.array([0.7, 0.7, -0.4]), success=True, message="ok", nit=3)

    monkeypatch.setattr(opt, "minimize", lying)
    _, cov = random_problem(0, n=3)
    with pytest.raises(OptimizationError, match="violates constraints"):
        min_variance(cov)
