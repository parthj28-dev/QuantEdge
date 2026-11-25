"""Tests for quantedge.riskbased: equal risk shares, known identities, caps, stress test."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from quantedge.exceptions import OptimizationError
from quantedge.riskbased import (
    diversification_ratio,
    inverse_volatility,
    max_diversification,
    risk_contributions,
    risk_parity,
)

NAMES = ["A", "B", "C", "D"]


def cov_frame(vols, corr, names=None) -> pd.DataFrame:
    vols = np.asarray(vols, float)
    names = names or NAMES[: len(vols)]
    return pd.DataFrame(np.asarray(corr, float) * np.outer(vols, vols), index=names, columns=names)


def equal_corr(n: int, rho: float) -> np.ndarray:
    return np.full((n, n), rho) + (1 - rho) * np.eye(n)


def random_cov(seed: int, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n + 5, n))
    cov = a.T @ a / (n + 5) * rng.uniform(0.01, 0.2)
    return pd.DataFrame(cov, index=[f"X{i}" for i in range(n)], columns=[f"X{i}" for i in range(n)])


COV = cov_frame([0.1, 0.2, 0.3, 0.25], [[1, 0.2, 0.4, 0.1], [0.2, 1, 0.3, 0.5],
                                        [0.4, 0.3, 1, 0.2], [0.1, 0.5, 0.2, 1]])


def test_risk_shares_sum_to_one_and_contributions_to_volatility():
    w = pd.Series([0.4, 0.3, 0.2, 0.1], index=NAMES)
    rc = risk_contributions(w, COV)
    assert rc["risk_share"].sum() == pytest.approx(1.0)
    vol = np.sqrt(w @ COV @ w)
    assert rc["risk_contribution"].sum() == pytest.approx(vol)


def test_risk_parity_gives_equal_shares():
    res = risk_parity(COV)
    shares = risk_contributions(res.weights, COV)["risk_share"]
    np.testing.assert_allclose(shares, 0.25, atol=1e-10)
    assert res.weights.sum() == pytest.approx(1.0)


def test_inverse_vol_formula():
    w = inverse_volatility(COV).weights
    inv = 1 / np.array([0.1, 0.2, 0.3, 0.25])
    np.testing.assert_allclose(w, inv / inv.sum())


@pytest.mark.parametrize("rho", [-0.6, 0.0, 0.3, 0.9])
def test_two_assets_risk_parity_equals_inverse_vol(rho):
    cov = cov_frame([0.1, 0.35], [[1, rho], [rho, 1]])
    np.testing.assert_allclose(risk_parity(cov).weights, inverse_volatility(cov).weights, atol=1e-10)


def test_equal_correlations_all_three_coincide():
    cov = cov_frame([0.1, 0.2, 0.3, 0.15], equal_corr(4, 0.35))
    iv = inverse_volatility(cov).weights
    np.testing.assert_allclose(risk_parity(cov).weights, iv, atol=1e-9)
    np.testing.assert_allclose(max_diversification(cov).weights, iv, atol=1e-6)


def test_uncorrelated_max_div_is_inverse_vol():
    cov = cov_frame([0.1, 0.2, 0.3], np.eye(3))
    np.testing.assert_allclose(max_diversification(cov).weights, inverse_volatility(cov).weights, atol=1e-6)


def test_max_diversification_beats_a_grid():
    cov = COV.iloc[:3, :3]
    best = diversification_ratio(max_diversification(cov).weights, cov)
    for a, b in itertools.product(np.linspace(0, 1, 41), repeat=2):
        if a + b <= 1:
            w = pd.Series([a, b, 1 - a - b], index=cov.index)
            assert diversification_ratio(w, cov) <= best + 1e-9


def test_inverse_vol_caps_are_water_filled():
    w = inverse_volatility(COV, max_weight=0.35).weights
    assert w.max() <= 0.35 + 1e-12 and w.sum() == pytest.approx(1.0)
    free = w[w < 0.35 - 1e-12]
    inv = 1 / np.sqrt(np.diag(COV.loc[free.index, free.index]))
    np.testing.assert_allclose(free / free.sum(), inv / inv.sum())


def test_capped_risk_parity_respects_caps_and_says_so():
    res = risk_parity(COV, max_weight=0.3)
    assert res.weights.max() <= 0.3                      # exactly within the cap, no rounding excess
    assert res.weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert res.notes and "caps" in res.notes[0]


def test_uncapped_risk_parity_has_no_note():
    assert risk_parity(COV).notes == ()


@pytest.mark.parametrize("fn", [inverse_volatility, risk_parity, max_diversification])
def test_zero_vol_asset_rejected(fn):
    cov = cov_frame([0.0, 0.2], [[1, 0], [0, 1]])
    with pytest.raises(OptimizationError, match="zero-volatility"):
        fn(cov)


@pytest.mark.parametrize("fn", [inverse_volatility, risk_parity, max_diversification])
def test_infeasible_caps(fn):
    with pytest.raises(OptimizationError):
        fn(COV, max_weight=0.2)


def test_risk_parity_stress_100_random_problems():
    for seed in range(100):
        cov = random_cov(seed, 2 + seed % 12)
        res = risk_parity(cov)
        shares = risk_contributions(res.weights, cov)["risk_share"]
        np.testing.assert_allclose(shares, 1 / len(cov), atol=1e-8, err_msg=f"seed {seed}")


def test_expected_return_reported_when_mu_given():
    mu = pd.Series([0.05, 0.08, 0.12, 0.10], index=NAMES)
    res = risk_parity(COV, mu=mu, rf_annual=0.0)
    assert res.expected_return == pytest.approx(float(res.weights @ mu))
    assert np.isnan(risk_parity(COV).expected_return)


def test_diversification_ratio_of_single_asset_is_one():
    w = pd.Series([1.0, 0, 0, 0], index=NAMES)
    assert diversification_ratio(w, COV) == pytest.approx(1.0)


def test_capped_simplex_projection_is_exactly_feasible():
    from quantedge.riskbased import _project_capped_simplex
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = rng.integers(2, 12)
        caps = np.full(n, max(1.0 / n + 0.01, rng.uniform(0.1, 1.0)))
        w = _project_capped_simplex(rng.normal(0.1, 0.3, n), caps)
        assert (w >= 0).all() and (w <= caps).all()
        assert w.sum() == pytest.approx(1.0, abs=1e-12)


def test_capped_risk_parity_when_solver_stops_early(monkeypatch):
    # Simulate the CI failure: SLSQP returns an infeasible point with an error message.
    import quantedge.riskbased as rb
    from scipy.optimize import OptimizeResult

    def broken(fun, x0, **kwargs):
        return OptimizeResult(x=np.asarray(x0) * 1.01, nit=3, success=False,
                              message="Positive directional derivative for linesearch")

    monkeypatch.setattr(rb, "minimize", broken)
    res = risk_parity(COV, max_weight=0.3)
    assert res.weights.max() <= 0.3 and res.weights.sum() == pytest.approx(1.0, abs=1e-12)
