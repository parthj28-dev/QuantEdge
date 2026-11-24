"""Tests for quantedge.estimators (Ledoit-Wolf and Black-Litterman)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantedge.config import TRADING_DAYS_PER_YEAR as N
from quantedge.estimators import (
    COV_LEDOIT_WOLF,
    COV_SAMPLE,
    absolute_view,
    black_litterman,
    condition_number,
    covariance,
    implied_returns,
    implied_risk_aversion,
    ledoit_wolf,
    relative_view,
)
from quantedge.exceptions import EstimationError
from quantedge.metrics import annual_arithmetic_risk_free, annualized_covariance
from quantedge.optimize import max_sharpe


def random_returns(seed: int, n_obs: int, n_assets: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mix = rng.normal(size=(n_assets, n_assets))
    x = rng.normal(size=(n_obs, n_assets)) @ mix * 0.01
    return pd.DataFrame(x, columns=[f"A{i}" for i in range(n_assets)],
                        index=pd.bdate_range("2020-01-01", periods=n_obs))


def cov_frame(vols, corr, names) -> pd.DataFrame:
    vols = np.asarray(vols, float)
    return pd.DataFrame(np.asarray(corr, float) * np.outer(vols, vols), index=names, columns=names)


# --------------------------------------------------------------------------- #
# Ledoit-Wolf
# --------------------------------------------------------------------------- #
# Reference intensities computed with sklearn.covariance.ledoit_wolf on the same seeded data
# (frozen here so scikit-learn is not a dependency).
@pytest.mark.parametrize("seed, n_obs, n_assets, sklearn_shrinkage", [
    (0, 60, 5, 0.1415484568),
    (1, 30, 40, 0.6068692125),   # more assets than observations
    (2, 500, 8, 0.0169995305),
    (3, 252, 3, 0.0238735943),
])
def test_shrinkage_matches_scikit_learn_reference(seed, n_obs, n_assets, sklearn_shrinkage):
    assert ledoit_wolf(random_returns(seed, n_obs, n_assets)).shrinkage == pytest.approx(sklearn_shrinkage, abs=1e-9)


def test_shrunk_matrix_is_the_documented_blend():
    r = random_returns(0, 60, 5)
    res = ledoit_wolf(r)
    sample = annualized_covariance(r).to_numpy()
    target = np.trace(sample) / 5 * np.eye(5)
    expected = (1 - res.shrinkage) * sample + res.shrinkage * target
    np.testing.assert_allclose(res.covariance.to_numpy(), expected, rtol=1e-12)
    assert res.target_variance == pytest.approx(np.trace(sample) / 5)


def test_shrinkage_fixes_singular_sample_covariance():
    r = random_returns(1, 30, 40)  # 40 assets, 30 days: sample covariance is singular
    assert condition_number(annualized_covariance(r)) == np.inf
    shrunk = ledoit_wolf(r).covariance
    assert np.isfinite(condition_number(shrunk))
    assert np.linalg.eigvalsh(shrunk.to_numpy()).min() > 0


def test_more_data_means_less_shrinkage():
    short = ledoit_wolf(random_returns(4, 40, 10)).shrinkage
    long = ledoit_wolf(random_returns(4, 4000, 10)).shrinkage
    assert long < short and long < 0.05


def test_identity_truth_gets_heavy_shrinkage():
    rng = np.random.default_rng(5)
    r = pd.DataFrame(rng.normal(0, 0.01, (50, 20)), index=pd.bdate_range("2020-01-01", periods=50))
    assert ledoit_wolf(r).shrinkage > 0.5


def test_single_asset_is_unchanged():
    r = random_returns(6, 100, 1)
    res = ledoit_wolf(r)
    assert res.shrinkage == 0.0
    np.testing.assert_allclose(res.covariance, annualized_covariance(r))


def test_covariance_dispatch():
    r = random_returns(7, 80, 4)
    pd.testing.assert_frame_equal(covariance(r, COV_SAMPLE), annualized_covariance(r))
    pd.testing.assert_frame_equal(covariance(r, COV_LEDOIT_WOLF), ledoit_wolf(r).covariance)
    with pytest.raises(ValueError, match="Unknown covariance"):
        covariance(r, "magic")


def test_ledoit_wolf_rejects_missing_values():
    r = random_returns(8, 60, 3)
    r.iloc[5, 1] = np.nan
    with pytest.raises(Exception, match="missing"):
        ledoit_wolf(r)


# --------------------------------------------------------------------------- #
# Black-Litterman
# --------------------------------------------------------------------------- #
NAMES = ["A", "B", "C"]
SIGMA = cov_frame([0.15, 0.20, 0.25], [[1, 0.3, 0.2], [0.3, 1, 0.4], [0.2, 0.4, 1]], NAMES)
W_MKT = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})


def test_implied_returns_formula():
    pi = implied_returns(SIGMA, W_MKT, risk_aversion=2.5)
    np.testing.assert_allclose(pi, 2.5 * SIGMA.to_numpy() @ W_MKT.to_numpy())


def test_no_views_returns_equilibrium_and_inflated_covariance():
    rf = 0.05
    res = black_litterman(SIGMA, W_MKT, risk_aversion=2.5, rf_annual=rf)
    pd.testing.assert_series_equal(res.posterior, res.prior, check_names=False)
    np.testing.assert_allclose(res.prior - annual_arithmetic_risk_free(rf),
                               implied_returns(SIGMA, W_MKT, 2.5))
    np.testing.assert_allclose(res.posterior_cov, SIGMA * 1.05)  # Sigma + tau Sigma


def test_reverse_optimisation_recovers_market_portfolio():
    rf = 0.04
    res = black_litterman(SIGMA, W_MKT, risk_aversion=3.0, rf_annual=rf)
    w = max_sharpe(res.posterior, res.posterior_cov, rf_annual=rf).weights
    np.testing.assert_allclose(w, W_MKT, atol=1e-5)


def test_fully_confident_absolute_view_is_matched_exactly():
    res = black_litterman(SIGMA, W_MKT, risk_aversion=2.5, rf_annual=0.03,
                          views=[absolute_view("B", 0.20, confidence=1.0)])
    assert res.posterior["B"] == pytest.approx(0.20)
    assert res.views.loc[0, "posterior_implied"] == pytest.approx(0.20)


def test_half_confidence_moves_halfway_for_a_single_absolute_view():
    res = black_litterman(SIGMA, W_MKT, risk_aversion=2.5, rf_annual=0.03,
                          views=[absolute_view("C", 0.30, confidence=0.5)])
    assert res.posterior["C"] == pytest.approx((res.prior["C"] + 0.30) / 2)


def test_relative_view_is_unaffected_by_risk_free_rate():
    v = [relative_view("A", "C", 0.02, confidence=0.7)]
    lo = black_litterman(SIGMA, W_MKT, risk_aversion=2.5, rf_annual=0.0, views=v)
    hi = black_litterman(SIGMA, W_MKT, risk_aversion=2.5, rf_annual=0.10, views=v)
    np.testing.assert_allclose(lo.posterior - lo.prior, hi.posterior - hi.prior, atol=1e-12)


def test_posterior_covariance_grows_by_a_psd_amount():
    res = black_litterman(SIGMA, W_MKT, risk_aversion=2.5,
                          views=[absolute_view("A", 0.1, 0.6), relative_view("B", "C", 0.01, 0.4)])
    assert np.linalg.eigvalsh((res.posterior_cov - SIGMA).to_numpy()).min() > -1e-12


def test_higher_confidence_moves_closer_to_the_view():
    gaps = []
    for c in (0.2, 0.5, 0.9):
        res = black_litterman(SIGMA, W_MKT, risk_aversion=2.5, views=[absolute_view("A", 0.25, c)])
        gaps.append(abs(res.posterior["A"] - 0.25))
    assert gaps[0] > gaps[1] > gaps[2]


@pytest.mark.parametrize("kwargs, match", [
    ({"views": [absolute_view("A", 0.1, 1.0), absolute_view("A", 0.1, 1.0)]}, "contradictory or duplicated"),
    ({"views": [absolute_view("Z", 0.1)]}, "unknown asset"),
    ({"views": [absolute_view("A", 0.1, 0.0)]}, r"confidence"),
    ({"views": [absolute_view("A", 0.1, 1.5)]}, r"confidence"),
    ({"tau": 0.0}, "tau"),
    ({"risk_aversion": -1.0}, "Risk aversion"),
])
def test_invalid_inputs_raise_estimation_error(kwargs, match):
    args = {"risk_aversion": 2.5, **kwargs}
    with pytest.raises(EstimationError, match=match):
        black_litterman(SIGMA, W_MKT, **args)


@pytest.mark.parametrize("weights", [pd.Series({"A": 0.5, "B": 0.5}),
                                     pd.Series({"A": 0.6, "B": 0.6, "C": -0.2}),
                                     pd.Series({"A": 0.5, "B": 0.3, "C": 0.1})])
def test_bad_market_weights(weights):
    with pytest.raises(EstimationError):
        implied_returns(SIGMA, weights, 2.5)


def test_relative_view_needs_two_assets():
    with pytest.raises(EstimationError):
        relative_view("A", "A", 0.01)


def test_implied_risk_aversion_formula_and_negative_premium():
    rng = np.random.default_rng(9)
    bench = pd.Series(rng.normal(0.0006, 0.01, 500), index=pd.bdate_range("2020-01-01", periods=500))
    rf = 0.03
    premium = (bench - (1 + rf) ** (1 / N) + 1).mean() * N
    assert implied_risk_aversion(bench, rf) == pytest.approx(premium / (bench.var(ddof=1) * N))
    with pytest.raises(EstimationError, match="not positive"):
        implied_risk_aversion(bench - 0.002, rf)
