"""Tests for quantedge.metrics. Expected values are computed by hand in the comments."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from quantedge.config import TRADING_DAYS_PER_YEAR as N
from quantedge.exceptions import DataError, InsufficientDataError
from quantedge.metrics import (
    annualized_covariance,
    annualized_mean_return,
    annualized_volatility,
    cagr,
    correlation_matrix,
    daily_risk_free,
    drawdown_series,
    max_drawdown,
    performance_summary,
    sharpe_ratio,
    simple_returns,
    zero_volatility_assets,
)


def series(values, start="2024-01-01") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


# --------------------------------------------------------------------------- #
# Returns
# --------------------------------------------------------------------------- #
def test_simple_returns_hand_computed():
    # 100 -> 110 is +10 %; 110 -> 99 is -10 %
    r = simple_returns(series([100, 110, 99]))
    np.testing.assert_allclose(r.to_numpy(), [0.10, -0.10])
    assert r.index[0] == pd.Timestamp("2024-01-02")  # first date has no return


def test_simple_returns_dataframe(synthetic_prices):
    r = simple_returns(synthetic_prices)
    assert r.shape == (len(synthetic_prices) - 1, 3)
    np.testing.assert_allclose(r.iloc[0], synthetic_prices.iloc[1] / synthetic_prices.iloc[0] - 1)


@pytest.mark.parametrize("prices, error", [
    ([100, np.nan, 99], DataError),
    ([100, 0, 99], DataError),
    ([100], InsufficientDataError),
])
def test_simple_returns_rejects_bad_prices(prices, error):
    with pytest.raises(error):
        simple_returns(series(prices))


def test_returns_with_nan_or_inf_are_rejected_not_filled():
    with pytest.raises(DataError, match="missing"):
        annualized_volatility(series([0.01, np.nan, 0.02]))
    with pytest.raises(DataError, match="infinite"):
        annualized_volatility(series([0.01, np.inf, 0.02]))


# --------------------------------------------------------------------------- #
# Risk-free conversion
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rf", [0.0, 0.04, 0.065, -0.005])
def test_daily_risk_free_compounds_back_to_annual(rf):
    assert (1 + daily_risk_free(rf)) ** N == pytest.approx(1 + rf, rel=1e-12)


@pytest.mark.parametrize("rf", [-1.0, -2.0, math.nan, math.inf])
def test_daily_risk_free_rejects_impossible_rates(rf):
    with pytest.raises(ValueError):
        daily_risk_free(rf)


# --------------------------------------------------------------------------- #
# Annualisation
# --------------------------------------------------------------------------- #
def test_annualized_mean_return():
    assert annualized_mean_return(series([0.001] * 10)) == pytest.approx(0.001 * N)


def test_cagr_constant_return_is_independent_of_sample_length():
    r = 0.0004
    expected = (1 + r) ** N - 1
    assert cagr(series([r] * N)) == pytest.approx(expected)
    assert cagr(series([r] * 63)) == pytest.approx(expected)


def test_cagr_is_below_arithmetic_mean_when_volatile():
    # +10 % then -10 %: arithmetic mean is 0, but wealth ends at 0.99 -> negative CAGR
    r = series([0.10, -0.10])
    assert annualized_mean_return(r) == pytest.approx(0.0)
    assert cagr(r) == pytest.approx(0.99 ** (N / 2) - 1)
    assert cagr(r) < 0


def test_annualized_volatility_hand_computed():
    # mean 0, deviations +-0.01, sample variance = 2 * 0.0001 / (2 - 1) = 0.0002
    expected = math.sqrt(0.0002) * math.sqrt(N)
    assert annualized_volatility(series([0.01, -0.01])) == pytest.approx(expected)


def test_volatility_needs_two_observations():
    with pytest.raises(InsufficientDataError):
        annualized_volatility(series([0.01]))


# --------------------------------------------------------------------------- #
# Sharpe ratio
# --------------------------------------------------------------------------- #
def test_sharpe_hand_computed_zero_rf():
    # mean 0.02, sample std 0.01 -> daily Sharpe 2 -> annual 2 * sqrt(252)
    assert sharpe_ratio(series([0.01, 0.02, 0.03])) == pytest.approx(2 * math.sqrt(N))


def test_sharpe_with_risk_free_rate():
    r = series([0.01, 0.02, 0.03])
    rf_d = daily_risk_free(0.05)
    expected = (0.02 - rf_d) / 0.01 * math.sqrt(N)  # constant rf leaves std unchanged
    assert sharpe_ratio(r, rf_annual=0.05) == pytest.approx(expected)


def test_sharpe_negative_when_return_below_risk_free():
    assert sharpe_ratio(series([0.0001, -0.0001, 0.0002, 0.0]), rf_annual=0.10) < 0


def test_sharpe_zero_volatility_is_nan_without_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert math.isnan(sharpe_ratio(series([0.001] * 30)))


def test_sharpe_zero_volatility_only_affects_that_column():
    frame = pd.DataFrame({"FLAT": [0.001] * 5, "MOVE": [0.01, -0.02, 0.03, 0.0, 0.01]},
                         index=pd.bdate_range("2024-01-01", periods=5))
    s = sharpe_ratio(frame)
    assert math.isnan(s["FLAT"])
    assert math.isfinite(s["MOVE"])
    assert zero_volatility_assets(frame) == ["FLAT"]


# --------------------------------------------------------------------------- #
# Drawdowns
# --------------------------------------------------------------------------- #
def test_max_drawdown_hand_computed():
    # Prices 100 -> 120 -> 90 -> 130 -> 65: worst fall is 130 -> 65 = -50 %
    r = simple_returns(series([100, 120, 90, 130, 65]))
    assert max_drawdown(r) == pytest.approx(-0.5)
    np.testing.assert_allclose(drawdown_series(r).to_numpy(), [0.0, -0.25, 0.0, -0.5])


def test_drawdown_counts_loss_on_first_day():
    # 100 -> 80 -> 120: the 20 % fall from the starting capital must count
    r = simple_returns(series([100, 80, 120]))
    assert max_drawdown(r) == pytest.approx(-0.2)


def test_drawdown_zero_for_rising_prices():
    r = simple_returns(series([100, 101, 102, 105]))
    assert max_drawdown(r) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Covariance and correlation
# --------------------------------------------------------------------------- #
def test_correlation_perfect_positive_and_negative():
    x = np.array([0.01, -0.02, 0.015, 0.0, -0.005])
    frame = pd.DataFrame({"X": x, "Y": 2 * x + 0.001, "Z": -x},
                         index=pd.bdate_range("2024-01-01", periods=5))
    corr = correlation_matrix(frame)
    assert corr.loc["X", "Y"] == pytest.approx(1.0)
    assert corr.loc["X", "Z"] == pytest.approx(-1.0)
    np.testing.assert_allclose(np.diag(corr), 1.0)


def test_correlation_matches_pandas_on_random_data(synthetic_prices):
    r = simple_returns(synthetic_prices)
    pd.testing.assert_frame_equal(correlation_matrix(r), r.corr(), atol=1e-12)


def test_correlation_zero_variance_asset_is_nan():
    frame = pd.DataFrame({"FLAT": [0.0] * 5, "MOVE": [0.01, -0.02, 0.03, 0.0, 0.01]},
                         index=pd.bdate_range("2024-01-01", periods=5))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        corr = correlation_matrix(frame)
    assert corr["FLAT"].isna().all() and corr.loc["FLAT"].isna().all()
    assert corr.loc["MOVE", "MOVE"] == pytest.approx(1.0)


def test_covariance_is_annualised_and_symmetric(synthetic_prices):
    r = simple_returns(synthetic_prices)
    cov = annualized_covariance(r)
    pd.testing.assert_frame_equal(cov, r.cov() * N)
    np.testing.assert_allclose(cov.to_numpy(), cov.to_numpy().T)
    # variance on the diagonal equals annualised volatility squared
    np.testing.assert_allclose(np.sqrt(np.diag(cov)), annualized_volatility(r).to_numpy())


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def test_performance_summary_shape_and_consistency(synthetic_prices):
    r = simple_returns(synthetic_prices)
    table = performance_summary(r, rf_annual=0.03)
    assert list(table.index) == ["AAA", "BBB", "CCC"]
    assert list(table.columns) == ["annual_return", "cagr", "volatility", "sharpe",
                                   "max_drawdown", "observations"]
    assert (table["observations"] == len(r)).all()
    assert (table["max_drawdown"] <= 0).all()
    pd.testing.assert_series_equal(table["sharpe"], sharpe_ratio(r, 0.03), check_names=False)
