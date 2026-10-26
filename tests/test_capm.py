"""Tests for quantedge.capm.

Strategy: build asset returns from a KNOWN beta and alpha, then check the
regression recovers them. All data is synthetic (fixed seeds).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from quantedge.capm import (
    PREMIUM_HISTORICAL,
    PREMIUM_USER,
    align_returns,
    capm_analysis,
    capm_expected_returns,
    estimate_beta,
    historical_market_premium,
)
from quantedge.config import TRADING_DAYS_PER_YEAR as N
from quantedge.exceptions import DataError, InsufficientDataError
from quantedge.metrics import annual_arithmetic_risk_free, annualized_mean_return, daily_risk_free


@pytest.fixture
def market() -> pd.Series:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2022-01-03", periods=500)
    return pd.Series(rng.normal(0.0004, 0.011, len(dates)), index=dates, name="MKT")


def exact_asset(market: pd.Series, beta: float, alpha_daily: float, rf_annual: float = 0.0) -> pd.Series:
    """r_i = rf_d + alpha + beta * (r_m - rf_d), with no noise at all."""
    rf_d = daily_risk_free(rf_annual)
    return rf_d + alpha_daily + beta * (market - rf_d)


# --------------------------------------------------------------------------- #
# Recovering known parameters
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("beta, alpha", [(1.5, 0.0002), (0.4, -0.0001), (-0.8, 0.0), (0.0, 0.0003)])
def test_exact_relationship_is_recovered(market, beta, alpha):
    asset = exact_asset(market, beta, alpha, rf_annual=0.06).rename("A")
    row = estimate_beta(asset, market, rf_annual=0.06).loc["A"]
    assert row["beta"] == pytest.approx(beta, abs=1e-10)
    assert row["alpha_daily"] == pytest.approx(alpha, abs=1e-12)
    assert row["alpha_annual"] == pytest.approx(alpha * N, abs=1e-10)
    assert row["beta_std_error"] == pytest.approx(0.0, abs=1e-9)
    assert row["residual_volatility"] == pytest.approx(0.0, abs=1e-9)
    if beta != 0.0:
        assert row["r_squared"] == pytest.approx(1.0)


def test_matches_numpy_least_squares_with_noise(market):
    rng = np.random.default_rng(11)
    rf = 0.065
    asset = exact_asset(market, 1.2, 0.0001, rf) + rng.normal(0, 0.008, len(market))
    row = estimate_beta(asset.rename("A"), market, rf).loc["A"]

    rf_d = daily_risk_free(rf)
    x, y = (market - rf_d).to_numpy(), (asset - rf_d).to_numpy()
    slope, intercept = np.polyfit(x, y, deg=1)
    assert row["beta"] == pytest.approx(slope, rel=1e-10)
    assert row["alpha_daily"] == pytest.approx(intercept, rel=1e-8)
    assert row["r_squared"] == pytest.approx(np.corrcoef(x, y)[0, 1] ** 2, rel=1e-10)
    # Standard error formula cross-check: residual std / sqrt(Sxx)
    resid = y - (intercept + slope * x)
    se = np.sqrt(resid @ resid / (len(x) - 2) / ((x - x.mean()) ** 2).sum())
    assert row["beta_std_error"] == pytest.approx(se, rel=1e-10)
    # True beta (1.2) should lie within ~3 standard errors of the estimate
    assert abs(row["beta"] - 1.2) < 3 * row["beta_std_error"]


def test_benchmark_regressed_on_itself(market):
    row = estimate_beta(market.rename("SELF"), market, rf_annual=0.05).loc["SELF"]
    assert row["beta"] == pytest.approx(1.0)
    assert row["alpha_annual"] == pytest.approx(0.0, abs=1e-12)
    assert row["r_squared"] == pytest.approx(1.0)


def test_beta_does_not_depend_on_risk_free_rate_but_alpha_does(market):
    # Subtracting a constant does not change covariance, so beta is unchanged.
    # alpha(rf) = alpha(0) - rf_d * (1 - beta): low-beta assets "gain" alpha as rf rises.
    rng = np.random.default_rng(3)
    asset = (0.5 * market + rng.normal(0, 0.005, len(market))).rename("A")
    no_rf = estimate_beta(asset, market, 0.0).loc["A"]
    with_rf = estimate_beta(asset, market, 0.07).loc["A"]
    assert with_rf["beta"] == pytest.approx(no_rf["beta"], rel=1e-12)
    expected_alpha = no_rf["alpha_daily"] - daily_risk_free(0.07) * (1 - no_rf["beta"])
    assert with_rf["alpha_daily"] == pytest.approx(expected_alpha, rel=1e-10)


def test_multiple_assets_in_one_call(market):
    assets = pd.DataFrame({
        "LOW": exact_asset(market, 0.5, 0.0),
        "HIGH": exact_asset(market, 2.0, 0.0),
    })
    table = estimate_beta(assets, market)
    assert list(table.index) == ["LOW", "HIGH"]
    np.testing.assert_allclose(table["beta"], [0.5, 2.0])
    assert (table["n_observations"] == len(market)).all()


# --------------------------------------------------------------------------- #
# Alignment on exact dates
# --------------------------------------------------------------------------- #
def test_alignment_uses_only_shared_dates(market):
    asset = exact_asset(market, 2.0, 0.0).rename("A")
    # Asset gets 5 extra dates with wild values; benchmark loses 10 dates.
    extra = pd.Series(0.5, index=pd.bdate_range("2030-01-01", periods=5), name="A")
    asset_long = pd.concat([asset, extra])
    market_short = market.drop(market.index[100:110])

    aligned = align_returns(asset_long, market_short)
    assert len(aligned.benchmark) == len(market) - 10
    assert aligned.assets.index.equals(aligned.benchmark.index)
    assert aligned.unmatched_asset_dates == 15  # 5 extra + 10 the benchmark lacks
    assert aligned.unmatched_benchmark_dates == 0

    result = capm_analysis(asset_long, market_short)
    # If dates were mismatched (or wild values leaked in) beta would not be exactly 2
    assert result.table.loc["A", "beta"] == pytest.approx(2.0, abs=1e-10)
    assert result.n_observations == len(market) - 10
    assert any("not identical" in note for note in result.notes)


def test_shifted_benchmark_destroys_the_relationship(market):
    # A one-day misalignment is a classic bug: it should make beta ~0, not 2.
    asset = exact_asset(market, 2.0, 0.0).rename("A")
    shifted = market.shift(1).dropna()
    beta = estimate_beta(asset, shifted).loc["A", "beta"]
    assert abs(beta) < 0.3


def test_unordered_inputs_are_aligned_by_date(market):
    asset = exact_asset(market, 1.3, 0.0).rename("A").sample(frac=1.0, random_state=1)
    assert estimate_beta(asset, market).loc["A", "beta"] == pytest.approx(1.3)


def test_too_few_shared_dates_raises(market):
    asset = exact_asset(market, 1.0, 0.0).iloc[:20].rename("A")
    with pytest.raises(InsufficientDataError, match="20 date"):
        capm_analysis(asset, market, min_observations=60)


def test_missing_returns_are_rejected_not_filled(market):
    asset = exact_asset(market, 1.0, 0.0).rename("A")
    asset.iloc[50] = np.nan
    with pytest.raises(DataError, match="missing"):
        estimate_beta(asset, market)


def test_duplicate_dates_raise(market):
    with pytest.raises(DataError, match="duplicate"):
        align_returns(market.rename("A"), pd.concat([market, market.iloc[[0]]]))


# --------------------------------------------------------------------------- #
# Zero variance
# --------------------------------------------------------------------------- #
def test_zero_variance_benchmark_raises_clear_error(market):
    flat = pd.Series(0.0003, index=market.index)
    with pytest.raises(DataError, match="zero variance"):
        estimate_beta(market.rename("A"), flat)


def test_zero_variance_asset_has_zero_beta_and_undefined_r_squared(market):
    assets = pd.DataFrame({"CASH": 0.0002, "MOVE": exact_asset(market, 1.1, 0.0)}, index=market.index)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no divide-by-zero warnings allowed
        table = estimate_beta(assets, market)
    assert table.loc["CASH", "beta"] == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(table.loc["CASH", "r_squared"])
    assert table.loc["MOVE", "r_squared"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Expected returns
# --------------------------------------------------------------------------- #
def test_capm_expected_returns_formula():
    betas = pd.Series({"ZERO": 0.0, "ONE": 1.0, "TWO": 2.0})
    rf_arith = annual_arithmetic_risk_free(0.05)
    expected = capm_expected_returns(betas, rf_annual=0.05, market_premium=0.06)
    np.testing.assert_allclose(expected, [rf_arith, rf_arith + 0.06, rf_arith + 0.12])
    # Arithmetic rf is slightly below the effective 5 %: that is compounding, not an error
    assert 0.048 < rf_arith < 0.05


@pytest.mark.parametrize("premium", [np.nan, np.inf])
def test_capm_rejects_non_finite_premium(premium):
    with pytest.raises(ValueError):
        capm_expected_returns(pd.Series({"A": 1.0}), 0.05, premium)


def test_historical_premium_identity(market):
    # With the historical premium: realised mean return = CAPM expected return + alpha.
    rng = np.random.default_rng(5)
    assets = pd.DataFrame({
        "A": exact_asset(market, 0.7, 0.0002, 0.04) + rng.normal(0, 0.006, len(market)),
        "B": exact_asset(market, 1.4, -0.0001, 0.04) + rng.normal(0, 0.009, len(market)),
    })
    result = capm_analysis(assets, market, rf_annual=0.04)
    t = result.table
    assert result.premium_source == PREMIUM_HISTORICAL
    assert result.market_premium == pytest.approx(historical_market_premium(market, 0.04))
    np.testing.assert_allclose(t["historical_return"], annualized_mean_return(assets))
    np.testing.assert_allclose(t["historical_return"], t["capm_expected_return"] + t["alpha_annual"],
                               atol=1e-12)


def test_user_supplied_premium_is_used(market):
    asset = exact_asset(market, 1.5, 0.0).rename("A")
    result = capm_analysis(asset, market, rf_annual=0.065, market_premium=0.06)
    assert result.premium_source == PREMIUM_USER
    assert result.market_premium == 0.06
    expected = annual_arithmetic_risk_free(0.065) + 1.5 * 0.06
    assert result.table.loc["A", "capm_expected_return"] == pytest.approx(expected)


def test_negative_historical_premium_is_flagged():
    dates = pd.bdate_range("2022-01-03", periods=200)
    rng = np.random.default_rng(9)
    falling = pd.Series(rng.normal(-0.001, 0.01, len(dates)), index=dates)
    result = capm_analysis((1.2 * falling).rename("A"), falling, rf_annual=0.05)
    assert result.market_premium < 0
    assert any("LOWER returns for higher-beta" in note for note in result.notes)


def test_inputs_are_not_mutated(market):
    asset = exact_asset(market, 1.0, 0.0).rename("A")
    a_copy, m_copy = asset.copy(), market.copy()
    capm_analysis(asset, market, rf_annual=0.03)
    pd.testing.assert_series_equal(asset, a_copy)
    pd.testing.assert_series_equal(market, m_copy)
