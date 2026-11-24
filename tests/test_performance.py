"""Tests for quantedge.performance: lecture examples, consistency with metrics/CAPM, timing models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantedge.capm import estimate_beta
from quantedge.exceptions import EstimationError, InsufficientDataError
from quantedge.metrics import sharpe_ratio
from quantedge.performance import (
    HENRIKSSON_MERTON,
    TREYNOR_MAZUY,
    market_timing,
    measures_from_summary,
    performance_measures,
    periodic_returns,
)


# --------------------------------------------------------------------------- #
# Lecture examples (summary-statistics mode)
# --------------------------------------------------------------------------- #
def test_treynor_example():                                  # L30: M 14 %, rf 8 %
    t = measures_from_summary(pd.DataFrame({"return": [0.12, 0.16, 0.18], "beta": [0.9, 1.05, 1.20]},
                                           index=["W", "X", "Y"]), market_return=0.14, risk_free=0.08)
    np.testing.assert_allclose(t["treynor"], [0.04 / 0.9, 0.08 / 1.05, 0.10 / 1.2, 0.06])
    assert list(t["treynor_rank"]) == [4, 2, 1, 3]


def test_sharpe_example():                                   # L30: M 14 % (20 %), rf 8 %
    t = measures_from_summary(pd.DataFrame({"return": [0.13, 0.17, 0.16], "beta": [1, 1, 1],
                                            "std": [0.18, 0.22, 0.23]}, index=["D", "E", "F"]),
                              market_return=0.14, market_std=0.20, risk_free=0.08)
    np.testing.assert_allclose(t["sharpe"], [0.05 / 0.18, 0.409, 0.348, 0.300], atol=5e-4)
    assert t.loc["E", "sharpe_rank"] == 1 and t.loc["D", "sharpe_rank"] == 4


def test_pqrs_exercise_diversification_and_rankings():       # L30 exercise, rf 6 %
    t = measures_from_summary(pd.DataFrame({"return": [0.15, 0.20, 0.10, 0.17], "beta": [1, 1.5, 0.6, 1.1],
                                            "std": [0.05, 0.10, 0.03, 0.06]}, index=list("PQRS")),
                              market_return=0.13, market_std=0.04, risk_free=0.06)
    assert t.loc["P", "unsystematic_variance"] == pytest.approx(0.05**2 - 0.04**2)
    assert t.loc["Market", "r_squared"] == pytest.approx(1.0)
    assert t.loc["Q", "treynor"] == pytest.approx(0.14 / 1.5)
    assert t.loc["P", "jensen_alpha"] == pytest.approx(0.15 - (0.06 + 1 * 0.07))
    assert t.loc["P", "m_squared"] == pytest.approx(0.06 + (0.09 / 0.05) * 0.04)
    # Rankings differ because Q and R are poorly diversified (low R^2): Sharpe punishes that, Treynor not.
    assert list(t["sharpe_rank"]) != list(t["treynor_rank"])


def test_summary_needs_columns():
    with pytest.raises(EstimationError):
        measures_from_summary(pd.DataFrame({"return": [0.1]}), market_return=0.1, risk_free=0.05)


# --------------------------------------------------------------------------- #
# Series mode
# --------------------------------------------------------------------------- #
@pytest.fixture
def market() -> pd.Series:
    rng = np.random.default_rng(11)
    return pd.Series(rng.normal(0.0004, 0.01, 750), index=pd.bdate_range("2021-01-01", periods=750), name="MKT")


@pytest.fixture
def funds(market) -> pd.DataFrame:
    rng = np.random.default_rng(12)
    return pd.DataFrame({
        "Levered": 0.0001 + 1.3 * market + rng.normal(0, 0.004, len(market)),
        "Defensive": 0.0002 + 0.6 * market + rng.normal(0, 0.006, len(market)),
    }, index=market.index)


def test_series_measures_match_metrics_and_capm(funds, market):
    rf = 0.05
    t = performance_measures(funds, market, rf_annual=rf)
    capm = estimate_beta(funds, market, rf)
    for col in funds:
        assert t.loc[col, "beta"] == pytest.approx(capm.loc[col, "beta"])
        assert t.loc[col, "jensen_alpha"] == pytest.approx(capm.loc[col, "alpha_annual"])
        assert t.loc[col, "sharpe"] == pytest.approx(sharpe_ratio(funds[col], rf))
        assert t.loc[col, "treynor"] == pytest.approx(t.loc[col, "annual_excess_return"] / t.loc[col, "beta"])
        assert t.loc[col, "r_squared"] == pytest.approx(capm.loc[col, "r_squared"])
    assert t.loc["MKT", "beta"] == pytest.approx(1.0)
    assert t.loc["MKT", "jensen_alpha"] == pytest.approx(0.0, abs=1e-12)


def test_information_ratio_hand_computed(funds, market):
    t = performance_measures(funds, market)
    active = funds["Levered"] - market
    te = active.std(ddof=1) * np.sqrt(252)
    assert t.loc["Levered", "tracking_error"] == pytest.approx(te)
    assert t.loc["Levered", "information_ratio"] == pytest.approx(active.mean() * 252 / te)


def test_m_squared_has_market_volatility(funds, market):
    t = performance_measures(funds, market, rf_annual=0.03)
    rf_arith = 252 * ((1.03) ** (1 / 252) - 1)
    m_vol = market.std(ddof=1) * np.sqrt(252)
    assert t.loc["Defensive", "m_squared"] == pytest.approx(rf_arith + t.loc["Defensive", "sharpe"] * m_vol)
    assert t.loc["MKT", "m_squared_vs_benchmark"] == pytest.approx(0.0, abs=1e-12)


def test_only_common_dates_are_used(funds, market):
    t = performance_measures(funds.iloc[100:], market.iloc[:600])
    assert t["observations"].iloc[0] == 500


def test_monthly_frequency(funds, market):
    fm, mm = periodic_returns(funds, "M"), periodic_returns(market, "M")
    assert fm.iloc[0, 0] == pytest.approx((1 + funds.loc["2021-01", "Levered"]).prod() - 1)
    t = performance_measures(fm, mm, periods_per_year=12)
    assert t.loc["Levered", "beta"] == pytest.approx(1.3, abs=0.15)


# --------------------------------------------------------------------------- #
# Market timing
# --------------------------------------------------------------------------- #
def test_henriksson_merton_recovers_two_betas_exactly(market):
    rf = 0.0
    fund = (0.0002 + 0.5 * market + 0.4 * market.clip(lower=0)).rename("Timer").to_frame()
    res = market_timing(fund, market, rf_annual=rf, model=HENRIKSSON_MERTON)
    row = res.table.loc["Timer"]
    assert row["beta"] == pytest.approx(0.5) and row["gamma"] == pytest.approx(0.4)
    assert row["up_market_beta"] == pytest.approx(0.9) and row["alpha"] == pytest.approx(0.0002)
    assert res.up_periods + res.down_periods == len(market)


def test_treynor_mazuy_detects_convexity(market):
    rng = np.random.default_rng(5)
    fund = pd.DataFrame({"Timer": 0.8 * market + 6.0 * market**2 + rng.normal(0, 0.001, len(market)),
                         "Constant": 0.8 * market + rng.normal(0, 0.001, len(market))})
    t = market_timing(fund, market, model=TREYNOR_MAZUY).table
    assert t.loc["Timer", "gamma"] == pytest.approx(6.0, rel=0.1) and t.loc["Timer", "gamma_t"] > 5
    assert abs(t.loc["Constant", "gamma_t"]) < 3


def test_timing_needs_up_and_down_markets(market):
    up_only = market.abs() + 0.001
    with pytest.raises(EstimationError, match="collinear"):
        market_timing((0.5 * up_only).rename("F").to_frame(), up_only, model=HENRIKSSON_MERTON)


def test_timing_needs_observations(market):
    with pytest.raises(InsufficientDataError):
        market_timing(market.iloc[:10].rename("F").to_frame(), market.iloc[:10])


def test_bad_model_name(funds, market):
    with pytest.raises(ValueError):
        market_timing(funds, market, model="magic")
