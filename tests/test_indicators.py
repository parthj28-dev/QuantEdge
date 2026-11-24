"""Tests for quantedge.indicators (SMA, trend, crosses) and the SMA filter inside backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantedge.backtest import (
    CASH,
    STRATEGY_FACTORIES,
    BacktestConfig,
    equal_weight,
    fixed_weights,
    run_backtest,
    sma_trend_filter,
)
from quantedge.exceptions import DataError, InsufficientDataError
from quantedge.indicators import (
    DEATH_CROSS,
    GOLDEN_CROSS,
    above_trend,
    crosses,
    price_to_sma_from_returns,
    sma,
    trend_status,
)
from quantedge.metrics import daily_risk_free, simple_returns


def series(values, start="2024-01-01") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_sma_hand_computed_and_leading_nans():
    s = sma(series([1, 2, 3, 4, 5]), 3)
    assert s.iloc[:2].isna().all()
    np.testing.assert_allclose(s.iloc[2:], [2, 3, 4])


def test_above_trend_is_strict_and_undefined_before_window():
    a = above_trend(series([3, 3, 3, 4, 2]), 3)
    assert np.isnan(a.iloc[0]) and np.isnan(a.iloc[1])
    assert a.iloc[2] == 0.0          # equal to its SMA is NOT above
    assert a.iloc[3] == 1.0 and a.iloc[4] == 0.0


def test_crosses_hand_example():
    p = series([5, 5, 5, 4, 3, 4, 6, 7, 6, 4, 3])
    ev = crosses(p, fast=2, slow=3)
    assert list(ev["event"]) == [GOLDEN_CROSS, DEATH_CROSS]
    f, s = sma(p, 2), sma(p, 3)
    for _, row in ev.iterrows():
        d = row["date"]
        prev = p.index[p.index.get_loc(d) - 1]
        if row["event"] == GOLDEN_CROSS:
            assert f[d] > s[d] and f[prev] <= s[prev]
        else:
            assert f[d] <= s[d] and f[prev] > s[prev]


def test_no_cross_until_both_smas_exist_two_days():
    p = series([1, 2, 3, 4, 5, 6])   # fast above slow from the first defined day: no event
    assert crosses(p, 2, 3).empty


@pytest.mark.parametrize("fast, slow", [(3, 3), (5, 2)])
def test_fast_must_be_shorter(fast, slow):
    with pytest.raises(ValueError):
        crosses(series(range(1, 20)), fast, slow)


@pytest.mark.parametrize("window", [1, 0, 2.5, True])
def test_bad_window(window):
    with pytest.raises(ValueError):
        sma(series(range(1, 20)), window)


def test_window_longer_than_data():
    with pytest.raises(InsufficientDataError):
        sma(series([1, 2, 3]), 5)


def test_trend_status_columns(synthetic_prices):
    t = trend_status(synthetic_prices, 20, 50)
    assert list(t.index) == list(synthetic_prices.columns)
    a = "AAA"
    assert t.loc[a, "sma_50"] == pytest.approx(synthetic_prices[a].iloc[-50:].mean())
    assert t.loc[a, "above_slow"] == (synthetic_prices[a].iloc[-1] > t.loc[a, "sma_50"])


def test_price_to_sma_from_returns_matches_prices(synthetic_prices):
    r = simple_returns(synthetic_prices)
    for n in (5, 20, 60):
        ratio = price_to_sma_from_returns(r.iloc[:150], n)
        expected = synthetic_prices.iloc[150] / synthetic_prices.iloc[151 - n:151].mean()
        np.testing.assert_allclose(ratio, expected, rtol=1e-12)


def test_price_to_sma_needs_enough_returns(synthetic_prices):
    r = simple_returns(synthetic_prices)
    with pytest.raises(InsufficientDataError, match="at least 9"):
        price_to_sma_from_returns(r.iloc[:8], 10)


# --------------------------------------------------------------------------- #
# Trend filter inside the backtest
# --------------------------------------------------------------------------- #
def test_filter_holds_exactly_the_assets_above_trend_on_every_rebalance(synthetic_prices):
    r = simple_returns(synthetic_prices)
    n = 30
    res = run_backtest(r, sma_trend_filter(equal_weight(), n), BacktestConfig(lookback=40, rebalance="W"))
    avg = sma(synthetic_prices, n)
    for date, w in res.weights.iterrows():
        above = synthetic_prices.loc[date] > avg.loc[date]
        held = w.drop(CASH) > 0
        assert (held == above).all(), date
        assert w[CASH] == pytest.approx((~above).sum() / 3)


def test_cash_earns_the_cash_rate_and_moving_to_cash_costs_only_the_sale():
    r = pd.DataFrame({"A": [0.0] * 6}, index=pd.bdate_range("2024-01-01", periods=6))
    half = pd.Series({"A": 0.5, CASH: 0.5})
    res = run_backtest(r, fixed_weights(half), BacktestConfig(lookback=2, rebalance=None,
                                                               cost_bps=100, cash_rate=0.10))
    rf_d = daily_risk_free(0.10)
    assert res.turnover.iloc[0] == pytest.approx(0.5)           # invested fraction only
    assert res.costs.iloc[0] == pytest.approx(0.01 * 0.5)
    first = (1 + 0.5 * rf_d) * (1 - 0.005) - 1
    assert res.returns.iloc[0] == pytest.approx(first)
    assert CASH in res.weights.columns


def test_strategies_without_cash_are_unchanged_by_cash_rate(synthetic_prices):
    r = simple_returns(synthetic_prices)
    a = run_backtest(r, equal_weight(), BacktestConfig(lookback=30, cash_rate=0.0))
    b = run_backtest(r, equal_weight(), BacktestConfig(lookback=30, cash_rate=0.25))
    pd.testing.assert_series_equal(a.returns, b.returns, check_exact=True)
    assert CASH not in a.weights.columns


def test_column_named_cash_is_rejected(synthetic_prices):
    r = simple_returns(synthetic_prices).rename(columns={"AAA": "Cash"})
    with pytest.raises(DataError, match="reserved"):
        run_backtest(r, equal_weight(), BacktestConfig(lookback=30))


def test_lookback_too_short_for_sma_skips_or_raises(synthetic_prices):
    r = simple_returns(synthetic_prices)
    with pytest.raises(InsufficientDataError):
        run_backtest(r, sma_trend_filter(equal_weight(), 100), BacktestConfig(lookback=50))


def test_registry_has_eight_strategies_that_all_run(synthetic_prices):
    r = simple_returns(synthetic_prices)
    assert len(STRATEGY_FACTORIES) == 8
    cfg = BacktestConfig(lookback=60, rebalance="M", cash_rate=0.05)
    for name, make in STRATEGY_FACTORIES.items():
        for cov in ("sample", "ledoit_wolf"):
            res = run_backtest(r, make(0.05, 0.6, cov, 50), cfg, name=name)
            assert np.isfinite(res.returns).all(), name
            assert (res.weights.drop(columns=[CASH], errors="ignore").max(axis=1) <= 0.6 + 1e-6).all() \
                or name == "Equal weight + SMA filter" or name == "Equal weight", name


def test_moving_into_cash_costs_only_the_amount_sold():
    r = pd.DataFrame({"A": [0.0] * 8}, index=pd.bdate_range("2024-01-01", periods=8))
    calls = {"n": 0}

    def strategy(window):
        calls["n"] += 1
        return pd.Series({"A": 1.0}) if calls["n"] == 1 else pd.Series({"A": 0.5, CASH: 0.5})

    res = run_backtest(r, strategy, BacktestConfig(lookback=2, rebalance=2, cost_bps=100))
    assert res.turnover.iloc[0] == pytest.approx(1.0)
    assert res.turnover.iloc[1] == pytest.approx(0.5)            # sold half; cash itself is not traded
    assert res.costs.iloc[1] == pytest.approx(0.005)


def test_price_equal_to_its_sma_is_not_above_trend():
    r = pd.DataFrame({"A": [0.0] * 30, "B": [0.001] * 30}, index=pd.bdate_range("2024-01-01", periods=30))
    res = run_backtest(r, sma_trend_filter(equal_weight(), 5), BacktestConfig(lookback=10, rebalance=None))
    w = res.weights.iloc[0]
    assert w["A"] == 0.0 and w[CASH] == pytest.approx(0.5) and w["B"] == pytest.approx(0.5)
