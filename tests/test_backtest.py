"""Tests for quantedge.backtest.

Look-ahead is tested from three independent angles:
  1. The strategy is never shown a date after the rebalance date (spy strategy).
  2. Changing the future cannot change past decisions or past returns (poisoned future).
  3. A trade decided at the close of day t does not earn day t's return (same-day trap).
Turnover and costs are checked against small examples worked out by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantedge.backtest import (
    BacktestConfig,
    backtest_summary,
    equal_weight,
    fixed_weights,
    max_sharpe_strategy,
    min_variance_strategy,
    rebalance_positions,
    run_backtest,
)
from quantedge.exceptions import DataError, InsufficientDataError, OptimizationError
from quantedge.metrics import simple_returns


def frame(rows, columns=("A", "B"), start="2024-01-01") -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(columns), index=pd.bdate_range(start, periods=len(rows)),
                        dtype=float)


@pytest.fixture
def daily_returns(synthetic_prices) -> pd.DataFrame:
    return simple_returns(synthetic_prices)  # 299 synthetic days, 3 assets


# --------------------------------------------------------------------------- #
# Hand-computed drift, turnover and costs
# --------------------------------------------------------------------------- #
HAND_RETURNS = [
    [0.00, 0.00],   # d0  history
    [0.00, 0.00],   # d1  history -> initial 50/50 allocation at the close (lookback 2)
    [0.10, 0.00],   # d2  A +10 %: gross 5 %; drift to (0.55, 0.50)/1.05; rebalance
    [0.00, 0.20],   # d3  B +20 %: gross 10 %; drift to (0.50, 0.60)/1.10; rebalance
    [0.00, 0.00],   # d4  last day: no trade
]


def test_turnover_and_drift_hand_computed():
    result = run_backtest(frame(HAND_RETURNS), equal_weight(), BacktestConfig(lookback=2, rebalance=1, cost_bps=0))
    # After d2 the drifted weights are 0.5238 / 0.4762, so tau = 2 * (0.55/1.05 - 0.5) = 0.05/1.05
    np.testing.assert_allclose(result.turnover.to_numpy(), [1.0, 0.05 / 1.05, 0.10 / 1.10])
    np.testing.assert_allclose(result.returns.to_numpy(), [0.05, 0.10, 0.0])
    assert result.wealth.iloc[-1] == pytest.approx(1.05 * 1.10)
    assert list(result.turnover.index) == list(frame(HAND_RETURNS).index[1:4])


def test_transaction_costs_hand_computed():
    c = 25 / 10_000  # 25 bps
    result = run_backtest(frame(HAND_RETURNS), equal_weight(), BacktestConfig(lookback=2, rebalance=1, cost_bps=25))
    tau2, tau3 = 0.05 / 1.05, 0.10 / 1.10
    expected = [
        1.05 * (1 - c * 1.0) * (1 - c * tau2) - 1,  # initial purchase cost is paid on the first day
        1.10 * (1 - c * tau3) - 1,
        0.0,
    ]
    np.testing.assert_allclose(result.returns.to_numpy(), expected, rtol=1e-12)
    np.testing.assert_allclose(result.costs.to_numpy(), [c, c * tau2, c * tau3])
    np.testing.assert_allclose(result.gross_returns.to_numpy(), [0.05, 0.10, 0.0])


def test_full_switch_costs_twice_the_rate():
    # Strategy alternates 100 % A / 100 % B: each switch sells 100 % and buys 100 % -> tau = 2.
    calls = {"n": 0}

    def flip(window):
        calls["n"] += 1
        return pd.Series({"A": 1.0, "B": 0.0} if calls["n"] % 2 else {"A": 0.0, "B": 1.0})

    rets = frame([[0, 0]] * 6)
    result = run_backtest(rets, flip, BacktestConfig(lookback=2, rebalance=1, cost_bps=10))
    np.testing.assert_allclose(result.turnover.to_numpy(), [1, 2, 2, 2])
    np.testing.assert_allclose(result.costs.to_numpy(), [0.001, 0.002, 0.002, 0.002])


def test_costs_scale_wealth_by_product_of_one_minus_cost(daily_returns):
    # Proportional costs don't change the weight path, so:
    #   wealth_with_costs = wealth_without_costs * prod(1 - c * tau_k)
    free = run_backtest(daily_returns, equal_weight(), BacktestConfig(lookback=40, rebalance=5, cost_bps=0))
    paid = run_backtest(daily_returns, equal_weight(), BacktestConfig(lookback=40, rebalance=5, cost_bps=30))
    pd.testing.assert_series_equal(free.turnover, paid.turnover)
    drag = np.prod(1 - 0.003 * paid.turnover.to_numpy())
    assert paid.wealth.iloc[-1] == pytest.approx(free.wealth.iloc[-1] * drag, rel=1e-12)
    assert paid.wealth.iloc[-1] < free.wealth.iloc[-1]


def test_zero_cost_net_equals_gross(daily_returns):
    result = run_backtest(daily_returns, equal_weight(), BacktestConfig(lookback=30, rebalance="M", cost_bps=0))
    pd.testing.assert_series_equal(result.returns, result.gross_returns)


def test_buy_and_hold_drifts_and_never_trades_again(daily_returns):
    w0 = pd.Series({"AAA": 0.2, "BBB": 0.3, "CCC": 0.5})
    cfg = BacktestConfig(lookback=20, rebalance=None, cost_bps=10)
    result = run_backtest(daily_returns, fixed_weights(w0), cfg)
    assert list(result.turnover) == [1.0]
    # Buy-and-hold value = (1 - c) * sum_i w_i * growth of asset i after the allocation date
    growth = (1 + daily_returns.iloc[20:]).prod()
    assert result.wealth.iloc[-1] == pytest.approx((1 - 0.001) * float(w0 @ growth), rel=1e-12)


def test_constant_prices_mean_no_turnover_after_initial():
    result = run_backtest(frame([[0, 0]] * 10), equal_weight(), BacktestConfig(lookback=3, rebalance=2))
    assert result.turnover.iloc[0] == 1.0
    np.testing.assert_allclose(result.turnover.iloc[1:], 0.0, atol=1e-15)


# --------------------------------------------------------------------------- #
# No look-ahead
# --------------------------------------------------------------------------- #
def test_strategy_never_sees_past_the_rebalance_date(daily_returns):
    seen: list[tuple[pd.Timestamp, int]] = []

    def spy(window: pd.DataFrame) -> pd.Series:
        seen.append((window.index[-1], len(window)))
        return pd.Series(1 / window.shape[1], index=window.columns)

    cfg = BacktestConfig(lookback=60, rebalance="M")
    result = run_backtest(daily_returns, spy, cfg)
    assert [d for d, _ in seen] == list(result.weights.index)   # last row seen == decision date
    assert all(n == 60 for _, n in seen)                          # exactly `lookback` rows
    assert all(result.returns.index > result.weights.index[0])    # performance starts after first trade


@pytest.mark.parametrize("strategy_factory", [equal_weight, lambda: min_variance_strategy(0.6),
                                              lambda: max_sharpe_strategy(0.03, 0.6)])
def test_poisoned_future_does_not_change_the_past(daily_returns, strategy_factory):
    cfg = BacktestConfig(lookback=60, rebalance="M", cost_bps=10)
    cut = daily_returns.index[200]
    poisoned = daily_returns.copy()
    poisoned.loc[poisoned.index > cut] = np.random.default_rng(1).uniform(-0.5, 0.5, poisoned.loc[poisoned.index > cut].shape)

    clean = run_backtest(daily_returns, strategy_factory(), cfg)
    dirty = run_backtest(poisoned, strategy_factory(), cfg)

    pd.testing.assert_frame_equal(clean.weights.loc[:cut], dirty.weights.loc[:cut])
    pd.testing.assert_series_equal(clean.returns.loc[:cut], dirty.returns.loc[:cut])
    assert not clean.returns.loc[cut:].iloc[1:].equals(dirty.returns.loc[cut:].iloc[1:])  # sanity: future did change


def test_trade_decided_at_close_does_not_earn_that_days_return():
    # "Chase the latest winner". On d2 asset A jumps 50 %; the strategy sees it (d2 is in its
    # window) and switches into A — but that switch happens at the close, so the 50 % must NOT
    # appear in the portfolio's d2 return.
    def chase_winner(window):
        winner = window.iloc[-1].idxmax()
        return pd.Series({c: float(c == winner) for c in window.columns})

    rets = frame([[0.0, 0.01], [0.0, 0.01], [0.50, 0.0], [0.0, 0.0]])
    result = run_backtest(rets, chase_winner, BacktestConfig(lookback=2, rebalance=1, cost_bps=0))
    np.testing.assert_allclose(result.returns.to_numpy(), [0.0, 0.0])
    assert result.weights.loc[rets.index[2], "A"] == 1.0  # it did switch, at the close


def test_strategy_cannot_modify_backtest_data(daily_returns):
    original = daily_returns.copy()

    def vandal(window):
        window.iloc[:, :] = 9.99  # mutating its own copy must not leak into the engine
        return pd.Series(1 / window.shape[1], index=window.columns)

    run_backtest(daily_returns, vandal, BacktestConfig(lookback=30, rebalance="M"))
    pd.testing.assert_frame_equal(daily_returns, original)


# --------------------------------------------------------------------------- #
# Schedule
# --------------------------------------------------------------------------- #
def test_monthly_rebalances_on_last_trading_day_of_each_month(daily_returns):
    cfg = BacktestConfig(lookback=30, rebalance="M")
    result = run_backtest(daily_returns, equal_weight(), cfg)
    month_ends = daily_returns.groupby(daily_returns.index.to_period("M")).tail(1).index
    first, last = daily_returns.index[29], daily_returns.index[-1]
    expected = [first] + [d for d in month_ends if first < d < last]
    assert list(result.weights.index) == expected


def test_every_h_days_schedule():
    idx = pd.bdate_range("2024-01-01", periods=20)
    assert rebalance_positions(idx, 5, first=4) == [9, 14]          # 19 is the last day: no trade
    assert rebalance_positions(idx, None, first=4) == []


def test_insufficient_history_raises(daily_returns):
    with pytest.raises(InsufficientDataError, match="lookback"):
        run_backtest(daily_returns.iloc[:50], equal_weight(), BacktestConfig(lookback=60))


# --------------------------------------------------------------------------- #
# Failures and validation
# --------------------------------------------------------------------------- #
def test_failed_rebalance_holds_drifted_weights_and_is_recorded(daily_returns):
    calls = {"n": 0}

    def sometimes_fails(window):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OptimizationError("simulated solver failure")
        return pd.Series(1 / window.shape[1], index=window.columns)

    result = run_backtest(daily_returns, sometimes_fails, BacktestConfig(lookback=30, rebalance="M"))
    assert len(result.skipped_rebalances) == 1
    skipped_date, reason = result.skipped_rebalances[0]
    assert "simulated" in reason and skipped_date not in result.weights.index
    assert calls["n"] == len(result.weights) + 1


def test_failed_first_allocation_raises(daily_returns):
    def broken(window):
        raise OptimizationError("cannot allocate")

    with pytest.raises(OptimizationError, match="cannot allocate"):
        run_backtest(daily_returns, broken, BacktestConfig(lookback=30))


@pytest.mark.parametrize("bad", [pd.Series({"AAA": 0.7, "BBB": 0.7, "CCC": -0.4}),
                                 pd.Series({"AAA": 0.5, "BBB": 0.6, "CCC": 0.0}),
                                 pd.Series({"AAA": 1.0}),
                                 pd.Series({"AAA": np.nan, "BBB": 0.5, "CCC": 0.5})])
def test_invalid_strategy_weights_rejected(daily_returns, bad):
    with pytest.raises(ValueError):
        run_backtest(daily_returns, fixed_weights(bad), BacktestConfig(lookback=30))


def test_missing_returns_rejected_not_filled(daily_returns):
    gappy = daily_returns.copy()
    gappy.iloc[100, 1] = np.nan
    with pytest.raises(DataError, match="missing"):
        run_backtest(gappy, equal_weight(), BacktestConfig(lookback=30))


@pytest.mark.parametrize("kwargs", [{"lookback": 1}, {"rebalance": 0}, {"rebalance": "D"},
                                    {"rebalance": True}, {"cost_bps": -1}, {"cost_bps": 10_000}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        BacktestConfig(**kwargs)


# --------------------------------------------------------------------------- #
# Optimiser strategies and the summary table
# --------------------------------------------------------------------------- #
def test_optimiser_strategies_respect_caps_at_every_rebalance(daily_returns):
    cfg = BacktestConfig(lookback=60, rebalance="M", cost_bps=10)
    for strategy in (min_variance_strategy(0.5), max_sharpe_strategy(0.02, 0.5)):
        w = run_backtest(daily_returns, strategy, cfg).weights
        np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-8)
        assert (w.to_numpy() >= 0).all() and (w.to_numpy() <= 0.5 + 1e-8).all()


def test_summary_table(daily_returns):
    cfg = BacktestConfig(lookback=60, rebalance="M", cost_bps=10)
    results = [run_backtest(daily_returns, equal_weight(), cfg, name="Equal weight"),
               run_backtest(daily_returns, min_variance_strategy(0.6), cfg, name="Min variance")]
    benchmark = daily_returns["AAA"].rename("AAA buy & hold")
    table = backtest_summary(results, rf_annual=0.03, benchmark=benchmark)
    assert list(table.index) == ["Equal weight", "Min variance", "AAA buy & hold"]
    assert {"sharpe", "max_drawdown", "annual_turnover", "total_costs"} <= set(table.columns)
    assert table.loc["AAA buy & hold", "total_costs"] == 0
    assert table.loc["Equal weight", "total_costs"] == pytest.approx(results[0].costs.sum())


def test_summary_rejects_benchmark_with_missing_dates(daily_returns):
    result = run_backtest(daily_returns, equal_weight(), BacktestConfig(lookback=30))
    with pytest.raises(DataError, match="Benchmark"):
        backtest_summary([result], benchmark=daily_returns["AAA"].iloc[:-10])
