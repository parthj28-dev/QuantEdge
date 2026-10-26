"""Walk-forward historical backtest with drift, turnover and transaction costs.

Timeline rules (the "no look-ahead" contract)
---------------------------------------------
Let p be a trading day (a row of the returns table). Return r_p is the change from
the close of day p-1 to the close of day p.

1. Decisions happen at the CLOSE of a rebalance day t. The strategy receives ONLY a
   copy of the last `lookback` returns ending on t (rows t-L+1 ... t). It never sees
   a row after t — it is not even given a reference to the full table.
2. Trades execute at the close of t, so the new weights earn returns from t+1 on.
   The return on day t itself is earned by the OLD (drifted) weights.
3. The first allocation happens on the first day with `lookback` returns of history
   (row L-1). Before that the portfolio does not exist, so performance starts the
   next day. No trade is placed on the final day (it could never earn anything).

Between rebalances the portfolio is left alone and its weights DRIFT:
    w_i,p = w_i,p-1 * (1 + r_i,p) / (1 + r_portfolio,p)

Transaction costs
-----------------
    turnover tau = sum_i | w_target_i - w_drifted_i |   (buys + sells, as a fraction of value)
    cost         = c * tau                              (c = cost_bps / 10,000, per unit traded)
    value after  = value before * (1 - c * tau)

The initial purchase from cash has tau = 1. Switching 100 % from one asset to another
has tau = 2 (sell 100 %, buy 100 %). Approximation: tau is measured on the pre-cost
value; the exact self-financing amount differs only by a term of order c^2.

Failed rebalances
-----------------
If a strategy raises a QuantEdgeError at a scheduled rebalance (e.g. the optimiser
fails on that window), the portfolio keeps its drifted weights and the date and
reason are recorded in `skipped_rebalances`. If the very first allocation fails,
there is nothing to hold, so the error is raised.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import TRADING_DAYS_PER_YEAR
from .exceptions import DataError, InsufficientDataError, QuantEdgeError
from .metrics import annualized_covariance, check_returns, performance_summary
from .optimize import Caps, historical_inputs, max_sharpe, min_variance

Strategy = Callable[[pd.DataFrame], pd.Series]

_CALENDAR_FREQUENCIES = {"W": "W", "M": "M", "Q": "Q"}
_WEIGHT_TOLERANCE = 1e-6


# --------------------------------------------------------------------------- #
# Configuration and results
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BacktestConfig:
    """lookback: daily returns used for each estimate.
    rebalance: "W" / "M" / "Q" (last trading day of each week / month / quarter),
               an int H (every H trading days), or None (buy and hold, never rebalance).
    cost_bps:  proportional cost per unit of value traded, in basis points.
    """

    lookback: int = 252
    rebalance: str | int | None = "M"
    cost_bps: float = 10.0

    def __post_init__(self) -> None:
        if not isinstance(self.lookback, int) or self.lookback < 2:
            raise ValueError("lookback must be an integer of at least 2 returns.")
        if isinstance(self.rebalance, bool) or not (
            self.rebalance is None
            or (isinstance(self.rebalance, int) and self.rebalance >= 1)
            or self.rebalance in _CALENDAR_FREQUENCIES
        ):
            raise ValueError("rebalance must be 'W', 'M', 'Q', a positive integer, or None.")
        if not np.isfinite(self.cost_bps) or not 0 <= self.cost_bps < 10_000:
            raise ValueError("cost_bps must be between 0 and 10,000.")


@dataclass(frozen=True)
class BacktestResult:
    name: str
    returns: pd.Series            # daily NET returns, first day after the initial allocation onward
    gross_returns: pd.Series      # same days, before transaction costs
    wealth: pd.Series             # value of 1 invested; starts at the allocation date with 1.0
    weights: pd.DataFrame         # target weights at each executed rebalance (incl. initial)
    turnover: pd.Series           # tau at each executed rebalance (initial = 1)
    costs: pd.Series              # cost as a fraction of portfolio value at each rebalance
    config: BacktestConfig
    skipped_rebalances: tuple[tuple[pd.Timestamp, str], ...] = field(default=())


# --------------------------------------------------------------------------- #
# Strategies: functions from an estimation window to target weights
# --------------------------------------------------------------------------- #
def equal_weight() -> Strategy:
    def strategy(window: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0 / window.shape[1], index=window.columns)

    return strategy


def fixed_weights(weights: Mapping[str, float] | pd.Series) -> Strategy:
    target = pd.Series(weights, dtype=float)

    def strategy(window: pd.DataFrame) -> pd.Series:
        return target

    return strategy


def min_variance_strategy(max_weight: Caps = 1.0) -> Strategy:
    def strategy(window: pd.DataFrame) -> pd.Series:
        return min_variance(annualized_covariance(window), max_weight=max_weight).weights

    return strategy


def max_sharpe_strategy(rf_annual: float = 0.0, max_weight: Caps = 1.0) -> Strategy:
    def strategy(window: pd.DataFrame) -> pd.Series:
        mu, cov = historical_inputs(window)
        return max_sharpe(mu, cov, rf_annual=rf_annual, max_weight=max_weight).weights

    return strategy


# --------------------------------------------------------------------------- #
# Schedule
# --------------------------------------------------------------------------- #
def rebalance_positions(index: pd.DatetimeIndex, rebalance: str | int | None, first: int) -> list[int]:
    """Row positions of scheduled rebalances strictly after `first` and before the last row."""
    last = len(index) - 1
    if rebalance is None:
        return []
    if isinstance(rebalance, int):
        return list(range(first + rebalance, last, rebalance))
    periods = index.to_period(_CALENDAR_FREQUENCIES[rebalance])
    period_ends = np.flatnonzero(periods[:-1] != periods[1:])  # last row of each period
    return [int(p) for p in period_ends if first < p < last]


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
def run_backtest(
    returns: pd.DataFrame,
    strategy: Strategy,
    config: BacktestConfig = BacktestConfig(),
    *,
    name: str = "strategy",
) -> BacktestResult:
    """Simulate `strategy` on daily simple `returns` (dates x assets) without look-ahead."""
    if not isinstance(returns, pd.DataFrame) or returns.shape[1] == 0:
        raise TypeError("returns must be a DataFrame with one column per asset.")
    check_returns(returns, min_obs=2)
    if returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise DataError("Return dates must be unique and sorted in increasing order.")

    lookback = config.lookback
    n_days = len(returns)
    if n_days < lookback + 2:
        raise InsufficientDataError(
            f"A {lookback}-day lookback needs at least {lookback + 2} daily returns "
            f"(history plus days to trade); got {n_days}. Shorten the lookback or extend the dates."
        )

    assets = list(returns.columns)
    values = returns.to_numpy(dtype=float)
    dates = returns.index
    start = lookback - 1                                   # first row with a full window
    schedule = set(rebalance_positions(dates, config.rebalance, start))
    c = config.cost_bps / 10_000.0

    def decide(p: int) -> np.ndarray:
        # The ONLY data the strategy ever receives: a copy of rows p-L+1 ... p.
        window = returns.iloc[p - lookback + 1 : p + 1].copy()
        return _validated_weights(strategy(window), assets, dates[p])

    # Initial allocation from cash at the close of `start`.
    weights = decide(start)
    pending_cost = c * float(np.abs(weights).sum())        # tau = 1 when buying from cash
    trade_dates, trade_weights = [dates[start]], [weights.copy()]
    turnovers, costs = [1.0], [pending_cost]
    skipped: list[tuple[pd.Timestamp, str]] = []

    gross = np.empty(n_days - start - 1)
    net = np.empty_like(gross)
    for k, p in enumerate(range(start + 1, n_days)):
        r = values[p]
        day_gross = float(weights @ r)                     # earned by weights held overnight
        weights = weights * (1.0 + r) / (1.0 + day_gross)  # drift to the close of day p
        factor = (1.0 + day_gross) * (1.0 - pending_cost)
        pending_cost = 0.0

        if p in schedule:
            try:
                target = decide(p)
            except QuantEdgeError as exc:
                skipped.append((dates[p], str(exc)))
            else:
                tau = float(np.abs(target - weights).sum())
                factor *= 1.0 - c * tau
                weights = target
                trade_dates.append(dates[p])
                trade_weights.append(target.copy())
                turnovers.append(tau)
                costs.append(c * tau)

        gross[k] = day_gross
        net[k] = factor - 1.0

    perf_index = dates[start + 1 :]
    net_returns = pd.Series(net, index=perf_index, name=name)
    wealth = pd.concat([pd.Series([1.0], index=[dates[start]]), (1.0 + net_returns).cumprod()])
    return BacktestResult(
        name=name,
        returns=net_returns,
        gross_returns=pd.Series(gross, index=perf_index, name=name),
        wealth=wealth.rename(name),
        weights=pd.DataFrame(trade_weights, index=pd.DatetimeIndex(trade_dates), columns=assets),
        turnover=pd.Series(turnovers, index=pd.DatetimeIndex(trade_dates), name="turnover"),
        costs=pd.Series(costs, index=pd.DatetimeIndex(trade_dates), name="cost"),
        config=config,
        skipped_rebalances=tuple(skipped),
    )


def _validated_weights(target: pd.Series, assets: list[str], when: pd.Timestamp) -> np.ndarray:
    """Strategies must return finite, long-only weights summing to 1 for exactly these assets."""
    if not isinstance(target, pd.Series):
        raise TypeError(f"Strategy must return a pandas Series of weights (on {when.date()}).")
    if set(target.index) != set(assets):
        raise ValueError(f"Strategy weights on {when.date()} do not match the assets {assets}.")
    w = target.reindex(assets).to_numpy(dtype=float)
    if not np.isfinite(w).all() or (w < -_WEIGHT_TOLERANCE).any() or abs(w.sum() - 1) > _WEIGHT_TOLERANCE:
        raise ValueError(
            f"Strategy weights on {when.date()} must be finite, non-negative and sum to 1 "
            f"(got sum {w.sum():.6f})."
        )
    w = np.clip(w, 0.0, None)
    return w / w.sum()


# --------------------------------------------------------------------------- #
# Comparison table
# --------------------------------------------------------------------------- #
def backtest_summary(
    results: list[BacktestResult],
    rf_annual: float = 0.0,
    benchmark: pd.Series | None = None,
) -> pd.DataFrame:
    """Risk/return statistics for each strategy (and an optional benchmark) over identical dates."""
    if not results:
        raise ValueError("No backtest results to summarise.")
    index = results[0].returns.index
    if any(not r.returns.index.equals(index) for r in results):
        raise DataError("All backtests must cover the same dates to be compared.")

    table = performance_summary(pd.concat([r.returns for r in results], axis=1), rf_annual)
    years = len(index) / TRADING_DAYS_PER_YEAR
    table["rebalances"] = [len(r.turnover) - 1 for r in results]
    table["annual_turnover"] = [r.turnover.iloc[1:].sum() / years for r in results]
    table["total_costs"] = [r.costs.sum() for r in results]
    table["skipped_rebalances"] = [len(r.skipped_rebalances) for r in results]

    if benchmark is not None:
        aligned = benchmark.reindex(index)
        if aligned.isna().any():
            raise DataError(
                f"Benchmark has no return on {int(aligned.isna().sum())} backtest date(s); "
                "fetch it together with the assets so the dates match."
            )
        bench_row = performance_summary(aligned.rename(str(benchmark.name or "benchmark")).to_frame(), rf_annual)
        table = pd.concat([table, bench_row])
        table.loc[bench_row.index, ["rebalances", "annual_turnover", "total_costs", "skipped_rebalances"]] = 0
    return table
