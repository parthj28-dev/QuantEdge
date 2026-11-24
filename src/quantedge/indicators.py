"""Simple moving averages and trend signals (v2.1). See docs/ARCHITECTURE.md, Section 4.9.

    SMA_n(t) = (1/n) * sum_{k=0}^{n-1} P_{t-k}       on adjusted closes

* The first n-1 values are undefined (NaN), never filled: an average of fewer prices
  is a different statistic.
* An asset is ABOVE TREND when P_t > SMA_n(t) strictly.
* Golden cross: the fast SMA moves from <= to > the slow SMA. Death cross: the reverse.
  No cross is reported until both SMAs exist on two consecutive days.

Inside a backtest the strategy only receives returns. Because P_t / SMA_n(t) is
scale-free it can be computed exactly from relative prices 1, (1+r_1), (1+r_1)(1+r_2), ...
over the last n-1 returns (`price_to_sma_from_returns`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .exceptions import DataError, InsufficientDataError

SeriesOrFrame = pd.Series | pd.DataFrame

GOLDEN_CROSS = "golden cross"
DEATH_CROSS = "death cross"


def _check_window(window: int) -> None:
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 2:
        raise ValueError(f"A moving-average window must be an integer of at least 2, got {window!r}.")


def _check_prices(prices: SeriesOrFrame, window: int) -> None:
    if not isinstance(prices, (pd.Series, pd.DataFrame)):
        raise TypeError("prices must be a pandas Series or DataFrame.")
    values = prices.to_numpy(dtype=float)
    if np.isnan(values).any() or np.isinf(values).any():
        raise DataError("Prices contain missing or infinite values; validate them first.")
    if (values <= 0).any():
        raise DataError("Prices must be strictly positive.")
    if len(prices) < window:
        raise InsufficientDataError(
            f"A {window}-day moving average needs at least {window} prices; got {len(prices)}."
        )


def sma(prices: SeriesOrFrame, window: int) -> SeriesOrFrame:
    """Simple moving average; the first window-1 values are NaN (undefined)."""
    _check_window(window)
    _check_prices(prices, window)
    return prices.rolling(window=window, min_periods=window).mean()


def above_trend(prices: SeriesOrFrame, window: int) -> SeriesOrFrame:
    """True where P_t > SMA_n(t) strictly; NaN (as float) where the SMA is undefined."""
    avg = sma(prices, window)
    out = (prices > avg).astype(float)
    return out.where(avg.notna())


def crosses(prices: pd.Series | pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """Every golden / death cross: columns date, asset, event, price, fast_sma, slow_sma."""
    _check_window(fast)
    _check_window(slow)
    if fast >= slow:
        raise ValueError(f"The fast window ({fast}) must be shorter than the slow window ({slow}).")
    frame = prices.to_frame() if isinstance(prices, pd.Series) else prices
    f, s = sma(frame, fast), sma(frame, slow)
    above = f > s
    both = f.notna() & s.notna()
    valid = both & both.shift(1, fill_value=False)
    prev = above.shift(1, fill_value=False)
    golden = valid & above & ~prev
    death = valid & ~above & prev
    rows = []
    for asset in frame.columns:
        for when in frame.index[golden[asset].to_numpy()]:
            rows.append((when, asset, GOLDEN_CROSS))
        for when in frame.index[death[asset].to_numpy()]:
            rows.append((when, asset, DEATH_CROSS))
    rows.sort(key=lambda r: (r[0], str(r[1])))
    out = pd.DataFrame(rows, columns=["date", "asset", "event"])
    out["price"] = [frame.at[d, a] for d, a, _ in rows]
    out["fast_sma"] = [f.at[d, a] for d, a, _ in rows]
    out["slow_sma"] = [s.at[d, a] for d, a, _ in rows]
    return out


def trend_status(prices: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """Latest snapshot per asset: price, both SMAs, distance from the slow SMA, trend and last cross."""
    frame = prices.to_frame() if isinstance(prices, pd.Series) else prices
    f, s = sma(frame, fast), sma(frame, slow)
    last = frame.iloc[-1]
    events = crosses(frame, fast, slow)
    rows = {}
    for asset in frame.columns:
        sl, fl = s[asset].iloc[-1], f[asset].iloc[-1]
        mine = events[events["asset"] == asset]
        rows[asset] = {
            "price": last[asset],
            f"sma_{fast}": fl,
            f"sma_{slow}": sl,
            "distance_from_slow": last[asset] / sl - 1 if pd.notna(sl) else np.nan,
            "above_slow": bool(last[asset] > sl) if pd.notna(sl) else None,
            "fast_above_slow": bool(fl > sl) if pd.notna(sl) and pd.notna(fl) else None,
            "last_cross": mine["event"].iloc[-1] if len(mine) else None,
            "last_cross_date": mine["date"].iloc[-1] if len(mine) else pd.NaT,
        }
    return pd.DataFrame(rows).T


def relative_prices(window_returns: pd.DataFrame | pd.Series, n_prices: int) -> pd.DataFrame | pd.Series:
    """Prices up to a common scale from the last n_prices-1 returns: 1, (1+r1), (1+r1)(1+r2), ..."""
    needed = n_prices - 1
    if len(window_returns) < needed:
        raise InsufficientDataError(
            f"{n_prices} prices need {needed} returns of history; the window has {len(window_returns)}. "
            f"Use a lookback of at least {needed}."
        )
    tail = window_returns.iloc[len(window_returns) - needed:]
    growth = (1.0 + tail).cumprod()
    if isinstance(tail, pd.Series):
        return pd.concat([pd.Series([1.0]), growth.reset_index(drop=True)], ignore_index=True)
    first = pd.DataFrame([np.ones(tail.shape[1])], columns=tail.columns)
    return pd.concat([first, growth.reset_index(drop=True)], ignore_index=True)


def price_to_sma_from_returns(window_returns: pd.DataFrame, window: int) -> pd.Series:
    """P_t / SMA_n(t) for each asset, computed exactly from the last n-1 returns."""
    _check_window(window)
    rel = relative_prices(window_returns, window)
    return rel.iloc[-1] / rel.mean()
