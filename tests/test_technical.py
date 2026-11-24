"""Tests for quantedge.technical: hand-computed indicators, signal consistency, candlesticks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantedge.exceptions import DataError, InsufficientDataError
from quantedge.indicators import sma
from quantedge.technical import (
    BEARISH,
    BULLISH,
    bollinger_bands,
    candlestick_patterns,
    ema,
    latest_readings,
    macd,
    rsi,
    support_resistance,
    technical_signals,
    trin,
    validate_ohlc,
    volume_confirmation,
)


def series(values, start="2024-01-01") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


@pytest.fixture
def walk() -> pd.Series:
    rng = np.random.default_rng(3)
    return series(100 * np.cumprod(1 + rng.normal(0.0003, 0.015, 400)))


# --------------------------------------------------------------------------- #
# EMA, Bollinger, MACD, RSI
# --------------------------------------------------------------------------- #
def test_ema_hand_computed():
    e = ema(series([1, 2, 3, 4, 5]), 3)              # multiplier 0.5, seed = mean(1, 2, 3)
    assert e.iloc[:2].isna().all()
    np.testing.assert_allclose(e.iloc[2:], [2.0, 3.0, 4.0])


def test_ema_weights_recent_prices_more_than_sma(walk):
    e, s = ema(walk, 20), sma(walk, 20)
    jump = walk.copy()
    jump.iloc[-1] *= 1.2
    assert ema(jump, 20).iloc[-1] - e.iloc[-1] > sma(jump, 20).iloc[-1] - s.iloc[-1]


def test_ema_matches_recursive_definition(walk):
    n = 10
    e = ema(walk, n)
    m = 2 / (n + 1)
    for t in range(n, 40):
        assert e.iloc[t] == pytest.approx(e.iloc[t - 1] + m * (walk.iloc[t] - e.iloc[t - 1]))


def test_bollinger_population_sd_and_percent_b(walk):
    bb = bollinger_bands(walk, 20, 2)
    window = walk.iloc[-20:]
    sd = window.std(ddof=0)
    assert bb["upper"].iloc[-1] == pytest.approx(window.mean() + 2 * sd)
    assert bb["lower"].iloc[-1] == pytest.approx(window.mean() - 2 * sd)
    pb = (walk.iloc[-1] - bb["lower"].iloc[-1]) / (bb["upper"].iloc[-1] - bb["lower"].iloc[-1])
    assert bb["percent_b"].iloc[-1] == pytest.approx(pb)
    assert bb["middle"].iloc[:19].isna().all()


def test_bollinger_flat_prices_have_undefined_percent_b():
    bb = bollinger_bands(series([5.0] * 25), 20)
    assert np.isnan(bb["percent_b"].iloc[-1]) and bb["bandwidth"].iloc[-1] == 0


def test_macd_definition_and_lecture_signal_line(walk):
    m = macd(walk)
    np.testing.assert_allclose(m["macd"], ema(walk, 12) - ema(walk, 26), equal_nan=True)
    first_signal = m["signal"].first_valid_index()
    assert walk.index.get_loc(first_signal) == 25 + 8          # 26-day EMA, then 9 values of MACD
    assert m["signal"].iloc[-1] == pytest.approx(m["macd"].iloc[-9:].mean())
    np.testing.assert_allclose(m["histogram"], m["macd"] - m["signal"], equal_nan=True)


def test_macd_ema_signal_option(walk):
    m = macd(walk, signal_kind="ema")
    assert m["signal"].iloc[-1] != pytest.approx(macd(walk)["signal"].iloc[-1])
    with pytest.raises(ValueError):
        macd(walk, signal_kind="wma")
    with pytest.raises(ValueError):
        macd(walk, fast=26, slow=12)


def test_rsi_hand_computed():
    r = rsi(series([10, 11, 10, 12, 12, 13]), 3)
    assert r.iloc[:3].isna().all()
    assert r.iloc[3] == pytest.approx(75.0)                    # gains 1,0,2 / losses 0,1,0
    assert r.iloc[4] == pytest.approx(100 - 100 / 3)           # gains 0,2,0 / losses 1,0,0
    assert r.iloc[5] == pytest.approx(100.0)                   # no losses


def test_rsi_flat_is_undefined_not_fifty():
    assert np.isnan(rsi(series([7.0] * 20), 14).iloc[-1])


def test_rsi_bounds_and_wilder(walk):
    for method in ("simple", "wilder"):
        r = rsi(walk, 14, method).dropna()
        assert ((r >= 0) & (r <= 100)).all()
    w = rsi(walk, 14, "wilder")
    s = rsi(walk, 14, "simple")
    assert w.first_valid_index() == s.first_valid_index()
    assert w.dropna().iloc[0] == pytest.approx(s.dropna().iloc[0])   # Wilder seeds with the simple mean


def test_support_resistance_uses_only_past_prices(walk):
    sr = support_resistance(walk, 10)
    assert sr["resistance"].iloc[50] == pytest.approx(walk.iloc[40:50].max())
    poisoned = walk.copy()
    poisoned.iloc[51:] = 1e6
    pd.testing.assert_frame_equal(support_resistance(poisoned, 10).iloc[:51].drop(columns="price"),
                                  sr.iloc[:51].drop(columns="price"))


def test_too_short_inputs():
    with pytest.raises(InsufficientDataError):
        rsi(series([1, 2, 3]), 14)
    with pytest.raises(InsufficientDataError):
        macd(series(range(1, 30)))


# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #
def test_signals_agree_with_indicator_values(walk):
    ev = technical_signals(walk)
    m, r, bb = macd(walk), rsi(walk), bollinger_bands(walk)
    assert len(ev) > 0
    for _, e in ev.iterrows():
        d = e["date"]
        prev = walk.index[walk.index.get_loc(d) - 1]
        if e["event"] == "MACD crossed above signal":
            assert m.at[d, "macd"] > m.at[d, "signal"] and m.at[prev, "macd"] <= m.at[prev, "signal"]
            assert e["signal"] == BULLISH
        elif e["event"].startswith("RSI rose above"):
            assert r[d] > 70 >= r[prev] and e["signal"] == BEARISH
        elif e["event"].startswith("RSI fell below"):
            assert r[d] < 30 <= r[prev] and e["signal"] == BULLISH
        elif e["event"] == "Close broke below lower band":
            assert walk[d] < bb.at[d, "lower"] and walk[prev] >= bb.at[prev, "lower"]


def test_latest_readings_interpretation(walk):
    up = series(np.linspace(100, 300, 260))
    lr = latest_readings(up)
    assert lr.loc["SMA trend", "signal"] == BULLISH
    assert lr.loc["RSI", "interpretation"] == "overbought"
    assert set(latest_readings(walk).index) == {"SMA trend", "MACD", "RSI", "Bollinger"}


# --------------------------------------------------------------------------- #
# Candlesticks
# --------------------------------------------------------------------------- #
def candles(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"],
                        index=pd.bdate_range("2024-01-01", periods=len(rows)), dtype=float)


def downtrend(n=6, start=110.0):
    return [(start - 2 * i + 1, start - 2 * i + 1.5, start - 2 * i - 0.5, start - 2 * i) for i in range(n)]


def uptrend(n=6, start=90.0):
    return [(start + 2 * i - 1, start + 2 * i + 0.5, start + 2 * i - 1.5, start + 2 * i) for i in range(n)]


def found(df, pattern):
    return df[df["pattern"] == pattern]


def test_hammer_in_downtrend_and_hanging_man_in_uptrend():
    hammer = (99.0, 99.6, 95.0, 99.5)                         # body 0.5, lower shadow 4, upper 0.1
    res = candlestick_patterns(candles(downtrend() + [hammer]))
    hit = found(res, "Hammer")
    assert len(hit) == 1 and hit["signal"].iloc[0] == BULLISH and hit["needs_confirmation"].iloc[0]
    hang = (110.0, 110.6, 106.0, 110.5)
    res = candlestick_patterns(candles(uptrend() + [hang]))
    assert len(found(res, "Hanging man")) == 1


def test_doji():
    res = candlestick_patterns(candles(uptrend() + [(101.0, 103.0, 99.0, 101.1)]))
    hit = found(res, "Doji")
    assert len(hit) == 1 and hit["signal"].iloc[0] == BEARISH   # doji after an uptrend warns of a top


def test_engulfing_patterns():
    small_red = (99.5, 100.0, 98.5, 99.0)
    big_green = (98.8, 101.5, 98.5, 101.0)
    res = candlestick_patterns(candles(downtrend() + [small_red, big_green]))
    assert len(found(res, "Bullish engulfing")) == 1
    small_green = (101.0, 101.8, 100.8, 101.5)
    big_red = (101.7, 102.0, 99.5, 100.5)
    res = candlestick_patterns(candles(uptrend() + [small_green, big_red]))
    assert len(found(res, "Bearish engulfing")) == 1


def test_harami():
    long_red = (100.0, 100.5, 94.5, 95.0)
    small_inside = (96.0, 97.2, 95.8, 97.0)
    res = candlestick_patterns(candles(downtrend() + [long_red, small_inside]))
    assert len(found(res, "Bullish harami")) == 1


def test_morning_and_evening_star():
    rows = downtrend() + [(100.0, 100.3, 94.7, 95.0), (93.6, 94.0, 93.0, 93.8), (94.5, 99.0, 94.4, 98.5)]
    assert len(found(candlestick_patterns(candles(rows)), "Morning star")) == 1
    rows = uptrend() + [(100.0, 105.3, 99.8, 105.0), (106.2, 107.0, 106.0, 106.4), (105.5, 105.6, 101.0, 101.5)]
    assert len(found(candlestick_patterns(candles(rows)), "Evening star")) == 1


def test_no_patterns_in_plain_trend():
    res = candlestick_patterns(candles(uptrend(15)))
    assert res[res["pattern"].isin(["Hammer", "Morning star", "Bullish engulfing"])].empty


@pytest.mark.parametrize("bad", [
    [(10, 9, 8, 9.5)],                       # high below open/close
    [(10, 11, 10.5, 10.8)],                  # low above open
    [(10, 11, 9, np.nan)],
])
def test_invalid_ohlc_rejected(bad):
    with pytest.raises(DataError):
        validate_ohlc(candles(bad))


def test_ohlc_missing_column():
    with pytest.raises(DataError, match="Close"):
        validate_ohlc(candles([(1, 2, 0.5, 1.5)]).drop(columns="Close"))


# --------------------------------------------------------------------------- #
# Breadth and volume
# --------------------------------------------------------------------------- #
def test_trin():
    assert trin(2000, 1000, 300e6, 100e6) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        trin(0, 1, 1, 1)


def test_volume_confirmation_table():
    p = series([10, 11, 12, 11, 12, 13])
    v = series([100, 100, 300, 300, 50, 50])
    t = volume_confirmation(p, v, window=2)
    assert t["strength"].iloc[:2].isna().all() if t["strength"].iloc[:2].dtype != object else \
        all(x is None for x in t["strength"].iloc[:2])
    assert t["direction"].iloc[2] == "up" and t["strength"].iloc[2] == "strong"
    assert t["direction"].iloc[3] == "down" and t["strength"].iloc[3] == "strong"
    assert t["direction"].iloc[4] == "up" and t["strength"].iloc[4] == "weak"


def test_todays_price_is_not_its_own_resistance():
    p = series(np.arange(1.0, 31.0))                  # rising every day
    sr = support_resistance(p, 5)
    assert (sr["resistance"].dropna() < sr["price"][sr["resistance"].notna()]).all()
    assert sr["resistance"].iloc[10] == pytest.approx(10.0)   # max of days 5..9 (prices 6..10)
