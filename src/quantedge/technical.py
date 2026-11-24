"""Technical analysis indicators, signals and candlestick patterns (v3.0, SAPM lectures 26-29).

Definitions follow the course slides; where practitioners differ, the alternative is an option.

    EMA_t        = EMA_{t-1} + m (P_t - EMA_{t-1}),  m = 2 / (n + 1); seeded with the SMA of the
                   first n prices, so the first n-1 values are undefined (NaN)
    Bollinger    middle = SMA_n, upper/lower = SMA_n +/- k sigma_n  (n = 20, k = 2; population sigma,
                   as in Bollinger's definition); %B = (P - lower)/(upper - lower)
    MACD         MACD = EMA_12 - EMA_26; signal = 9-day SMA of MACD (the lecture's definition;
                   `signal_kind="ema"` gives the common 9-day EMA); histogram = MACD - signal
    RSI          RSI = 100 - 100 / (1 + RS), RS = average gain / average loss over n = 14 changes
                   (simple averages as in the lecture; `method="wilder"` gives Wilder's smoothing).
                   Undefined (NaN) when there were no gains and no losses.

Signals (lecture rules)
    MACD crosses above its signal line -> buy; below -> sell.
    RSI above 70 overbought (sell / short), below 30 oversold (go long).
    Close above the upper Bollinger band -> sell; below the lower band -> long.
    Narrowing bands (low bandwidth) warn of an impending large move.

Candlestick patterns need daily Open/High/Low/Close. The prior trend is the sign of the
close's change over `trend_window` days before the pattern. Single-candle reversal patterns
(hammer, hanging man, doji) need confirmation by the next session; that is reported, not assumed.

Technical analysis is shown for teaching. The course notes that academic evidence on its
profitability is weak; nothing here is a recommendation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .exceptions import DataError, InsufficientDataError
from .indicators import _check_prices, _check_window, sma

SeriesOrFrame = pd.Series | pd.DataFrame
OHLC = ["Open", "High", "Low", "Close"]

BULLISH, BEARISH, NEUTRAL = "bullish", "bearish", "neutral"


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #
def ema(prices: SeriesOrFrame, window: int) -> SeriesOrFrame:
    """Exponential moving average seeded with the first full SMA; earlier values NaN."""
    _check_window(window)
    _check_prices(prices, window)
    return _ema_values(prices, window)


def _ema_values(values: SeriesOrFrame, window: int) -> SeriesOrFrame:
    """EMA of a series that may begin with NaNs (used for the MACD signal line)."""
    frame = values.to_frame() if isinstance(values, pd.Series) else values
    m = 2.0 / (window + 1.0)
    out = pd.DataFrame(np.nan, index=frame.index, columns=frame.columns)
    for col in frame.columns:
        x = frame[col].to_numpy(dtype=float)
        valid = np.flatnonzero(~np.isnan(x))
        if len(valid) < window:
            continue
        first = valid[0]
        if np.isnan(x[first:]).any():
            raise DataError("EMA input has gaps after its first value.")
        seed = first + window - 1
        e = np.full(len(x), np.nan)
        e[seed] = x[first:seed + 1].mean()
        for t in range(seed + 1, len(x)):
            e[t] = e[t - 1] + m * (x[t] - e[t - 1])
        out[col] = e
    return out.iloc[:, 0].rename(values.name) if isinstance(values, pd.Series) else out


def bollinger_bands(prices: pd.Series, window: int = 20, k: float = 2.0, ddof: int = 0) -> pd.DataFrame:
    """Columns: middle, upper, lower, percent_b, bandwidth = (upper - lower) / middle."""
    if not np.isfinite(k) or k <= 0:
        raise ValueError("k (number of standard deviations) must be positive.")
    mid = sma(prices, window)
    sd = prices.rolling(window, min_periods=window).std(ddof=ddof)
    upper, lower = mid + k * sd, mid - k * sd
    width = upper - lower
    return pd.DataFrame({
        "price": prices,
        "middle": mid,
        "upper": upper,
        "lower": lower,
        "percent_b": (prices - lower) / width.where(width > 0),
        "bandwidth": width / mid,
    })


def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9,
         signal_kind: str = "sma") -> pd.DataFrame:
    """Columns: macd, signal, histogram."""
    if fast >= slow:
        raise ValueError(f"The fast EMA ({fast}) must be shorter than the slow EMA ({slow}).")
    _check_window(signal)
    _check_prices(prices, slow + signal - 1)
    line = ema(prices, fast) - ema(prices, slow)
    if signal_kind == "sma":
        sig = line.rolling(signal, min_periods=signal).mean()
    elif signal_kind == "ema":
        sig = _ema_values(line, signal)
    else:
        raise ValueError("signal_kind must be 'sma' (lecture) or 'ema'.")
    return pd.DataFrame({"macd": line, "signal": sig, "histogram": line - sig})


def rsi(prices: pd.Series, window: int = 14, method: str = "simple") -> pd.Series:
    """Relative strength index (0-100). NaN for the first `window` prices and for flat windows."""
    _check_window(window)
    _check_prices(prices, window + 1)
    change = prices.diff()
    gain, loss = change.clip(lower=0.0), (-change).clip(lower=0.0)
    if method == "simple":
        avg_gain = gain.rolling(window, min_periods=window).mean()
        avg_loss = loss.rolling(window, min_periods=window).mean()
    elif method == "wilder":
        avg_gain, avg_loss = _wilder(gain, window), _wilder(loss, window)
    else:
        raise ValueError("method must be 'simple' (lecture) or 'wilder'.")
    out = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)          # avg_loss = 0 -> RS = inf -> 100
    flat = (avg_gain == 0) & (avg_loss == 0)
    return out.mask(flat).rename(prices.name)


def _wilder(x: pd.Series, n: int) -> pd.Series:
    v = x.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    if len(v) > n:
        out[n] = v[1:n + 1].mean()                              # v[0] is the undefined first change
        for t in range(n + 1, len(v)):
            out[t] = (out[t - 1] * (n - 1) + v[t]) / n
    return pd.Series(out, index=x.index)


def support_resistance(prices: pd.Series, window: int = 20) -> pd.DataFrame:
    """Support = lowest and resistance = highest price of the PREVIOUS `window` days (no look-ahead)."""
    _check_window(window)
    _check_prices(prices, window + 1)
    prior = prices.shift(1)
    return pd.DataFrame({"price": prices,
                         "support": prior.rolling(window, min_periods=window).min(),
                         "resistance": prior.rolling(window, min_periods=window).max()})


# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #
def _crossings(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(up, down): a moves from <= b to > b, or from > b to <= b; only where both defined twice."""
    above = a > b
    ok = a.notna() & b.notna()
    ok = ok & ok.shift(1, fill_value=False)
    prev = above.shift(1, fill_value=False)
    return ok & above & ~prev, ok & ~above & prev


def technical_signals(prices: pd.Series, *, rsi_window: int = 14, rsi_upper: float = 70,
                      rsi_lower: float = 30, bb_window: int = 20, bb_k: float = 2.0,
                      macd_signal_kind: str = "sma", rsi_method: str = "simple") -> pd.DataFrame:
    """Every dated event from the lecture's rules: columns date, indicator, event, signal."""
    if not rsi_lower < rsi_upper:
        raise ValueError("The oversold level must be below the overbought level.")
    events: list[tuple] = []
    m = macd(prices, signal_kind=macd_signal_kind)
    up, down = _crossings(m["macd"], m["signal"])
    events += [(d, "MACD", "MACD crossed above signal", BULLISH) for d in prices.index[up]]
    events += [(d, "MACD", "MACD crossed below signal", BEARISH) for d in prices.index[down]]
    zero = pd.Series(0.0, index=prices.index)
    up, down = _crossings(m["macd"], zero)
    events += [(d, "MACD", "MACD crossed above zero", BULLISH) for d in prices.index[up]]
    events += [(d, "MACD", "MACD crossed below zero", BEARISH) for d in prices.index[down]]

    r = rsi(prices, rsi_window, rsi_method)
    upper, lower = pd.Series(float(rsi_upper), index=prices.index), pd.Series(float(rsi_lower), index=prices.index)
    into_ob, _ = _crossings(r, upper)
    _, into_os = _crossings(r, lower)
    events += [(d, "RSI", f"RSI rose above {rsi_upper:g} (overbought)", BEARISH) for d in prices.index[into_ob]]
    events += [(d, "RSI", f"RSI fell below {rsi_lower:g} (oversold)", BULLISH) for d in prices.index[into_os]]

    bb = bollinger_bands(prices, bb_window, bb_k)
    up, _ = _crossings(prices, bb["upper"])
    _, down = _crossings(prices, bb["lower"])
    events += [(d, "Bollinger", "Close broke above upper band", BEARISH) for d in prices.index[up]]
    events += [(d, "Bollinger", "Close broke below lower band", BULLISH) for d in prices.index[down]]

    events.sort(key=lambda e: (e[0], e[1], e[2]))
    return pd.DataFrame(events, columns=["date", "indicator", "event", "signal"])


def latest_readings(prices: pd.Series, *, rsi_window: int = 14, rsi_upper: float = 70, rsi_lower: float = 30,
                    bb_window: int = 20, bb_k: float = 2.0, fast: int = 50, slow: int = 200,
                    macd_signal_kind: str = "sma", rsi_method: str = "simple") -> pd.DataFrame:
    """One row per indicator with its latest value and the lecture's interpretation."""
    rows = []
    p = float(prices.iloc[-1])
    if len(prices) >= slow:
        f, s = float(sma(prices, fast).iloc[-1]), float(sma(prices, slow).iloc[-1])
        rows.append(("SMA trend", f"{fast}-day {f:,.2f} vs {slow}-day {s:,.2f}",
                     "uptrend (fast above slow)" if f > s else "downtrend (fast at or below slow)",
                     BULLISH if f > s else BEARISH))
    m = macd(prices, signal_kind=macd_signal_kind).iloc[-1]
    if pd.notna(m["signal"]):
        trend = "above" if m["macd"] > 0 else "below"
        rows.append(("MACD", f"MACD {m['macd']:.3f}, signal {m['signal']:.3f}",
                     f"MACD {'above' if m['macd'] > m['signal'] else 'at or below'} its signal line, "
                     f"{trend} zero",
                     BULLISH if m["macd"] > m["signal"] else BEARISH))
    r = float(rsi(prices, rsi_window, rsi_method).iloc[-1])
    if np.isfinite(r):
        state = (("overbought", BEARISH) if r > rsi_upper else ("oversold", BULLISH) if r < rsi_lower
                 else ("neutral", NEUTRAL))
        rows.append(("RSI", f"{r:.1f}", state[0], state[1]))
    bb = bollinger_bands(prices, bb_window, bb_k)
    last = bb.iloc[-1]
    if pd.notna(last["upper"]):
        width_rank = float((bb["bandwidth"].dropna() <= last["bandwidth"]).mean())
        where = ("above upper band", BEARISH) if p > last["upper"] else (
            ("below lower band", BULLISH) if p < last["lower"] else ("inside the bands", NEUTRAL))
        squeeze = " · bands unusually narrow (squeeze)" if width_rank <= 0.1 else ""
        rows.append(("Bollinger", f"%B {last['percent_b']:.2f}, bandwidth {last['bandwidth']:.2%}",
                     where[0] + squeeze, where[1]))
    return pd.DataFrame(rows, columns=["indicator", "reading", "interpretation", "signal"]).set_index("indicator")


# --------------------------------------------------------------------------- #
# Candlestick patterns
# --------------------------------------------------------------------------- #
def validate_ohlc(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Require Open/High/Low/Close, positive, finite, High >= max(O, C) and Low <= min(O, C)."""
    missing = [c for c in OHLC if c not in ohlc.columns]
    if missing:
        raise DataError(f"OHLC data is missing column(s): {', '.join(missing)}.")
    data = ohlc[OHLC].astype(float)
    if data.isna().to_numpy().any() or not np.isfinite(data.to_numpy()).all():
        raise DataError("OHLC data contains missing or infinite values; they are not filled.")
    if (data.to_numpy() <= 0).any():
        raise DataError("OHLC prices must be positive.")
    bad = (data["High"] < data[["Open", "Close"]].max(axis=1) - 1e-9) | \
          (data["Low"] > data[["Open", "Close"]].min(axis=1) + 1e-9)
    if bad.any():
        raise DataError(f"High/Low inconsistent with Open/Close on {bad.sum()} day(s), "
                        f"first {data.index[bad.to_numpy()][0]}.")
    if not data.index.is_monotonic_increasing or data.index.has_duplicates:
        raise DataError("OHLC dates must be unique and increasing.")
    return data


def candlestick_patterns(ohlc: pd.DataFrame, *, trend_window: int = 5, doji_ratio: float = 0.1,
                         shadow_ratio: float = 2.0) -> pd.DataFrame:
    """Detect the lecture's patterns. Columns: date, pattern, signal, needs_confirmation.

    Rules (body = |C - O|, range = H - L, prior trend = sign of C_{t-1} - C_{t-1-trend_window}):
    * Doji: body <= doji_ratio x range.
    * Hammer (downtrend) / hanging man (uptrend): lower shadow >= shadow_ratio x body,
      upper shadow <= body, body > 0.
    * Bullish engulfing (downtrend): red then green body that covers the previous body.
      Bearish engulfing (uptrend): green then red covering body.
    * Bullish harami (downtrend): long red, then a smaller body inside it. Bearish: the mirror.
    * Morning star (downtrend): long red; small body gapping below it; green closing above the
      first body's midpoint. Evening star (uptrend): the mirror.
    A "long" body is at least the average body of the previous `trend_window` days; a "small"
    body is at most 30 % of the first candle's body.
    """
    d = validate_ohlc(ohlc)
    if len(d) < trend_window + 2:
        raise InsufficientDataError(f"Need at least {trend_window + 2} candles, got {len(d)}.")
    o, h, l, c = (d[x].to_numpy() for x in OHLC)
    body = np.abs(c - o)
    rng = h - l
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    green, red = c > o, c < o
    avg_body = pd.Series(body).shift(1).rolling(trend_window, min_periods=trend_window).mean().to_numpy()
    out = []

    def trend_before(i: int) -> int:
        j = i - 1 - trend_window
        if j < 0:
            return 0
        return int(np.sign(c[i - 1] - c[j]))

    for i in range(len(d)):
        tr = trend_before(i)
        date = d.index[i]
        if rng[i] > 0 and body[i] <= doji_ratio * rng[i]:
            out.append((date, "Doji", BEARISH if tr > 0 else BULLISH if tr < 0 else NEUTRAL, True))
        elif body[i] > 0 and lower[i] >= shadow_ratio * body[i] and upper[i] <= body[i]:
            if tr < 0:
                out.append((date, "Hammer", BULLISH, True))
            elif tr > 0:
                out.append((date, "Hanging man", BEARISH, True))
        if i >= 1:
            p = i - 1
            if tr < 0 and red[p] and green[i] and o[i] <= c[p] and c[i] >= o[p] and body[i] > body[p]:
                out.append((date, "Bullish engulfing", BULLISH, False))
            if tr > 0 and green[p] and red[i] and o[i] >= c[p] and c[i] <= o[p] and body[i] > body[p]:
                out.append((date, "Bearish engulfing", BEARISH, False))
            inside = max(o[i], c[i]) <= max(o[p], c[p]) and min(o[i], c[i]) >= min(o[p], c[p]) and body[i] < body[p]
            long_prev = np.isfinite(avg_body[p]) and body[p] >= avg_body[p]
            if inside and long_prev and tr < 0 and red[p]:
                out.append((date, "Bullish harami", BULLISH, False))
            if inside and long_prev and tr > 0 and green[p]:
                out.append((date, "Bearish harami", BEARISH, False))
        if i >= 2:
            a, b = i - 2, i - 1
            tr_a = trend_before(a)
            long_a = np.isfinite(avg_body[a]) and body[a] >= avg_body[a] and body[a] > 0
            small_b = body[b] <= 0.3 * body[a]
            mid_a = (o[a] + c[a]) / 2
            if tr_a < 0 and long_a and red[a] and small_b and max(o[b], c[b]) < c[a] and green[i] and c[i] > mid_a:
                out.append((date, "Morning star", BULLISH, False))
            if tr_a > 0 and long_a and green[a] and small_b and min(o[b], c[b]) > c[a] and red[i] and c[i] < mid_a:
                out.append((date, "Evening star", BEARISH, False))
    return pd.DataFrame(out, columns=["date", "pattern", "signal", "needs_confirmation"])


# --------------------------------------------------------------------------- #
# Breadth and volume
# --------------------------------------------------------------------------- #
def trin(advancing_issues: float, declining_issues: float, advancing_volume: float,
         declining_volume: float) -> float:
    """Arms index = (adv issues / dec issues) / (adv volume / dec volume). Above 1 bearish, below 1 bullish."""
    values = [advancing_issues, declining_issues, advancing_volume, declining_volume]
    if any(not np.isfinite(v) or v <= 0 for v in values):
        raise ValueError("TRIN needs positive counts and volumes for advancing and declining issues.")
    return (advancing_issues / declining_issues) / (advancing_volume / declining_volume)


def volume_confirmation(prices: pd.Series, volume: pd.Series, window: int = 20) -> pd.DataFrame:
    """The lecture's table: price direction x volume vs its average.

    up & high volume -> bullish, strong;  up & low -> bullish, weak;
    down & high volume -> bearish, strong; down & low -> bearish, weak.
    (The slide prints 'weak/strong' for the bearish rows from a buyer's point of view;
    here strength always refers to the move itself.)
    """
    if not prices.index.equals(volume.index):
        raise DataError("Prices and volume must cover the same dates.")
    if (volume < 0).any() or volume.isna().any():
        raise DataError("Volume must be non-negative with no missing values.")
    avg = volume.shift(1).rolling(window, min_periods=window).mean()
    move = prices.diff()
    high = volume > avg
    direction = np.where(move > 0, "up", np.where(move < 0, "down", "flat"))
    strength = np.where(avg.isna() | (move == 0) | move.isna(), None, np.where(high, "strong", "weak"))
    return pd.DataFrame({"price_change": move, "volume": volume, "average_volume": avg,
                         "direction": direction, "strength": strength}, index=prices.index)
