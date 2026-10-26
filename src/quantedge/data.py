"""Data acquisition and validation.

Responsibilities
----------------
1. Turn user input ("aapl, msft") into a clean list of ticker symbols.
2. Download adjusted daily prices (yfinance by default, or any injected function).
3. Load / save prices as CSV so that an analysis can be reproduced exactly later.
4. Validate prices *strictly* and produce a report of everything that was
   excluded or dropped.

Missing-data policy (the most important design decision in this module)
-----------------------------------------------------------------------
We never forward-fill, back-fill, interpolate or zero-fill prices.

* Forward-filling a missing price creates an artificial 0 % return on that day.
  That understates volatility, distorts correlations, and can flatter the
  Sharpe ratio — a silent bias.
* Instead we keep only dates on which *every* selected ticker has a real price
  ("listwise deletion"). The return computed across a dropped date then
  represents the true price change over the gap — nothing is invented.
* Every exclusion is recorded in a `ValidationReport` so the user can see it.

Invalid values (prices <= 0, +/-infinity, duplicate dates, non-numeric data)
are not "missing" — they mean the data source is broken — so they raise
`DataError` instead of being dropped.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd

from .config import ADJUSTED_PRICE_FIELD, DEFAULT_MIN_OBSERVATIONS
from .exceptions import DataError, InsufficientDataError

# A downloader takes (tickers, start, end) and returns a raw DataFrame shaped like
# yfinance's output. Injecting it keeps the network out of tests.
Downloader = Callable[[list[str], str, str], pd.DataFrame]

# Yahoo symbols: letters/digits plus . - ^ = &  e.g. RELIANCE.NS, BRK-B, ^NSEI, M&M.NS, EURUSD=X
_TICKER_PATTERN = re.compile(r"^[A-Z0-9^][A-Z0-9.\-^=&]{0,19}$")


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ValidationReport:
    """Everything the validator removed or noticed. Shown to the user in the UI."""

    requested: tuple[str, ...]
    used: tuple[str, ...]
    missing_tickers: tuple[str, ...]
    dropped_dates: int
    start: pd.Timestamp
    end: pd.Timestamp
    n_prices: int
    notes: tuple[str, ...]

    @property
    def n_returns(self) -> int:
        """Number of daily returns the validated prices will produce."""
        return max(self.n_prices - 1, 0)

    def summary(self) -> str:
        lines = [
            f"Tickers used: {', '.join(self.used)}",
            f"Period: {self.start.date()} to {self.end.date()} "
            f"({self.n_prices} prices, {self.n_returns} daily returns)",
        ]
        if self.missing_tickers:
            lines.append(f"Excluded (no data): {', '.join(self.missing_tickers)}")
        if self.dropped_dates:
            lines.append(f"Dates dropped because at least one ticker had no price: {self.dropped_dates}")
        lines.extend(self.notes)
        return "\n".join(lines)


@dataclass(frozen=True)
class PriceData:
    """Validated prices plus the report describing how they were obtained."""

    prices: pd.DataFrame
    report: ValidationReport


# --------------------------------------------------------------------------- #
# Ticker parsing
# --------------------------------------------------------------------------- #
def parse_tickers(raw: str | Iterable[str]) -> list[str]:
    """Normalise ticker input: split, strip, upper-case, de-duplicate (order kept).

    >>> parse_tickers("aapl, msft  AAPL")
    ['AAPL', 'MSFT']
    """
    if isinstance(raw, str):
        parts = re.split(r"[,\s;]+", raw)
    else:
        parts = [str(p) for p in raw]

    tickers: list[str] = []
    for part in parts:
        symbol = part.strip().upper()
        if symbol and symbol not in tickers:
            tickers.append(symbol)

    if not tickers:
        raise DataError("No tickers were provided.")

    invalid = [t for t in tickers if not _TICKER_PATTERN.match(t)]
    if invalid:
        raise DataError(
            f"Invalid ticker symbol(s): {', '.join(invalid)}. "
            "Use Yahoo Finance symbols, e.g. AAPL, RELIANCE.NS, ^NSEI."
        )
    return tickers


# --------------------------------------------------------------------------- #
# Downloading
# --------------------------------------------------------------------------- #
def yfinance_downloader(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Default downloader. Imported lazily so tests and CSV users never need the network.

    `auto_adjust=False` keeps the explicit 'Adj Close' column (adjusted for splits
    and dividends), so there is no ambiguity about which 'Close' we are using.
    """
    import yfinance as yf

    return yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="column",
        multi_level_index=True,
    )


def extract_adjusted_close(
    raw: pd.DataFrame, tickers: Sequence[str], field: str = ADJUSTED_PRICE_FIELD
) -> pd.DataFrame:
    """Pull the adjusted-close table (dates x tickers) out of a raw download.

    Handles both column layouts yfinance can produce — (field, ticker) and
    (ticker, field) — and single-level columns for a single ticker.
    """
    if not isinstance(raw, pd.DataFrame):
        raise DataError("The data source returned no table.")

    columns = raw.columns
    if isinstance(columns, pd.MultiIndex):
        if field in columns.get_level_values(0):
            out = raw.xs(field, axis=1, level=0)
        elif field in columns.get_level_values(1):
            out = raw.xs(field, axis=1, level=1)
        else:
            fields = sorted({str(c) for c in columns.get_level_values(0)})
            raise DataError(f"'{field}' column not found in downloaded data (got: {fields}).")
    else:
        if field not in columns:
            raise DataError(f"'{field}' column not found in downloaded data (got: {list(columns)}).")
        if len(tickers) != 1:
            raise DataError("Cannot tell which ticker each column belongs to (single-level columns).")
        out = raw[[field]].rename(columns={field: tickers[0]})

    out = out.copy()
    out.columns = [str(c).upper() for c in out.columns]
    return out


def fetch_prices(
    tickers: str | Iterable[str],
    start: str | date | pd.Timestamp,
    end: str | date | pd.Timestamp,
    *,
    downloader: Downloader | None = None,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
) -> PriceData:
    """Download adjusted daily prices and validate them.

    Note: with yfinance, `end` is *exclusive* (prices up to the day before).
    """
    symbols = parse_tickers(tickers)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if start_ts >= end_ts:
        raise DataError(f"Start date {start_ts.date()} must be before end date {end_ts.date()}.")

    download = downloader or yfinance_downloader
    try:
        raw = download(symbols, start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"))
    except Exception as exc:  # network errors, rate limits, library errors...
        raise DataError(f"Price download failed: {exc}") from exc

    if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
        raise DataError(
            f"No data returned for {', '.join(symbols)} between {start_ts.date()} and "
            f"{end_ts.date()}. Check the symbols (e.g. 'RELIANCE.NS' for NSE) and the dates."
        )

    adjusted = extract_adjusted_close(raw, symbols)
    return validate_prices(adjusted, requested=symbols, min_observations=min_observations)


# --------------------------------------------------------------------------- #
# CSV round-trip (reproducibility)
# --------------------------------------------------------------------------- #
def load_prices_csv(
    source: str | Path | IO,
    *,
    requested: str | Iterable[str] | None = None,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
) -> PriceData:
    """Load a CSV with a date column first and one price column per ticker.

    Dates must be ISO formatted (YYYY-MM-DD). We refuse to guess formats because
    "03-04-2024" is 3 April in India and 4 March in the US — a silent misread
    would scramble the time series.
    """
    try:
        frame = pd.read_csv(source, index_col=0)
    except Exception as exc:
        raise DataError(f"Could not read price CSV: {exc}") from exc
    try:
        frame.index = pd.to_datetime(frame.index, format="ISO8601")
    except (ValueError, TypeError) as exc:
        raise DataError(
            "The first CSV column must contain dates in YYYY-MM-DD format (e.g. 2024-01-31)."
        ) from exc
    return validate_prices(frame, requested=requested, min_observations=min_observations)


def save_prices_csv(prices: pd.DataFrame, path: str | Path) -> Path:
    """Save validated prices so the exact same analysis can be re-run offline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path, index_label="Date")
    return path


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_prices(
    prices: pd.DataFrame,
    *,
    requested: str | Iterable[str] | None = None,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
) -> PriceData:
    """Validate a dates x tickers price table. Never fills missing values.

    Steps: check structure -> select requested tickers -> exclude tickers with no
    data -> reject invalid values -> drop dates with any missing price ->
    check there is enough data left.
    """
    if not isinstance(prices, pd.DataFrame) or prices.shape[1] == 0:
        raise DataError("No price data supplied.")
    if min_observations < 1:
        raise ValueError("min_observations must be at least 1.")

    df = prices.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    if df.columns.duplicated().any():
        dupes = sorted(set(df.columns[df.columns.duplicated()]))
        raise DataError(f"Duplicate ticker columns: {', '.join(dupes)}.")

    # --- index: real, unique, sorted dates -------------------------------- #
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataError("Price index must contain dates.")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    if df.index.hasnans:
        raise DataError("Price index contains missing dates.")
    if df.index.duplicated().any():
        first = df.index[df.index.duplicated()][0]
        raise DataError(f"Duplicate date in price data: {first.date()}.")
    df = df.sort_index()

    # --- ticker selection --------------------------------------------------- #
    wanted = parse_tickers(requested) if requested is not None else list(df.columns)
    absent = [t for t in wanted if t not in df.columns]
    df = df[[t for t in wanted if t in df.columns]]

    non_numeric = [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])
    ]
    if non_numeric:
        raise DataError(f"Non-numeric price values for: {', '.join(non_numeric)}.")
    df = df.astype(float)

    no_data = [c for c in df.columns if df[c].isna().all()]
    df = df.drop(columns=no_data)
    missing_tickers = absent + no_data
    if df.shape[1] == 0:
        raise DataError(f"No usable price data for any requested ticker ({', '.join(wanted)}).")

    # --- invalid values: reject, don't drop -------------------------------- #
    _reject(df, np.isinf(df.to_numpy()), "infinite price")
    _reject(df, (df <= 0).to_numpy(), "non-positive price")  # NaN <= 0 is False

    # --- missing values: drop dates, report, never fill --------------------- #
    notes: list[str] = [f"{t}: no data returned — excluded." for t in missing_tickers]
    na_counts = df.isna().sum()
    for ticker, n_missing in na_counts.items():
        if n_missing:
            first_ok = df[ticker].first_valid_index().date()
            last_ok = df[ticker].last_valid_index().date()
            notes.append(
                f"{ticker}: {n_missing} missing price(s) (data available {first_ok} to {last_ok}); "
                "those dates were dropped for all tickers, not filled."
            )
    complete = df.dropna(how="any")
    dropped = len(df) - len(complete)

    n_returns = max(len(complete) - 1, 0)
    if n_returns < min_observations:
        message = (
            f"Only {n_returns} daily return(s) remain after aligning dates; "
            f"at least {min_observations} are required."
        )
        if na_counts.max() > 0:
            limiting = na_counts.idxmax()
            message += (
                f" The sample is limited mainly by {limiting} "
                f"({int(na_counts.max())} missing prices) — try removing it."
            )
        message += " Try a longer date range."
        raise InsufficientDataError(message)

    report = ValidationReport(
        requested=tuple(wanted),
        used=tuple(complete.columns),
        missing_tickers=tuple(missing_tickers),
        dropped_dates=int(dropped),
        start=complete.index[0],
        end=complete.index[-1],
        n_prices=len(complete),
        notes=tuple(notes),
    )
    return PriceData(prices=complete, report=report)


def _reject(df: pd.DataFrame, mask: np.ndarray, what: str) -> None:
    """Raise DataError naming the first (ticker, date) where `mask` is True."""
    if mask.any():
        row, col = np.argwhere(mask)[0]
        raise DataError(
            f"{df.columns[col]} has a {what} ({df.iat[row, col]}) on {df.index[row].date()}. "
            "The data source is unreliable for this ticker; remove it or use another source."
        )
