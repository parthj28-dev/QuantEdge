"""Shared test fixtures.

All data here is SYNTHETIC and generated with a fixed random seed. It exists only
to test the code deterministically and offline; it is never presented as market data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def business_dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2023-01-02", periods=300)


@pytest.fixture
def synthetic_prices(business_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Three random-walk price series (seed 42), strictly positive, no gaps."""
    rng = np.random.default_rng(42)
    daily = rng.normal(loc=0.0005, scale=0.01, size=(len(business_dates), 3))
    prices = 100.0 * np.cumprod(1.0 + daily, axis=0)
    return pd.DataFrame(prices, index=business_dates, columns=["AAA", "BBB", "CCC"])


@pytest.fixture
def make_yf_frame():
    """Build a DataFrame shaped like `yf.download(..., auto_adjust=False)` output.

    'Close' is deliberately different from 'Adj Close' so tests can prove the
    adjusted column is the one being used.
    """

    def _make(adj_close: pd.DataFrame, field_first: bool = True) -> pd.DataFrame:
        fields = {
            "Adj Close": adj_close,
            "Close": adj_close * 1.05,
            "Volume": adj_close * 0 + 1_000,
        }
        pieces = []
        for field, frame in fields.items():
            cols = (
                pd.MultiIndex.from_product([[field], frame.columns], names=["Price", "Ticker"])
                if field_first
                else pd.MultiIndex.from_product([frame.columns, [field]], names=["Ticker", "Price"])
            )
            pieces.append(pd.DataFrame(frame.to_numpy(), index=frame.index, columns=cols))
        return pd.concat(pieces, axis=1)

    return _make


@pytest.fixture
def fake_downloader():
    """Return a downloader that serves a fixed frame and records how it was called."""

    def _factory(frame: pd.DataFrame | None = None, error: Exception | None = None):
        calls: list[tuple] = []

        def _download(tickers, start, end):
            calls.append((list(tickers), start, end))
            if error is not None:
                raise error
            return frame

        _download.calls = calls
        return _download

    return _factory
