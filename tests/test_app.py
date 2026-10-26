"""Smoke tests for the Streamlit app, run headlessly with streamlit.testing (no browser, no network).

Data enters through the app's own "Upload CSV" path with SYNTHETIC prices, so the
whole pipeline (validation -> metrics -> CAPM -> optimisation -> frontier -> backtest
-> charts) is exercised exactly as a user would drive it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def synthetic_csv(n_days: int = 400, seed: int = 21) -> bytes:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    market = rng.normal(0.0004, 0.01, n_days)
    betas = {"AAA": 0.8, "BBB": 1.1, "CCC": 1.4, "DDD": 0.5}
    rets = {t: b * market + rng.normal(0.0001, 0.008, n_days) for t, b in betas.items()}
    rets["MKT"] = market
    prices = 100 * (1 + pd.DataFrame(rets, index=dates)).cumprod()
    return prices.to_csv(index_label="Date").encode()


def load_csv_app(benchmark: str = "MKT") -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value("Upload CSV").run()
    at.sidebar.file_uploader[0].upload("prices.csv", synthetic_csv(), "text/csv").run()
    at.sidebar.selectbox[0].set_value(benchmark).run()
    next(b for b in at.sidebar.button if b.label == "Load data").click().run()
    return at


def error_texts(at: AppTest) -> list[str]:
    return [e.value for e in at.error]


def test_app_starts_with_instructions():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert any("Load data" in i.value for i in at.info)


def test_full_pipeline_renders_without_errors():
    at = load_csv_app()
    assert not at.exception, at.exception
    assert error_texts(at) == []
    assert [t.label for t in at.tabs] == ["Overview", "Risk", "CAPM", "Optimisation", "Backtest", "Methodology"]
    # Overview scatter, heatmap, allocation bar, pie, frontier
    assert len(at.get("plotly_chart")) == 5
    assert len(at.dataframe) >= 4  # stats, correlation, CAPM, portfolios (+ expanders)


def test_backtest_runs_from_the_form():
    at = load_csv_app()
    lookback = next(n for n in at.number_input if n.label.startswith("Lookback"))
    lookback.set_value(60)
    next(b for b in at.button if b.label == "Run backtest").click().run()
    assert not at.exception, at.exception
    assert error_texts(at) == []
    # equity, drawdown and weights-over-time charts are added
    assert len(at.get("plotly_chart")) == 8


def test_infeasible_cap_is_reported_but_page_still_renders():
    at = load_csv_app()
    at.sidebar.slider[0].set_value(5).run()   # 4 assets x 5 % cannot reach 100 %
    assert not at.exception
    errors = error_texts(at)
    assert any("Infeasible" in e for e in errors)
    assert [t.label for t in at.tabs][0] == "Overview"  # other sections still shown


def test_capm_returns_need_a_benchmark():
    at = load_csv_app(benchmark="(none)")
    next(r for r in at.sidebar.radio if r.label.startswith("Expected returns")).set_value("CAPM").run()
    assert not at.exception
    assert any("need a benchmark" in e for e in error_texts(at))


def test_invalid_ticker_shows_error_without_network():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.sidebar.text_area[0].set_value("AAPL, BAD*TICKER").run()
    next(b for b in at.sidebar.button if b.label == "Load data").click().run()
    assert not at.exception
    assert any("Invalid ticker" in e for e in error_texts(at))
