"""Tests for quantedge.charts: each figure must faithfully contain the data it was given."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from quantedge import charts
from quantedge.backtest import BacktestConfig, equal_weight, min_variance_strategy, run_backtest
from quantedge.frontier import efficient_frontier
from quantedge.metrics import correlation_matrix, performance_summary, simple_returns
from quantedge.optimize import historical_inputs, max_sharpe, min_variance
from quantedge.metrics import annual_arithmetic_risk_free

LIGHT = charts.THEMES["light"]["series"]


@pytest.fixture
def returns(synthetic_prices) -> pd.DataFrame:
    return simple_returns(synthetic_prices)


@pytest.fixture
def backtests(returns):
    cfg = BacktestConfig(lookback=40, rebalance="M", cost_bps=10)
    return [run_backtest(returns, equal_weight(), cfg, name="EW"),
            run_backtest(returns, min_variance_strategy(0.6), cfg, name="MV")]


def assert_valid(fig: go.Figure) -> None:
    assert isinstance(fig, go.Figure)
    fig.to_json()  # must serialise (this is what Streamlit sends to the browser)


# --------------------------------------------------------------------------- #
# Allocations
# --------------------------------------------------------------------------- #
def test_allocation_bar_contains_exact_weights_in_fixed_colour_order():
    w = pd.DataFrame({"P1": [0.2, 0.8], "P2": [0.5, 0.5], "P3": [1.0, 0.0]}, index=["A", "B"])
    fig = charts.allocation_bar(w)
    assert_valid(fig)
    assert [t.name for t in fig.data] == ["P1", "P2", "P3"]
    assert [t.marker.color for t in fig.data] == LIGHT[:3]
    np.testing.assert_allclose(fig.data[1].y, [0.5, 0.5])
    assert fig.layout.yaxis.tickformat == ".0%"


def test_allocation_bar_limits():
    with pytest.raises(ValueError):
        charts.allocation_bar(pd.DataFrame())
    with pytest.raises(ValueError):
        charts.allocation_bar(pd.DataFrame(np.full((2, 9), 0.5), index=["A", "B"]))


def test_allocation_pie_folds_small_and_excess_holdings():
    w = pd.Series({"A": 0.30, "B": 0.25, "C": 0.15, "D": 0.10, "E": 0.08, "F": 0.06, "G": 0.059, "H": 0.001})
    fig = charts.allocation_pie(w, max_slices=6)
    assert_valid(fig)
    labels, values = list(fig.data[0].labels), np.array(fig.data[0].values)
    assert len(labels) == 6 and labels[-1] == "Other (3)"
    assert values.sum() == pytest.approx(1.0)
    assert values[-1] == pytest.approx(0.06 + 0.059 + 0.001)


def test_allocation_pie_drops_zero_and_dust_weights_without_other_slice():
    # Optimisers leave dust like 3e-11 on unused assets; that must not become a "0.0 %" slice.
    fig = charts.allocation_pie(pd.Series({"A": 0.6, "B": 0.4 - 3e-11, "C": 0.0, "D": 3e-11}))
    assert list(fig.data[0].labels) == ["A", "B"]
    assert sum(fig.data[0].values) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Frontier
# --------------------------------------------------------------------------- #
def test_frontier_chart_plots_frontier_assets_portfolios_and_cml(returns):
    mu, cov = historical_inputs(returns)
    rf = 0.03
    frontier = efficient_frontier(mu, cov, n_points=12, max_weight=0.7, rf_annual=rf)
    ports = {"Min variance": min_variance(cov, mu=mu, max_weight=0.7, rf_annual=rf),
             "Max Sharpe": max_sharpe(mu, cov, rf_annual=rf, max_weight=0.7)}
    assets = pd.DataFrame({"volatility": np.sqrt(np.diag(cov)), "expected_return": mu}, index=cov.index)
    rf_arith = annual_arithmetic_risk_free(rf)
    fig = charts.frontier_chart(frontier, assets=assets, portfolios=ports, rf_arith=rf_arith)
    assert_valid(fig)

    by_name = {t.name: t for t in fig.data}
    np.testing.assert_allclose(by_name["Efficient frontier"].x, frontier.points["volatility"])
    np.testing.assert_allclose(by_name["Efficient frontier"].y, frontier.points["expected_return"])
    assert list(by_name["Individual assets"].text) == list(cov.index)
    assert by_name["Max Sharpe"].x[0] == pytest.approx(ports["Max Sharpe"].volatility)
    # The CML starts at the risk-free rate and passes through the max-Sharpe portfolio.
    cml = by_name["Capital market line"]
    slope = (cml.y[1] - cml.y[0]) / (cml.x[1] - cml.x[0])
    assert cml.y[0] == pytest.approx(rf_arith)
    assert rf_arith + slope * ports["Max Sharpe"].volatility == pytest.approx(ports["Max Sharpe"].expected_return)


def test_frontier_chart_limits_highlighted_portfolios(returns):
    mu, cov = historical_inputs(returns)
    frontier = efficient_frontier(mu, cov, n_points=5)
    p = min_variance(cov, mu=mu)
    with pytest.raises(ValueError, match="at most 3"):
        charts.frontier_chart(frontier, portfolios={str(i): p for i in range(4)})


# --------------------------------------------------------------------------- #
# Correlation and risk/return
# --------------------------------------------------------------------------- #
def test_heatmap_keeps_values_and_shows_undefined_for_nan():
    idx = pd.bdate_range("2024-01-01", periods=5)
    frame = pd.DataFrame({"FLAT": 0.0, "X": [0.01, -0.02, 0.03, 0.0, 0.01],
                          "Y": [0.02, -0.01, 0.01, 0.01, 0.0]}, index=idx)
    corr = correlation_matrix(frame)
    fig = charts.correlation_heatmap(corr)
    assert_valid(fig)
    z = np.array(fig.data[0].z, dtype=float)
    np.testing.assert_array_equal(np.isnan(z), np.isnan(corr.to_numpy()))   # NaN stays NaN, not 0
    np.testing.assert_allclose(z[1:, 1:], corr.to_numpy()[1:, 1:])
    assert (fig.data[0].zmin, fig.data[0].zmax) == (-1, 1)                  # fixed symmetric scale
    assert "undefined" in fig.data[0].customdata[0][1]


def test_risk_return_scatter(returns):
    stats = performance_summary(returns)
    fig = charts.risk_return_scatter(stats)
    assert_valid(fig)
    np.testing.assert_allclose(fig.data[0].x, stats["volatility"])
    assert list(fig.data[0].text) == list(stats.index)


# --------------------------------------------------------------------------- #
# Backtests
# --------------------------------------------------------------------------- #
def test_equity_curves_start_at_one_and_benchmark_is_aligned(backtests, returns):
    bench = returns["AAA"].rename("AAA")
    fig = charts.equity_curves(backtests, benchmark=bench)
    assert_valid(fig)
    assert [t.name for t in fig.data] == ["EW", "MV", "AAA"]
    for trace in fig.data:
        assert trace.y[0] == pytest.approx(1.0)
    expected_end = float((1 + bench.loc[backtests[0].returns.index]).prod())
    assert fig.data[2].y[-1] == pytest.approx(expected_end)
    np.testing.assert_allclose(fig.data[0].y, backtests[0].wealth)


def test_equity_curves_reject_benchmark_with_gaps(backtests, returns):
    with pytest.raises(ValueError, match="missing"):
        charts.equity_curves(backtests, benchmark=returns["AAA"].iloc[:-5])


def test_drawdown_chart_values_are_non_positive(backtests, returns):
    fig = charts.drawdown_chart(backtests, benchmark=returns["BBB"])
    assert_valid(fig)
    assert len(fig.data) == 3
    for trace in fig.data:
        assert np.max(trace.y) <= 1e-12


def test_weights_over_time_stack_to_one(backtests):
    fig = charts.weights_over_time(backtests[1])
    assert_valid(fig)
    stacked = np.sum([np.asarray(t.y) for t in fig.data], axis=0)
    np.testing.assert_allclose(stacked, 1.0, atol=1e-8)


def test_weights_over_time_folds_more_than_eight_assets():
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(rng.normal(0, 0.01, (80, 10)), index=pd.bdate_range("2024-01-01", periods=80),
                        columns=[f"S{i}" for i in range(10)])
    result = run_backtest(rets, equal_weight(), BacktestConfig(lookback=20, rebalance=10))
    fig = charts.weights_over_time(result)
    assert len(fig.data) == 8 and fig.data[-1].name == "Other"
    np.testing.assert_allclose(np.sum([np.asarray(t.y) for t in fig.data], axis=0), 1.0)


# --------------------------------------------------------------------------- #
# Themes and layering
# --------------------------------------------------------------------------- #
def test_dark_mode_uses_dark_palette_and_bad_mode_is_rejected():
    w = pd.Series({"A": 0.5, "B": 0.5}, name="P")
    assert charts.allocation_bar(w, mode="dark").data[0].marker.color == charts.THEMES["dark"]["series"][0]
    with pytest.raises(ValueError, match="mode"):
        charts.allocation_bar(w, mode="neon")


def test_package_never_imports_streamlit():
    # Architecture rule: the math and chart layers must work without Streamlit.
    src = Path(__file__).resolve().parents[1] / "src" / "quantedge"
    offenders = [p.name for p in src.glob("*.py") if "import streamlit" in p.read_text()]
    assert offenders == []
    code = ("import sys, quantedge.charts, quantedge.backtest, quantedge.capm, quantedge.data; "
            "sys.exit(1 if 'streamlit' in sys.modules else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


# --------------------------------------------------------------------------- #
# v2 - v3 charts
# --------------------------------------------------------------------------- #
def test_layering_covers_all_modules():
    code = ("import sys, quantedge.charts, quantedge.estimators, quantedge.riskbased, quantedge.indicators, "
            "quantedge.valuation, quantedge.technical, quantedge.performance; "
            "sys.exit(1 if 'streamlit' in sys.modules else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_expected_returns_and_risk_contribution_charts(returns):
    from quantedge.riskbased import risk_contributions, risk_parity
    from quantedge.metrics import annualized_covariance
    table = pd.DataFrame({"Hist": [0.1, 0.2, 0.05], "BL": [0.08, 0.12, 0.07]}, index=["A", "B", "C"])
    fig = charts.expected_returns_chart(table)
    assert [t.name for t in fig.data] == ["Hist", "BL"]
    np.testing.assert_allclose(fig.data[1].y, [0.08, 0.12, 0.07])
    cov = annualized_covariance(returns)
    rc = risk_contributions(risk_parity(cov).weights, cov)
    fig = charts.risk_contribution_chart(rc)
    np.testing.assert_allclose(fig.data[1].y, rc["risk_share"])


def test_trend_chart_marks_crosses(synthetic_prices):
    from quantedge.indicators import crosses, sma
    px = synthetic_prices["AAA"]
    ev = crosses(px, 5, 20)
    fig = charts.trend_chart(px, {"SMA 5": sma(px, 5), "SMA 20": sma(px, 20)}, events=ev)
    marked = sum(len(t.x) for t in fig.data if t.mode == "markers")
    assert marked == len(ev) > 0


def test_technical_chart_has_three_panels(synthetic_prices):
    from quantedge import technical as ta
    px = synthetic_prices["BBB"]
    fig = charts.technical_chart(px, bands=ta.bollinger_bands(px), macd_frame=ta.macd(px), rsi_line=ta.rsi(px),
                                 ema_lines={"EMA 20": ta.ema(px, 20)}, events=ta.technical_signals(px))
    names = [t.name for t in fig.data]
    assert {"Upper band", "Lower band", "MACD", "Signal", "RSI", "EMA 20"} <= set(names)
    rsi_trace = next(t for t in fig.data if t.name == "RSI")
    assert rsi_trace.yaxis == "y3"


def test_candlestick_and_dcf_and_sml_and_timing():
    from quantedge.valuation import multistage_ddm
    idx = pd.bdate_range("2024-01-01", periods=3)
    ohlc = pd.DataFrame({"Open": [10, 11, 12], "High": [11, 12, 13], "Low": [9, 10, 11], "Close": [11, 12, 11]}, idx)
    pats = pd.DataFrame({"date": [idx[2]], "pattern": ["Doji"], "signal": ["bearish"], "needs_confirmation": [True]})
    fig = charts.candlestick_chart(ohlc, patterns=pats)
    assert fig.data[0].type == "candlestick" and list(fig.data[1].text) == ["Doji"]
    res = multistage_ddm(2, [0.08] * 3, 0.04, 0.12)
    fig = charts.dcf_chart(res.schedule, res.pv_terminal)
    assert sum(fig.data[0].y) == pytest.approx(res.value)
    table = pd.DataFrame({"beta": [0.8, 1.2], "return": [0.1, 0.15], "jensen_alpha": [0.01, -0.01]}, index=["X", "Y"])
    fig = charts.security_market_line(table, rf=0.05, market_return=0.12)
    assert fig.data[0].y[0] == pytest.approx(0.05)
    m = pd.Series(np.linspace(-0.02, 0.02, 50))
    fig = charts.timing_scatter(0.5 * m, m, alpha=0.0, beta=0.5, gamma=0.0, model="treynor_mazuy")
    np.testing.assert_allclose(fig.data[1].y, 0.5 * np.asarray(fig.data[1].x))
