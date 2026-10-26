"""QuantEdge — Streamlit dashboard.

This file is ONLY orchestration and display. Every number shown is computed by the
tested `quantedge` package; nothing here does financial math. Expected problems
(bad tickers, too little data, infeasible caps, solver failures) arrive as
QuantEdgeError and are shown per section, so one failing section never hides the rest.

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import datetime as dt
from contextlib import contextmanager

import numpy as np
import pandas as pd
import streamlit as st

from quantedge import charts
from quantedge.backtest import (
    BacktestConfig,
    backtest_summary,
    equal_weight,
    max_sharpe_strategy,
    min_variance_strategy,
    run_backtest,
)
from quantedge.capm import PREMIUM_HISTORICAL, capm_analysis
from quantedge.config import TRADING_DAYS_PER_YEAR
from quantedge.data import PriceData, fetch_prices, load_prices_csv, parse_tickers
from quantedge.exceptions import QuantEdgeError
from quantedge.frontier import efficient_frontier
from quantedge.metrics import (
    annual_arithmetic_risk_free,
    annualized_covariance,
    correlation_matrix,
    performance_summary,
    simple_returns,
    zero_volatility_assets,
)
from quantedge.optimize import (
    OptimizationResult,
    historical_inputs,
    max_sharpe,
    min_variance,
    portfolio_stats,
)

st.set_page_config(page_title="QuantEdge", page_icon="📈", layout="wide")

DEFAULT_TICKERS = "RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ITC.NS"
DEFAULT_BENCHMARK = "^NSEI"
REBALANCE_CHOICES = {
    "Monthly": "M",
    "Quarterly": "Q",
    "Weekly": "W",
    "Every N trading days": "N",
    "Buy and hold (never rebalance)": None,
}
PCT = "{:.2%}"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@contextmanager
def guarded(section: str):
    """Show expected errors inside the section instead of crashing the page."""
    try:
        yield
    except QuantEdgeError as exc:
        st.error(f"**{section}** — {exc}")


def chart_mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:  # theme unknown (e.g. in tests)
        return "light"


def styled(table: pd.DataFrame, pct_cols=(), num_cols=(), int_cols=()):
    """Format for display only; NaN is shown as 'undefined', never as 0."""
    fmt = {c: PCT for c in pct_cols if c in table}
    fmt.update({c: "{:.2f}" for c in num_cols if c in table})
    fmt.update({c: "{:,.0f}" for c in int_cols if c in table})
    return table.style.format(fmt, na_rep="undefined")


@st.cache_data(ttl=3600, show_spinner="Downloading prices from Yahoo Finance…")
def cached_fetch(tickers: tuple[str, ...], start: str, end: str) -> PriceData:
    return fetch_prices(list(tickers), start, end)


@st.cache_data(show_spinner="Tracing the efficient frontier…")
def cached_frontier(mu: pd.Series, cov: pd.DataFrame, n_points: int, cap: float, rf: float):
    return efficient_frontier(mu, cov, n_points=n_points, max_weight=cap, rf_annual=rf)


@st.cache_data(show_spinner="Running walk-forward backtests…")
def cached_backtests(returns: pd.DataFrame, lookback: int, rebalance, cost_bps: float,
                     rf: float, cap: float):
    config = BacktestConfig(lookback=lookback, rebalance=rebalance, cost_bps=cost_bps)
    strategies = {
        "Equal weight": equal_weight(),
        "Min variance": min_variance_strategy(cap),
        "Max Sharpe": max_sharpe_strategy(rf, cap),
    }
    return [run_backtest(returns, s, config, name=name) for name, s in strategies.items()]


# --------------------------------------------------------------------------- #
# Sidebar: data and assumptions
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("1 · Data")
    source = st.radio("Source", ["Yahoo Finance", "Upload CSV"], horizontal=True)

    if source == "Yahoo Finance":
        tickers_text = st.text_area("Tickers (Yahoo symbols)", DEFAULT_TICKERS,
                                    help="Comma or space separated, e.g. AAPL, MSFT or RELIANCE.NS.")
        benchmark_text = st.text_input("Benchmark (optional)", DEFAULT_BENCHMARK,
                                       help="Index used for CAPM beta and comparisons, e.g. ^NSEI or ^GSPC.")
        today = dt.date.today()
        start = st.date_input("Start date", today.replace(year=today.year - 5), max_value=today)
        end = st.date_input("End date (exclusive)", today, max_value=today)
        if st.button("Load data", type="primary", width="stretch"):
            try:
                assets = parse_tickers(tickers_text)
                bench = parse_tickers(benchmark_text)[0] if benchmark_text.strip() else None
                request = assets + ([bench] if bench and bench not in assets else [])
                data = cached_fetch(tuple(request), str(start), str(end))
                st.session_state["dataset"] = {"data": data, "benchmark": bench, "source": "Yahoo Finance"}
            except QuantEdgeError as exc:
                st.error(str(exc))
    else:
        upload = st.file_uploader("Price CSV", type="csv",
                                  help="First column: dates (YYYY-MM-DD). Then one adjusted-price column per ticker.")
        if upload is not None:
            columns = list(pd.read_csv(upload, nrows=0).columns[1:])
            upload.seek(0)
            bench_choice = st.selectbox("Benchmark column (optional)", ["(none)", *columns])
            if st.button("Load data", type="primary", width="stretch"):
                try:
                    data = load_prices_csv(upload)
                    bench = None if bench_choice == "(none)" else bench_choice.strip().upper()
                    st.session_state["dataset"] = {"data": data, "benchmark": bench,
                                                   "source": f"CSV: {upload.name}"}
                except QuantEdgeError as exc:
                    st.error(str(exc))

    st.header("2 · Assumptions")
    rf = st.number_input(
        "Risk-free rate (% per year)", min_value=-5.0, max_value=50.0, value=6.5, step=0.25,
        help="Placeholder assumption — set it to the current yield on a short-term government bill.",
    ) / 100
    cap = st.slider("Maximum weight per asset (%)", 5, 100, 40, step=5) / 100
    mu_source = st.radio("Expected returns for optimisation", ["Historical mean", "CAPM"],
                         help="CAPM needs a benchmark.")
    premium_mode = st.radio("Market risk premium (CAPM)", ["Historical (benchmark)", "Custom"])
    custom_premium = None
    if premium_mode == "Custom":
        custom_premium = st.number_input("Market risk premium (% per year)", -10.0, 30.0, 6.0, 0.5) / 100


# --------------------------------------------------------------------------- #
# Main page
# --------------------------------------------------------------------------- #
st.title("QuantEdge — Portfolio Risk & Optimization Lab")
st.caption(
    "Research and educational tool. Results describe historical data under stated assumptions; "
    "they are not forecasts and not investment advice."
)

dataset = st.session_state.get("dataset")
if dataset is None:
    st.info("Choose tickers and dates in the sidebar, then press **Load data**.")
    st.stop()

data: PriceData = dataset["data"]
report = data.report
benchmark = dataset["benchmark"]
if benchmark and benchmark not in data.prices.columns:
    st.warning(f"Benchmark {benchmark} has no usable data, so CAPM and benchmark comparisons are off.")
    benchmark = None

all_returns = simple_returns(data.prices)
asset_cols = [c for c in data.prices.columns if c != benchmark]
if not asset_cols:
    st.error("No assets left besides the benchmark. Add at least one ticker.")
    st.stop()
asset_returns = all_returns[asset_cols]
bench_returns = all_returns[benchmark] if benchmark else None
mode = chart_mode()

with st.expander(f"Data: {dataset['source']} · {len(asset_cols)} assets · "
                 f"{report.start.date()} → {report.end.date()} · {report.n_returns} daily returns",
                 expanded=bool(report.missing_tickers or report.dropped_dates)):
    st.text(report.summary())
    st.download_button("Download validated prices (CSV)", data.prices.to_csv(index_label="Date"),
                       file_name="quantedge_prices.csv", mime="text/csv",
                       help="Reload this file later via 'Upload CSV' to reproduce the exact analysis.")

zero_vol = zero_volatility_assets(asset_returns)
if zero_vol:
    st.warning(f"Zero-volatility series: {', '.join(zero_vol)}. Their Sharpe ratio and correlations are "
               "undefined, and maximum-Sharpe optimisation will refuse them.")

tab_overview, tab_risk, tab_capm, tab_opt, tab_bt, tab_method = st.tabs(
    ["Overview", "Risk", "CAPM", "Optimisation", "Backtest", "Methodology"]
)

# --- Overview ----------------------------------------------------------------- #
with tab_overview:
    with guarded("Performance statistics"):
        stats = performance_summary(all_returns, rf_annual=rf)
        st.subheader("Historical statistics")
        st.dataframe(styled(stats, pct_cols=["annual_return", "cagr", "volatility", "max_drawdown"],
                            num_cols=["sharpe"], int_cols=["observations"]))
        st.caption(f"Annualised with {TRADING_DAYS_PER_YEAR} trading days. 'annual_return' is the "
                   "arithmetic mean × 252 (optimiser input); CAGR is the compounded growth actually "
                   f"experienced. Sharpe uses a {rf:.2%} risk-free rate.")
        st.plotly_chart(charts.risk_return_scatter(stats, mode=mode), width="stretch")

# --- Risk --------------------------------------------------------------------- #
with tab_risk:
    with guarded("Correlation"):
        corr = correlation_matrix(all_returns)
        st.plotly_chart(charts.correlation_heatmap(corr, mode=mode), width="stretch")
        with st.expander("Correlation table"):
            st.dataframe(corr.style.format("{:.3f}", na_rep="undefined"))

# --- CAPM --------------------------------------------------------------------- #
capm_result = None
with tab_capm:
    if bench_returns is None:
        st.info("Add a benchmark (e.g. ^NSEI or ^GSPC) to estimate betas.")
    else:
        with guarded("CAPM"):
            capm_result = capm_analysis(asset_returns, bench_returns, rf_annual=rf,
                                        market_premium=custom_premium)
            premium_label = ("historical benchmark premium" if capm_result.premium_source == PREMIUM_HISTORICAL
                             else "custom premium")
            st.subheader(f"CAPM vs {benchmark}")
            st.caption(f"{capm_result.n_observations} aligned daily returns · {premium_label} "
                       f"{capm_result.market_premium:.2%} · rf (arithmetic annual) "
                       f"{annual_arithmetic_risk_free(rf):.2%}")
            st.dataframe(styled(
                capm_result.table[["beta", "beta_std_error", "r_squared", "alpha_annual",
                                   "residual_volatility", "historical_return", "capm_expected_return"]],
                pct_cols=["alpha_annual", "residual_volatility", "historical_return", "capm_expected_return"],
                num_cols=["beta", "beta_std_error", "r_squared"],
            ))
            for note in capm_result.notes:
                st.warning(note)
            st.caption("historical_return = capm_expected_return + alpha_annual when the historical "
                       "premium is used. Alpha is the part of past performance CAPM does not explain.")

# --- Optimisation --------------------------------------------------------------- #
with tab_opt:
    mu = None
    if mu_source == "CAPM":
        if capm_result is None:
            st.error("CAPM expected returns need a benchmark with valid data (see the CAPM tab).")
        else:
            mu = capm_result.table["capm_expected_return"]
    else:
        mu = historical_inputs(asset_returns)[0]

    if mu is not None and len(asset_cols) < 2:
        st.info("Optimisation needs at least two assets.")
    elif mu is not None:
        cov = annualized_covariance(asset_returns)
        st.caption(f"Long-only, weights sum to 100 %, each ≤ {cap:.0%}. Expected returns: {mu_source}.")
        portfolios: dict[str, OptimizationResult] = {}
        with guarded("Minimum variance"):
            portfolios["Min variance"] = min_variance(cov, mu=mu, max_weight=cap, rf_annual=rf)
        with guarded("Maximum Sharpe"):
            portfolios["Max Sharpe"] = max_sharpe(mu, cov, rf_annual=rf, max_weight=cap)
        ew = pd.Series(1 / len(asset_cols), index=asset_cols, name="weight")
        ew_stats = portfolio_stats(ew, mu, cov, rf)
        portfolios["Equal weight"] = OptimizationResult(ew, ew_stats.expected_return, ew_stats.volatility,
                                                         ew_stats.sharpe, "equal_weight", 0)

        summary = pd.DataFrame({name: {"expected_return": p.expected_return, "volatility": p.volatility,
                                       "sharpe": p.sharpe} for name, p in portfolios.items()}).T
        left, right = st.columns([3, 2])
        with left:
            st.subheader("Portfolios (ex-ante, from the estimates)")
            st.dataframe(styled(summary, pct_cols=["expected_return", "volatility"], num_cols=["sharpe"]))
            weights = pd.DataFrame({name: p.weights for name, p in portfolios.items()})
            st.plotly_chart(charts.allocation_bar(weights, mode=mode), width="stretch")
            with st.expander("Weights table"):
                st.dataframe(styled(weights, pct_cols=list(weights.columns)))
        with right:
            pick = st.selectbox("Allocation of", list(portfolios))
            st.plotly_chart(charts.allocation_pie(portfolios[pick].weights, mode=mode, title=pick),
                            width="stretch")
        for p in portfolios.values():
            for note in p.notes:
                st.warning(note)

        st.subheader("Efficient frontier")
        n_points = st.slider("Frontier points", 5, 60, 25)
        with guarded("Efficient frontier"):
            frontier = cached_frontier(mu, cov, n_points, cap, rf)
            asset_points = pd.DataFrame({"volatility": np.sqrt(np.diag(cov)), "expected_return": mu},
                                        index=cov.index)
            st.plotly_chart(charts.frontier_chart(frontier, assets=asset_points, portfolios=portfolios,
                                                  rf_arith=annual_arithmetic_risk_free(rf), mode=mode),
                            width="stretch")
            for note in frontier.notes:
                st.warning(note)
            with st.expander("Frontier table"):
                st.dataframe(styled(pd.concat([frontier.points, frontier.weights], axis=1),
                                    pct_cols=["target_return", "expected_return", "volatility",
                                              *frontier.weights.columns], num_cols=["sharpe"]))
        st.info(
            "These portfolios are optimal only for the *estimated* expected returns and covariance. "
            "Small changes in the estimates (e.g. a different date range) can move the weights a lot — "
            "the Backtest tab shows how the approach performs when re-estimated without hindsight."
        )

# --- Backtest ----------------------------------------------------------------- #
with tab_bt:
    st.caption("Walk-forward: at each rebalance the strategy sees only the trailing lookback window, "
               "trades at that day's close and earns returns from the next day. Costs are charged on "
               "turnover. Max Sharpe here uses historical-mean expected returns from each window.")
    with st.form("backtest_settings"):
        c1, c2, c3, c4 = st.columns(4)
        lookback = c1.number_input("Lookback (trading days)", 20, 2000, 252, step=21)
        rebalance_label = c2.selectbox("Rebalance", list(REBALANCE_CHOICES))
        every_n = c3.number_input("N (if 'Every N')", 1, 500, 21)
        cost_bps = c4.number_input("Cost (bps per unit traded)", 0.0, 500.0, 10.0, step=5.0)
        submitted = st.form_submit_button("Run backtest", type="primary")

    rebalance = REBALANCE_CHOICES[rebalance_label]
    rebalance = int(every_n) if rebalance == "N" else rebalance
    key = (dataset["source"], tuple(asset_cols), str(report.start), str(report.end),
           int(lookback), rebalance, float(cost_bps), rf, cap)
    if submitted:
        with guarded("Backtest"):
            st.session_state["backtest"] = (key, cached_backtests(asset_returns, int(lookback), rebalance,
                                                                  float(cost_bps), rf, cap))
    stored = st.session_state.get("backtest")
    if stored and stored[0] != key:
        st.info("Settings or data changed — press **Run backtest** to update.")
    elif stored:
        results = stored[1]
        bench_named = bench_returns.rename(f"{benchmark} (buy & hold)") if bench_returns is not None else None
        with guarded("Backtest summary"):
            table = backtest_summary(results, rf_annual=rf, benchmark=bench_named)
            st.dataframe(styled(table, pct_cols=["annual_return", "cagr", "volatility", "max_drawdown",
                                                 "total_costs"],
                                num_cols=["sharpe", "annual_turnover"],
                                int_cols=["observations", "rebalances", "skipped_rebalances"]))
            st.caption(f"Performance period {results[0].returns.index[0].date()} → "
                       f"{results[0].returns.index[-1].date()}. annual_turnover = average fraction of the "
                       "portfolio traded per year (excluding the first purchase).")
        log_scale = st.toggle("Log scale", value=False)
        st.plotly_chart(charts.equity_curves(results, benchmark=bench_named, mode=mode, log_scale=log_scale),
                        width="stretch")
        st.plotly_chart(charts.drawdown_chart(results, benchmark=bench_named, mode=mode), width="stretch")
        show = st.selectbox("Weights over time for", [r.name for r in results])
        chosen = next(r for r in results if r.name == show)
        st.plotly_chart(charts.weights_over_time(chosen, mode=mode), width="stretch")
        for r in results:
            for date, reason in r.skipped_rebalances:
                st.warning(f"{r.name}: rebalance on {date.date()} skipped (kept drifted weights) — {reason}")

# --- Methodology -------------------------------------------------------------- #
with tab_method:
    st.markdown(f"""
**Conventions** (full derivations in `docs/ARCHITECTURE.md`)

- Prices: adjusted close. Dates where any series is missing are **dropped and reported, never filled**.
- Returns: simple daily returns. Annualisation: mean × {TRADING_DAYS_PER_YEAR}, volatility × √{TRADING_DAYS_PER_YEAR}.
- Risk-free: annual effective rate → daily (1+r)^(1/{TRADING_DAYS_PER_YEAR}) − 1; compared with
  arithmetic returns as {TRADING_DAYS_PER_YEAR} × daily rate.
- CAPM: OLS of asset excess returns on benchmark excess returns over identical dates.
- Optimisation: SciPy SLSQP, long-only, fully invested, per-asset cap; every solution re-verified.
- Backtest: walk-forward, trailing window only, trade at close, drift between rebalances,
  cost = bps × turnover.

**Limitations.** Expected returns estimated from history are very noisy and optimised weights are
highly sensitive to them. Sample covariance is unstable with many assets and short histories.
yfinance is an unofficial data source. Backtests ignore taxes, slippage beyond the fixed cost,
and liquidity. Past performance does not predict future results.
""")
