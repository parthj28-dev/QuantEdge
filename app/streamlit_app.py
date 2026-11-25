"""QuantEdge — Streamlit dashboard.

This file is ONLY orchestration and display. Every number shown is computed by the
tested `quantedge` package; nothing here does financial math. Expected problems
(bad tickers, too little data, infeasible caps, solver failures, invalid model inputs)
arrive as QuantEdgeError and are shown per section, so one failing section never hides
the rest.

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import datetime as dt
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:  # installed package (pip install -e .) or, on a host that only installs requirements, the src folder
    import quantedge  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantedge import __version__, charts
from quantedge import performance as ppm
from quantedge import technical as ta
from quantedge import valuation as fa
from quantedge.backtest import STRATEGY_FACTORIES, BacktestConfig, backtest_summary, run_backtest
from quantedge.capm import PREMIUM_HISTORICAL, capm_analysis
from quantedge.config import TRADING_DAYS_PER_YEAR
from quantedge.data import (
    OHLCVData,
    PriceData,
    fetch_ohlcv,
    fetch_prices,
    load_ohlcv_csv,
    load_prices_csv,
    parse_tickers,
)
from quantedge.estimators import (
    COV_LEDOIT_WOLF,
    COV_SAMPLE,
    absolute_view,
    black_litterman,
    condition_number,
    covariance,
    implied_risk_aversion,
    ledoit_wolf,
    relative_view,
)
from quantedge.exceptions import QuantEdgeError
from quantedge.frontier import efficient_frontier
from quantedge.indicators import crosses, sma, trend_status
from quantedge.metrics import (
    annual_arithmetic_risk_free,
    annualized_covariance,
    annualized_mean_return,
    correlation_matrix,
    performance_summary,
    simple_returns,
    zero_volatility_assets,
)
from quantedge.optimize import OptimizationResult, max_sharpe, min_variance, portfolio_stats
from quantedge.riskbased import (
    diversification_ratio,
    inverse_volatility,
    max_diversification,
    risk_contributions,
    risk_parity,
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
COV_CHOICES = {"Sample": COV_SAMPLE, "Ledoit-Wolf shrinkage": COV_LEDOIT_WOLF}
DEFAULT_STRATEGIES = ["Equal weight", "Equal weight + SMA filter", "Min variance", "Max Sharpe", "Risk parity"]
FREQUENCIES = {"Daily": ("D", TRADING_DAYS_PER_YEAR), "Weekly": ("W", 52), "Monthly": ("M", 12)}
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


def styled(table: pd.DataFrame, pct_cols=(), num_cols=(), int_cols=(), money_cols=(), extra=None):
    """Format for display only; NaN is shown as 'undefined', never as 0. One format call, so
    every column keeps its format (a second Styler.format call would reset the others)."""
    fmt = {c: f for c, f in (extra or {}).items() if c in table}
    fmt.update({c: PCT for c in pct_cols if c in table})
    fmt.update({c: "{:.2f}" for c in num_cols if c in table})
    fmt.update({c: "{:,.0f}" for c in int_cols if c in table})
    fmt.update({c: "{:,.2f}" for c in money_cols if c in table})
    return table.style.format(fmt, na_rep="undefined")


def pct_input(label: str, value: float, *, key: str, min_value: float = -50.0, max_value: float = 100.0,
              step: float = 0.5, help: str | None = None) -> float:
    """Percent number input returning a decimal."""
    return st.number_input(label, min_value, max_value, float(value), step, key=key, help=help) / 100


@st.cache_data(ttl=3600, show_spinner="Downloading prices from Yahoo Finance…")
def cached_fetch(tickers: tuple[str, ...], start: str, end: str) -> PriceData:
    return fetch_prices(list(tickers), start, end)


@st.cache_data(ttl=3600, show_spinner="Downloading daily candles…")
def cached_ohlcv(ticker: str, start: str, end: str) -> OHLCVData:
    return fetch_ohlcv(ticker, start, end)


@st.cache_data(show_spinner="Tracing the efficient frontier…")
def cached_frontier(mu: pd.Series, cov: pd.DataFrame, n_points: int, cap: float, rf: float):
    return efficient_frontier(mu, cov, n_points=n_points, max_weight=cap, rf_annual=rf)


@st.cache_data(show_spinner="Running walk-forward backtests…")
def cached_backtests(returns: pd.DataFrame, names: tuple[str, ...], lookback: int, rebalance, cost_bps: float,
                     rf: float, cap: float, cov_method: str, sma_window: int):
    """Run each strategy; one that cannot start (e.g. SMA window longer than the lookback) is
    reported instead of stopping the others."""
    config = BacktestConfig(lookback=lookback, rebalance=rebalance, cost_bps=cost_bps, cash_rate=rf)
    results, failures = [], []
    for n in names:
        try:
            results.append(run_backtest(returns, STRATEGY_FACTORIES[n](rf, cap, cov_method, sma_window),
                                        config, name=n))
        except QuantEdgeError as exc:
            failures.append((n, str(exc)))
    return results, failures


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
                st.session_state["dataset"] = {"data": data, "benchmark": bench, "source": "Yahoo Finance",
                                               "start": str(start), "end": str(end)}
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
    cov_label = st.radio("Covariance estimate", list(COV_CHOICES), horizontal=True,
                         help="Ledoit-Wolf shrinks the noisy sample covariance towards a simple target.")
    cov_method = COV_CHOICES[cov_label]
    mu_source = st.radio("Expected returns for optimisation", ["Historical mean", "CAPM", "Black-Litterman"],
                         help="CAPM needs a benchmark. Black-Litterman is set up in the Estimation tab.")
    premium_mode = st.radio("Market risk premium (CAPM)", ["Historical (benchmark)", "Custom"])
    custom_premium = None
    if premium_mode == "Custom":
        custom_premium = st.number_input("Market risk premium (% per year)", -10.0, 30.0, 6.0, 0.5) / 100


# --------------------------------------------------------------------------- #
# Main page
# --------------------------------------------------------------------------- #
st.title("QuantEdge — Portfolio Risk & Optimization Lab")
st.caption(
    f"v{__version__} · Research and educational tool. Results describe historical data under stated "
    "assumptions; they are not forecasts and not investment advice."
)
mode = chart_mode()
dataset = st.session_state.get("dataset")

TAB_NAMES = ["Overview", "Risk", "CAPM", "Estimation", "Optimisation", "Backtest", "Trend",
             "Technical", "Valuation", "Performance", "Methodology"]
(tab_overview, tab_risk, tab_capm, tab_est, tab_opt, tab_bt, tab_trend,
 tab_ta, tab_fa, tab_ppm, tab_method) = st.tabs(TAB_NAMES)

if dataset is None:
    st.info("Choose tickers and dates in the sidebar, then press **Load data**. "
            "The **Valuation** tab and the summary-statistics mode of **Performance** work without data.")
    for tab in (tab_overview, tab_risk, tab_capm, tab_est, tab_opt, tab_bt, tab_trend, tab_ta):
        with tab:
            st.caption("Load data in the sidebar to use this tab.")
    data = report = benchmark = all_returns = asset_returns = bench_returns = None
    asset_cols: list[str] = []
else:
    data = dataset["data"]
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

capm_result = None
bl_result = None
backtest_results = None

# --- Overview ----------------------------------------------------------------- #
if dataset is not None:
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

# --- Estimation ---------------------------------------------------------------- #
    with tab_est:
        st.subheader("Covariance")
        with guarded("Covariance"):
            sample_cov = annualized_covariance(asset_returns)
            lw = ledoit_wolf(asset_returns)
            c1, c2, c3 = st.columns(3)
            c1.metric("Shrinkage intensity δ", f"{lw.shrinkage:.3f}")
            c2.metric("Condition number (sample)", f"{condition_number(sample_cov):,.1f}")
            c3.metric("Condition number (Ledoit-Wolf)", f"{condition_number(lw.covariance):,.1f}")
            st.caption("δ = 0 keeps the sample covariance; δ = 1 replaces it with equal variances and zero "
                       "correlation. A lower condition number means a more stable matrix to optimise with. "
                       f"The sidebar choice (**{cov_label}**) is used by the optimiser.")

        st.subheader("Black-Litterman expected returns")
        with guarded("Black-Litterman"):
            cov_used = covariance(asset_returns, cov_method)
            mkt_default = pd.DataFrame({"asset": asset_cols, "market_weight_%": [round(100 / len(asset_cols), 4)]
                                        * len(asset_cols)})
            c1, c2 = st.columns([2, 3])
            with c1:
                st.caption("Market weights (placeholder: equal). Use market capitalisations for a real prior.")
                mkt_table = st.data_editor(mkt_default, key="bl_market", hide_index=True, disabled=["asset"],
                                           width="stretch")
                ra_mode = st.radio("Risk aversion δ", ["Implied by benchmark", "Manual"], horizontal=True,
                                   key="bl_ra_mode")
                risk_aversion = None
                if ra_mode == "Implied by benchmark":
                    if bench_returns is None:
                        st.warning("No benchmark loaded: enter δ manually.")
                    else:
                        try:
                            risk_aversion = implied_risk_aversion(bench_returns, rf)
                            st.caption(f"Implied δ = (E[R_m] − rf) / σ²_m = **{risk_aversion:.2f}**")
                        except QuantEdgeError as exc:
                            st.warning(f"{exc} Using the manual value below.")
                if risk_aversion is None:
                    risk_aversion = st.number_input("δ", 0.1, 20.0, 2.5, 0.1, key="bl_ra")
                tau = st.number_input("τ (uncertainty of the prior)", 0.001, 1.0, 0.05, 0.01, key="bl_tau")
            with c2:
                st.caption("Views (total annual returns). Relative: 'asset' beats 'versus' by the value. "
                           "Confidence 100 % = certain, 50 % = as uncertain as the prior.")
                views_default = pd.DataFrame({
                    "type": pd.Series([], dtype="str"), "asset": pd.Series([], dtype="str"),
                    "versus": pd.Series([], dtype="str"), "value_%": pd.Series([], dtype="float"),
                    "confidence_%": pd.Series([], dtype="float")})
                views_table = st.data_editor(
                    views_default, key="bl_views", num_rows="dynamic", hide_index=True, width="stretch",
                    column_config={
                        "type": st.column_config.SelectboxColumn(options=["absolute", "relative"], required=True),
                        "asset": st.column_config.SelectboxColumn(options=asset_cols, required=True),
                        "versus": st.column_config.SelectboxColumn(options=asset_cols),
                        "value_%": st.column_config.NumberColumn(format="%.2f", required=True),
                        "confidence_%": st.column_config.NumberColumn(min_value=1, max_value=100, default=50),
                    })
            weights_mkt = pd.Series(mkt_table["market_weight_%"].to_numpy(dtype=float) / 100,
                                    index=mkt_table["asset"].to_numpy())
            views = []
            for _, v in views_table.dropna(subset=["type", "asset", "value_%"]).iterrows():
                conf = float(v["confidence_%"]) / 100 if pd.notna(v["confidence_%"]) else 0.5
                if v["type"] == "relative":
                    if pd.isna(v["versus"]):
                        st.warning(f"Relative view on {v['asset']} needs a 'versus' asset; ignored.")
                        continue
                    views.append(relative_view(v["asset"], v["versus"], float(v["value_%"]) / 100, conf))
                else:
                    views.append(absolute_view(v["asset"], float(v["value_%"]) / 100, conf))
            bl_result = black_litterman(cov_used, weights_mkt, risk_aversion=risk_aversion, rf_annual=rf,
                                        views=views, tau=tau)
            table = pd.DataFrame({"Historical mean": annualized_mean_return(asset_returns),
                                  "BL prior (equilibrium)": bl_result.prior,
                                  "BL posterior": bl_result.posterior})
            if capm_result is not None:
                table.insert(1, "CAPM", capm_result.table["capm_expected_return"])
            st.plotly_chart(charts.expected_returns_chart(table, mode=mode), width="stretch")
            with st.expander("Expected returns table"):
                st.dataframe(styled(table, pct_cols=list(table.columns)))
            if len(bl_result.views):
                st.dataframe(styled(bl_result.views.set_index("label"),
                                    pct_cols=["value", "confidence", "prior_implied", "posterior_implied"]))

# --- Optimisation --------------------------------------------------------------- #
    with tab_opt:
        mu, cov = None, None
        if mu_source == "CAPM":
            if capm_result is None:
                st.error("CAPM expected returns need a benchmark with valid data (see the CAPM tab).")
            else:
                mu, cov = capm_result.table["capm_expected_return"], covariance(asset_returns, cov_method)
        elif mu_source == "Black-Litterman":
            if bl_result is None:
                st.error("Black-Litterman returns could not be computed (see the Estimation tab).")
            else:
                mu, cov = bl_result.posterior, bl_result.posterior_cov
        else:
            mu, cov = annualized_mean_return(asset_returns), covariance(asset_returns, cov_method)

        if mu is not None and len(asset_cols) < 2:
            st.info("Optimisation needs at least two assets.")
        elif mu is not None:
            st.caption(f"Long-only, weights sum to 100 %, each ≤ {cap:.0%}. Expected returns: {mu_source}. "
                       f"Covariance: {'Black-Litterman posterior' if mu_source == 'Black-Litterman' else cov_label}.")
            portfolios: dict[str, OptimizationResult] = {}
            builders = {
                "Min variance": lambda: min_variance(cov, mu=mu, max_weight=cap, rf_annual=rf),
                "Max Sharpe": lambda: max_sharpe(mu, cov, rf_annual=rf, max_weight=cap),
                "Inverse volatility": lambda: inverse_volatility(cov, max_weight=cap, mu=mu, rf_annual=rf),
                "Risk parity": lambda: risk_parity(cov, max_weight=cap, mu=mu, rf_annual=rf),
                "Max diversification": lambda: max_diversification(cov, max_weight=cap, mu=mu, rf_annual=rf),
            }
            for name, build in builders.items():
                with guarded(name):
                    portfolios[name] = build()
            ew = pd.Series(1 / len(asset_cols), index=asset_cols, name="weight")
            ew_stats = portfolio_stats(ew, mu, cov, rf)
            portfolios["Equal weight"] = OptimizationResult(ew, ew_stats.expected_return, ew_stats.volatility,
                                                             ew_stats.sharpe, "equal_weight", 0)

            summary = pd.DataFrame({name: {"expected_return": p.expected_return, "volatility": p.volatility,
                                           "sharpe": p.sharpe,
                                           "diversification_ratio": diversification_ratio(p.weights, cov)}
                                    for name, p in portfolios.items()}).T
            left, right = st.columns([3, 2])
            with left:
                st.subheader("Portfolios (ex-ante, from the estimates)")
                st.dataframe(styled(summary, pct_cols=["expected_return", "volatility"],
                                    num_cols=["sharpe", "diversification_ratio"]))
                weights = pd.DataFrame({name: p.weights for name, p in portfolios.items()})
                st.plotly_chart(charts.allocation_bar(weights, mode=mode), width="stretch")
                with st.expander("Weights table"):
                    st.dataframe(styled(weights, pct_cols=list(weights.columns)))
            with right:
                pick = st.selectbox("Allocation of", list(portfolios))
                st.plotly_chart(charts.allocation_pie(portfolios[pick].weights, mode=mode, title=pick),
                                width="stretch")
                with guarded("Risk contributions"):
                    st.plotly_chart(charts.risk_contribution_chart(risk_contributions(portfolios[pick].weights, cov),
                                                                   mode=mode, title=f"{pick}: weight vs risk"),
                                    width="stretch")
            for p in portfolios.values():
                for note in p.notes:
                    st.warning(note)

            st.subheader("Efficient frontier")
            n_points = st.slider("Frontier points", 5, 60, 25)
            highlight = st.multiselect("Highlight (up to 3)", list(portfolios),
                                       default=[p for p in ("Min variance", "Max Sharpe", "Risk parity")
                                                if p in portfolios], max_selections=3)
            with guarded("Efficient frontier"):
                frontier = cached_frontier(mu, cov, n_points, cap, rf)
                asset_points = pd.DataFrame({"volatility": np.sqrt(np.diag(cov)), "expected_return": mu},
                                            index=cov.index)
                st.plotly_chart(charts.frontier_chart(frontier, assets=asset_points,
                                                      portfolios={k: portfolios[k] for k in highlight},
                                                      rf_arith=annual_arithmetic_risk_free(rf), mode=mode),
                                width="stretch")
                for note in frontier.notes:
                    st.warning(note)
                with st.expander("Frontier table"):
                    st.dataframe(styled(pd.concat([frontier.points, frontier.weights], axis=1),
                                        pct_cols=["target_return", "expected_return", "volatility",
                                                  *frontier.weights.columns], num_cols=["sharpe"]))
            st.info(
                "These portfolios are optimal only for the *estimated* inputs. Small changes in the estimates "
                "can move the weights a lot — the Backtest tab shows how each approach performs when "
                "re-estimated without hindsight."
            )

# --- Backtest ----------------------------------------------------------------- #
    with tab_bt:
        st.caption("Walk-forward: at each rebalance the strategy sees only the trailing lookback window, "
                   "trades at that day's close and earns returns from the next day. Costs are charged on "
                   "turnover. SMA-filter strategies move below-trend assets to cash earning the risk-free rate.")
        with st.form("backtest_settings"):
            chosen_strategies = st.multiselect("Strategies (up to 7)", list(STRATEGY_FACTORIES),
                                               default=DEFAULT_STRATEGIES, max_selections=7)
            c1, c2, c3, c4, c5 = st.columns(5)
            lookback = c1.number_input("Lookback (trading days)", 20, 2000, 252, step=21)
            rebalance_label = c2.selectbox("Rebalance", list(REBALANCE_CHOICES))
            every_n = c3.number_input("N (if 'Every N')", 1, 500, 21)
            cost_bps = c4.number_input("Cost (bps per unit traded)", 0.0, 500.0, 10.0, step=5.0)
            sma_window = c5.number_input("SMA filter window", 2, 400, 200, step=10,
                                         help="Needs a lookback of at least window − 1.")
            submitted = st.form_submit_button("Run backtest", type="primary")

        rebalance = REBALANCE_CHOICES[rebalance_label]
        rebalance = int(every_n) if rebalance == "N" else rebalance
        key = (dataset["source"], tuple(asset_cols), str(report.start), str(report.end), tuple(chosen_strategies),
               int(lookback), rebalance, float(cost_bps), rf, cap, cov_method, int(sma_window))
        if submitted:
            if not chosen_strategies:
                st.error("Pick at least one strategy.")
            else:
                with guarded("Backtest"):
                    st.session_state["backtest"] = (key, cached_backtests(
                        asset_returns, tuple(chosen_strategies), int(lookback), rebalance, float(cost_bps),
                        rf, cap, cov_method, int(sma_window)))
        stored = st.session_state.get("backtest")
        if stored and stored[0] != key:
            st.info("Settings or data changed — press **Run backtest** to update.")
        elif stored:
            results, failures = stored[1]
            for name, reason in failures:
                st.error(f"**{name}** could not run — {reason}")
            backtest_results = results or None
        if stored and stored[0] == key and backtest_results:
            bench_named = bench_returns.rename(f"{benchmark} (buy & hold)") if bench_returns is not None else None
            with guarded("Backtest summary"):
                table = backtest_summary(results, rf_annual=rf, benchmark=bench_named)
                st.dataframe(styled(table, pct_cols=["annual_return", "cagr", "volatility", "max_drawdown",
                                                     "total_costs"],
                                    num_cols=["sharpe", "annual_turnover"],
                                    int_cols=["observations", "rebalances", "skipped_rebalances"]))
                st.caption(f"Performance period {results[0].returns.index[0].date()} → "
                           f"{results[0].returns.index[-1].date()}. annual_turnover = average fraction of the "
                           "portfolio traded per year (excluding the first purchase). The Performance tab "
                           "adds Treynor, Jensen, information ratio and market-timing tests.")
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

# --- Trend (SMA) ---------------------------------------------------------------- #
    with tab_trend:
        c1, c2, c3 = st.columns(3)
        fast = c1.number_input("Fast SMA (days)", 2, 400, 50, key="trend_fast")
        slow = c2.number_input("Slow SMA (days)", 3, 500, 200, key="trend_slow")
        trend_asset = c3.selectbox("Chart", list(data.prices.columns), key="trend_asset")
        with guarded("Trend"):
            if fast >= slow:
                st.error("The fast SMA must be shorter than the slow SMA.")
            else:
                status = trend_status(data.prices, int(fast), int(slow))
                st.dataframe(styled(status, pct_cols=["distance_from_slow"],
                                    money_cols=["price", f"sma_{fast}", f"sma_{slow}"]))
                px = data.prices[trend_asset]
                events = crosses(px, int(fast), int(slow))
                st.plotly_chart(charts.trend_chart(px, {f"SMA {fast}": sma(px, int(fast)),
                                                        f"SMA {slow}": sma(px, int(slow))},
                                                   events=events, mode=mode), width="stretch")
                with st.expander(f"Crosses for {trend_asset} ({len(events)})"):
                    st.dataframe(events, hide_index=True)

# --- Technical analysis --------------------------------------------------------- #
    with tab_ta:
        st.caption("Lecture rules: MACD crossing its signal line, RSI above 70 / below 30, closes outside the "
                   "Bollinger bands. Shown for learning — the evidence that these rules beat the market after "
                   "costs is weak, so use them alongside fundamental research.")
        c1, c2, c3, c4 = st.columns(4)
        ta_asset = c1.selectbox("Security", list(data.prices.columns), key="ta_asset")
        rsi_window = c2.number_input("RSI window", 2, 100, 14, key="ta_rsi")
        rsi_method = c3.selectbox("RSI averaging", ["simple", "wilder"], key="ta_rsi_method",
                                  help="'simple' as in the lecture; 'wilder' is the common smoothed version.")
        macd_kind = c4.selectbox("MACD signal line", ["sma", "ema"], key="ta_macd",
                                 help="Lecture: 9-day SMA of MACD. Many platforms use a 9-day EMA.")
        c1, c2, c3, c4 = st.columns(4)
        rsi_hi = c1.number_input("Overbought above", 50, 95, 70, key="ta_hi")
        rsi_lo = c2.number_input("Oversold below", 5, 50, 30, key="ta_lo")
        bb_n = c3.number_input("Bollinger window", 5, 100, 20, key="ta_bbn")
        bb_k = c4.number_input("Bollinger k (σ)", 0.5, 4.0, 2.0, 0.5, key="ta_bbk")
        with guarded("Technical indicators"):
            px = data.prices[ta_asset]
            readings = ta.latest_readings(px, rsi_window=int(rsi_window), rsi_upper=rsi_hi, rsi_lower=rsi_lo,
                                          bb_window=int(bb_n), bb_k=bb_k, macd_signal_kind=macd_kind,
                                          rsi_method=rsi_method)
            st.subheader(f"{ta_asset}: latest readings ({px.index[-1].date()})")
            st.dataframe(readings)
            events = ta.technical_signals(px, rsi_window=int(rsi_window), rsi_upper=rsi_hi, rsi_lower=rsi_lo,
                                          bb_window=int(bb_n), bb_k=bb_k, macd_signal_kind=macd_kind,
                                          rsi_method=rsi_method)
            st.plotly_chart(charts.technical_chart(
                px, bands=ta.bollinger_bands(px, int(bb_n), bb_k), macd_frame=ta.macd(px, signal_kind=macd_kind),
                rsi_line=ta.rsi(px, int(rsi_window), rsi_method), ema_lines={"EMA 20": ta.ema(px, 20)},
                rsi_levels=(rsi_lo, rsi_hi), events=events, mode=mode), width="stretch")
            with st.expander(f"Signal log ({len(events)} events)"):
                st.dataframe(events.iloc[::-1], hide_index=True)

        st.subheader("Candlestick patterns")
        ohlc_data = None
        with guarded("Candlesticks"):
            if dataset["source"] == "Yahoo Finance":
                if st.button(f"Load daily candles for {ta_asset}", key="ta_load_ohlc"):
                    st.session_state["ohlc"] = cached_ohlcv(ta_asset, dataset["start"], dataset["end"])
            else:
                ohlc_file = st.file_uploader("OHLC CSV (Date, Open, High, Low, Close[, Volume])", type="csv",
                                             key="ta_ohlc_csv")
                if ohlc_file is not None:
                    st.session_state["ohlc"] = load_ohlcv_csv(ohlc_file, ticker=ohlc_file.name)
            ohlc_data = st.session_state.get("ohlc")
            if ohlc_data is None:
                st.caption("Candlestick patterns need Open/High/Low/Close, which the adjusted price table does "
                           "not have. Load them above.")
            else:
                c1, c2 = st.columns(2)
                n_days = c1.slider("Days shown", 30, 500, 120, key="ta_days")
                trend_window = c2.number_input("Prior-trend window (days)", 2, 30, 5, key="ta_trendwin")
                frame = ohlc_data.frame
                patterns = ta.candlestick_patterns(frame, trend_window=int(trend_window))
                recent = frame.iloc[-n_days:]
                shown = patterns[patterns["date"] >= recent.index[0]]
                st.plotly_chart(charts.candlestick_chart(recent, patterns=shown, mode=mode,
                                                         title=f"{ohlc_data.ticker}: daily candles"), width="stretch")
                if ohlc_data.dropped_rows:
                    st.caption(f"{ohlc_data.dropped_rows} incomplete day(s) were dropped, not filled.")
                st.dataframe(shown.iloc[::-1], hide_index=True)
                st.caption("Hammer, hanging man and doji need confirmation by the next session before acting.")
                if "Volume" in frame:
                    vc = ta.volume_confirmation(frame["Close"], frame["Volume"])
                    st.caption("Volume confirmation (last 10 days): a move on above-average volume is strong.")
                    st.dataframe(vc.iloc[-10:].iloc[::-1], width="stretch")

# --- Valuation (fundamental analysis) ------------------------------------------------ #
with tab_fa:
    st.caption("Intrinsic value per share from the course models. Defaults reproduce the lecture examples. "
               "Signal: intrinsic value above the market price → Buy; below → Sell.")
    model = st.radio("Model", ["Multistage dividend discount", "Payout & ROE driven DDM", "Free cash flow (DCF)",
                               "Relative valuation (multiples)", "Quick tools"], horizontal=True, key="fa_model")

    def required_return_widget(prefix: str, default_k: float) -> float:
        how = st.radio("Required return k", ["Enter k", "CAPM: rf + β × MRP"], horizontal=True,
                       key=f"{prefix}_khow")
        if how == "Enter k":
            return pct_input("k (% per year)", default_k * 100, key=f"{prefix}_k")
        c1, c2 = st.columns(2)
        beta = c1.number_input("Beta", -3.0, 5.0, 1.0, 0.05, key=f"{prefix}_beta")
        mrp = pct_input("Market risk premium (%)", 7.0, key=f"{prefix}_mrp")
        k = fa.capm_required_return(rf, beta, mrp)
        st.caption(f"k = {rf:.2%} + {beta:.2f} × {mrp:.2%} = **{k:.2%}** (rf from the sidebar)")
        return k

    def show_dcf(res: fa.DCFResult, price: float | None, unit: str = "dividend"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Intrinsic value", f"{res.value:,.2f}")
        if price:
            c2.metric("Market price", f"{price:,.2f}", f"{res.value / price - 1:+.1%} upside")
            c3.metric("Signal", fa.valuation_signal(res.value, price))
        st.plotly_chart(charts.dcf_chart(res.schedule, res.pv_terminal, mode=mode), width="stretch")
        sched = res.schedule.rename(columns={"cash_flow": unit})
        st.dataframe(styled(sched, pct_cols=["growth"], money_cols=[unit, "present_value"],
                            num_cols=["discount_factor"]))
        st.caption(f"Terminal value at year {len(res.schedule)}: {res.terminal_value:,.2f} "
                   f"(next {unit} {res.terminal_cash_flow:,.2f} growing at {res.terminal_growth:.2%}); "
                   f"it is {res.terminal_share:.0%} of today's value.")

    with guarded("Valuation"):
        if model == "Multistage dividend discount":
            c1, c2, c3 = st.columns(3)
            d0 = c1.number_input("Dividend just paid D0", 0.0, 1e6, 6.0, 0.5, key="fa_d0")
            price = c2.number_input("Market price", 0.01, 1e7, 200.0, 1.0, key="fa_p")
            years = c3.number_input("High-growth years", 0, 30, 3, key="fa_n")
            c1, c2 = st.columns(2)
            g1 = pct_input("High growth g1 (%)", 8.0, key="fa_g1")
            g2 = pct_input("Perpetual growth g2 (%)", 6.0, key="fa_g2")
            k = required_return_widget("fa_ddm", 0.10)
            show_dcf(fa.multistage_ddm(d0, [g1] * int(years), g2, k), price)
            st.caption("Default inputs: lecture 22, problem 1 (answer ≈ ₹168, sell).")
        elif model == "Payout & ROE driven DDM":
            c1, c2, c3 = st.columns(3)
            eps0 = c1.number_input("EPS today E0", 0.01, 1e6, 20.0, 1.0, key="fa_eps")
            price = c2.number_input("Market price", 0.01, 1e7, 80.0, 1.0, key="fa_p2")
            years = c3.number_input("Years in stage 1", 1, 30, 4, key="fa_n2")
            c1, c2, c3, c4 = st.columns(4)
            pay1 = c1.number_input("Stage-1 payout %", 0.0, 100.0, 25.0, key="fa_pay1") / 100
            roe1 = c2.number_input("Stage-1 ROE %", -50.0, 100.0, 20.0, key="fa_roe1") / 100
            pay2 = c3.number_input("Later payout %", 0.0, 100.0, 50.0, key="fa_pay2") / 100
            roe2 = c4.number_input("Later ROE %", -50.0, 100.0, 20.0, key="fa_roe2") / 100
            k = required_return_widget("fa_edd", 0.30)
            res = fa.earnings_ddm(eps0, [fa.EarningsStage(int(years), pay1, roe1)], pay2, roe2, k)
            st.caption(f"Growth g = retention × ROE: stage 1 {(1 - pay1) * roe1:.2%}, later {(1 - pay2) * roe2:.2%}. "
                       "Default inputs: lecture 22, problem 2.")
            show_dcf(res, price)
        elif model == "Free cash flow (DCF)":
            basis = st.radio("Cash flow", ["FCFE (discount at cost of equity)", "FCFF (discount at WACC)"],
                             horizontal=True, key="fa_basis")
            st.caption("FCFF = EBIT(1 − t) + depreciation − capex − ΔNWC.  "
                       "FCFE = FCFF − interest(1 − t) + net borrowing.")
            c1, c2, c3, c4, c5 = st.columns(5)
            ebit = c1.number_input("EBIT", -1e9, 1e9, 1000.0, key="fa_ebit")
            tax = c2.number_input("Tax rate %", 0.0, 99.0, 25.0, key="fa_tax") / 100
            dep = c3.number_input("Depreciation", 0.0, 1e9, 200.0, key="fa_dep")
            capex = c4.number_input("Capex", 0.0, 1e9, 350.0, key="fa_capex")
            dnwc = c5.number_input("Increase in NWC", -1e9, 1e9, 50.0, key="fa_dnwc")
            base = fa.fcff(ebit, tax, dep, capex, dnwc)
            c1, c2, c3, c4 = st.columns(4)
            interest = c1.number_input("Interest", 0.0, 1e9, 100.0, key="fa_int")
            borrowing = c2.number_input("Net borrowing", -1e9, 1e9, 0.0, key="fa_borrow")
            net_debt = c3.number_input("Net debt", -1e9, 1e10, 1000.0, key="fa_nd")
            shares = c4.number_input("Shares outstanding", 1.0, 1e10, 100.0, key="fa_sh")
            is_fcfe = basis.startswith("FCFE")
            cf0 = fa.fcfe_from_fcff(base, interest, tax, borrowing) if is_fcfe else base
            st.caption(f"Current FCFF {base:,.2f}" + (f" · FCFE {cf0:,.2f}" if is_fcfe else ""))
            c1, c2, c3 = st.columns(3)
            years = c1.number_input("Explicit years", 1, 20, 5, key="fa_fy")
            g1 = pct_input("Growth in explicit years (%)", 10.0, key="fa_fg1")
            g2 = pct_input("Terminal growth (%)", 4.0, key="fa_fg2")
            r = (required_return_widget("fa_dcf", 0.12) if is_fcfe
                 else pct_input("WACC (%)", 10.0, key="fa_wacc"))
            price = st.number_input("Market price per share", 0.01, 1e7, 60.0, key="fa_p3")
            flows = [cf0 * (1 + g1) ** t for t in range(1, int(years) + 1)]
            res = fa.dcf_value(flows, g2, r)
            if is_fcfe:
                per_share = fa.DCFResult(res.value / shares, res.schedule.assign(
                    cash_flow=res.schedule["cash_flow"] / shares,
                    present_value=res.schedule["present_value"] / shares), res.terminal_value / shares,
                    res.pv_terminal / shares, res.terminal_cash_flow / shares, res.terminal_growth, r)
                show_dcf(per_share, price, unit="FCFE per share")
            else:
                eq = fa.equity_from_firm_value(res.value, net_debt, shares)
                st.metric("Equity value per share = (firm value − net debt) / shares", f"{eq:,.2f}",
                          f"{fa.valuation_signal(eq, price)} vs {price:,.2f}")
                show_dcf(res, None, unit="FCFF (firm)")
        elif model == "Relative valuation (multiples)":
            st.caption("Implied price = peer (or industry) multiple × the company's own metric. "
                       "EV/EBITDA: (multiple × EBITDA − net debt) / shares.")
            c1, c2, c3 = st.columns(3)
            company = {
                "eps": c1.number_input("EPS", -1e6, 1e6, 10.0, key="rv_eps"),
                "book_value_per_share": c2.number_input("Book value / share", -1e6, 1e6, 60.0, key="rv_bv"),
                "sales_per_share": c3.number_input("Sales / share", 0.0, 1e7, 120.0, key="rv_sales"),
            }
            c1, c2, c3, c4 = st.columns(4)
            company["cash_flow_per_share"] = c1.number_input("Cash flow / share", -1e6, 1e6, 14.0, key="rv_cf")
            company["ebitda"] = c2.number_input("EBITDA (total)", -1e10, 1e10, 2500.0, key="rv_ebitda")
            company["net_debt"] = c3.number_input("Net debt (total)", -1e10, 1e10, 3000.0, key="rv_nd")
            company["shares"] = c4.number_input("Shares outstanding", 1.0, 1e10, 100.0, key="rv_sh")
            peers = st.data_editor(pd.DataFrame({"multiple": list(fa.MULTIPLES), "peer_value": [18.0, 2.5, 1.6, 11.0, 9.0]}),
                                   key="rv_peers", hide_index=True, disabled=["multiple"])
            price = st.number_input("Market price", 0.01, 1e7, 170.0, key="rv_price")
            table = fa.relative_valuation(company, dict(zip(peers["multiple"], peers["peer_value"].astype(float))),
                                          market_price=price)
            st.dataframe(styled(table, pct_cols=["upside"], money_cols=["implied_price", "company_metric"],
                                num_cols=["peer_multiple"]))
            c1, c2 = st.columns(2)
            g = pct_input("Expected EPS growth for PEG (%)", 15.0, key="rv_g")
            pe_now = price / company["eps"] if company["eps"] > 0 else float("nan")
            c1.metric("Current P/E", f"{pe_now:.1f}" if np.isfinite(pe_now) else "undefined")
            if np.isfinite(pe_now) and g > 0:
                c2.metric("PEG", f"{fa.peg_ratio(pe_now, g):.2f}")
        else:
            st.markdown("**Justified P/E and PVGO**")
            c1, c2, c3, c4 = st.columns(4)
            roe = c1.number_input("ROE %", -50.0, 100.0, 15.0, key="qt_roe") / 100
            b = c2.number_input("Retention b %", 0.0, 100.0, 60.0, key="qt_b") / 100
            k = c3.number_input("k %", 0.1, 100.0, 12.5, key="qt_k") / 100
            eps0 = c4.number_input("EPS today", 0.01, 1e6, 2.5, key="qt_eps")
            g = fa.sustainable_growth(b, roe)
            fpe, tpe = fa.forward_pe(b, roe, k), fa.trailing_pe(b, roe, k)
            value = fa.gordon_value(k, g, d1=eps0 * (1 + g) * (1 - b))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("g = b × ROE", f"{g:.2%}")
            c2.metric("Forward P/E", f"{fpe:.2f}")
            c3.metric("Trailing P/E", f"{tpe:.2f}")
            c4.metric("Value (Gordon)", f"{value:,.2f}")
            c1, c2 = st.columns(2)
            c1.metric("No-growth value E1 / k", f"{eps0 * (1 + g) / k:,.2f}")
            c2.metric("PVGO = P0 − E1 / k", f"{fa.pvgo(value, eps0 * (1 + g), k):,.2f}",
                      help="Negative when ROE < k: retaining earnings destroys value.")
            st.caption("Defaults: lecture 25 example (P/E 11.43 vs market 10 → underpriced; value ≈ 31.14).")
            st.markdown("**Holding-period return vs required return**")
            c1, c2, c3 = st.columns(3)
            p0 = c1.number_input("Price today", 0.01, 1e7, 100.0, key="qt_p0")
            d1 = c2.number_input("Expected dividend D1", 0.0, 1e6, 4.0, key="qt_d1")
            p1 = c3.number_input("Expected price P1", 0.0, 1e7, 110.0, key="qt_p1")
            hpr = fa.holding_period_return(p0, d1, p1)
            req = required_return_widget("qt_hpr", 0.12)
            st.metric("Expected HPR", f"{hpr:.2%}", f"{fa.return_signal(hpr, req)} (k = {req:.2%})")

# --- Performance measurement ----------------------------------------------------- #
with tab_ppm:
    ppm_mode = st.radio("Input", ["From loaded data", "From summary statistics (lecture problems)"],
                        horizontal=True, key="ppm_mode")
    if ppm_mode.startswith("From summary"):
        st.caption("Annual average return, beta and standard deviation per portfolio. Defaults: lecture 30 "
                   "exercise (P, Q, R, S; market 13 %, σ 4 %; rf 6 %).")
        inputs = st.data_editor(pd.DataFrame({"portfolio": list("PQRS"), "return_%": [15.0, 20.0, 10.0, 17.0],
                                              "beta": [1.0, 1.5, 0.6, 1.1], "std_%": [5.0, 10.0, 3.0, 6.0]}),
                                key="ppm_summary", num_rows="dynamic", hide_index=True)
        c1, c2, c3 = st.columns(3)
        m_ret = c1.number_input("Market return %", -50.0, 100.0, 13.0, key="ppm_mr") / 100
        m_sd = c2.number_input("Market std %", 0.01, 200.0, 4.0, key="ppm_msd") / 100
        rf_s = c3.number_input("Risk-free %", -5.0, 50.0, 6.0, key="ppm_rf") / 100
        with guarded("Performance measures"):
            clean = inputs.dropna(subset=["portfolio", "return_%", "beta"])
            frame = pd.DataFrame({"return": clean["return_%"].to_numpy() / 100, "beta": clean["beta"].to_numpy(),
                                  "std": clean["std_%"].to_numpy() / 100}, index=clean["portfolio"].astype(str))
            table = ppm.measures_from_summary(frame, market_return=m_ret, market_std=m_sd, risk_free=rf_s)
            st.dataframe(styled(table, pct_cols=["return", "std", "excess_return", "treynor", "jensen_alpha",
                                                 "m_squared", "m_squared_vs_market"],
                                num_cols=["beta", "sharpe", "r_squared"], int_cols=["sharpe_rank", "treynor_rank"],
                                extra={"systematic_variance": "{:.4f}", "unsystematic_variance": "{:.4f}"}))
            st.plotly_chart(charts.security_market_line(table, rf=rf_s, market_return=m_ret, mode=mode),
                            width="stretch")
            st.caption("Sharpe and Treynor agree for fully diversified portfolios. When they rank differently, "
                       "look at unsystematic variance / R²: Sharpe penalises undiversified risk, Treynor ignores it.")
    elif dataset is None or bench_returns is None:
        st.info("Load data with a benchmark to measure performance from returns, or switch to summary statistics.")
    else:
        c1, c2 = st.columns(2)
        freq_label = c1.selectbox("Return frequency", list(FREQUENCIES), index=2, key="ppm_freq",
                                  help="Monthly returns are the usual choice for fund evaluation.")
        include_bt = c2.toggle("Include backtest strategies", value=backtest_results is not None,
                               disabled=backtest_results is None,
                               help="Run a backtest first to evaluate its strategies here.")
        freq, periods = FREQUENCIES[freq_label]
        with guarded("Performance measures"):
            port = asset_returns.copy()
            if include_bt and backtest_results is not None:
                bt = pd.concat([r.returns for r in backtest_results], axis=1)
                port = port.loc[bt.index].join(bt)
                st.caption(f"Evaluation period limited to the backtest: {bt.index[0].date()} → {bt.index[-1].date()}.")
            bench = bench_returns.loc[port.index]
            port_p, bench_p = ppm.periodic_returns(port, freq), ppm.periodic_returns(bench, freq)
            table = ppm.performance_measures(port_p, bench_p, rf_annual=rf, periods_per_year=periods,
                                             min_observations=12)
            st.dataframe(styled(table, pct_cols=["annual_return", "annual_excess_return", "volatility", "treynor",
                                                 "jensen_alpha", "tracking_error", "m_squared",
                                                 "m_squared_vs_benchmark"],
                                num_cols=["beta", "sharpe", "alpha_t_stat", "information_ratio", "r_squared"],
                                int_cols=["observations", "sharpe_rank", "treynor_rank"],
                                extra={"systematic_variance": "{:.4f}", "unsystematic_variance": "{:.4f}"}))
            st.caption(f"{freq_label} returns, annualised with {periods} periods per year. Jensen's alpha is "
                       "significant at about 5 % when |t| > 2. M² is the return at the benchmark's volatility.")
            sml = table.rename(columns={"annual_return": "return"})
            m_row = sml.loc[str(bench_p.name)]
            st.plotly_chart(charts.security_market_line(sml, rf=annual_arithmetic_risk_free(rf) if freq == "D"
                                                        else periods * ((1 + rf) ** (1 / periods) - 1),
                                                        market_return=float(m_row["return"]), mode=mode),
                            width="stretch")

            st.subheader("Market timing")
            c1, c2 = st.columns(2)
            model_label = c1.selectbox("Model", ["Treynor-Mazuy (squared term)", "Henriksson-Merton (two betas)"],
                                       key="ppm_model")
            model_key = ppm.TREYNOR_MAZUY if model_label.startswith("Treynor") else ppm.HENRIKSSON_MERTON
            timing = ppm.market_timing(port_p, bench_p, rf_annual=rf, model=model_key, periods_per_year=periods,
                                       min_observations=12)
            st.dataframe(styled(timing.table, num_cols=["alpha_t", "beta", "beta_t", "gamma_t", "r_squared",
                                                        "up_market_beta"], int_cols=["observations"],
                                extra={"alpha": "{:.5f}", "gamma": "{:.4f}"}))
            st.caption(f"γ > 0 with t > 2 suggests timing ability. {timing.up_periods} up-market and "
                       f"{timing.down_periods} down-market periods. Alpha is per period.")
            who = c2.selectbox("Plot", list(timing.table.index), key="ppm_who")
            rf_p = (1 + rf) ** (1 / periods) - 1
            row = timing.table.loc[who]
            st.plotly_chart(charts.timing_scatter(port_p[who] - rf_p, bench_p - rf_p, alpha=row["alpha"],
                                                  beta=row["beta"], gamma=row["gamma"], model=model_key, mode=mode),
                            width="stretch")

# --- Methodology -------------------------------------------------------------- #
with tab_method:
    st.markdown(f"""
**Conventions** (full derivations in `docs/ARCHITECTURE.md`)

- Prices: adjusted close. Dates where any series is missing are **dropped and reported, never filled**.
- Returns: simple daily returns. Annualisation: mean × {TRADING_DAYS_PER_YEAR}, volatility × √{TRADING_DAYS_PER_YEAR}.
- Risk-free: annual effective rate → daily (1+r)^(1/{TRADING_DAYS_PER_YEAR}) − 1; compared with
  arithmetic returns as {TRADING_DAYS_PER_YEAR} × daily rate.
- CAPM: OLS of asset excess returns on benchmark excess returns over identical dates.
- Estimation: Ledoit-Wolf shrinkage towards a scaled identity; Black-Litterman with He-Litterman Ω and
  confidence-scaled view uncertainty.
- Optimisation: SciPy SLSQP, long-only, fully invested, per-asset cap; every solution re-verified. Risk parity
  by Newton's method; maximum diversification as max-Sharpe with μ = σ.
- Backtest: walk-forward, trailing window only, trade at close, drift between rebalances, cost = bps × turnover,
  cash earns the risk-free rate.
- Trend & technical analysis: SMA, EMA (SMA-seeded), Bollinger (20, 2σ), MACD (12, 26, 9-day SMA signal as in the
  lecture), RSI (14, simple averages), candlestick patterns with a prior-trend filter.
- Valuation: zero-growth, Gordon, multistage and payout/ROE-driven DDM, PVGO, FCFF/FCFE DCF, justified P/E and
  peer multiples.
- Performance: Sharpe, Treynor, Jensen's alpha (with t-stat), information ratio, M², systematic vs unsystematic
  risk, Treynor-Mazuy and Henriksson-Merton market timing.

**Limitations.** Expected returns estimated from history are very noisy and optimised weights are highly
sensitive to them. yfinance is an unofficial data source. Backtests ignore taxes, slippage beyond the fixed cost,
and liquidity. Technical signals are shown for learning and are not evidence of an edge. Valuation outputs are
only as good as the growth and discount-rate assumptions typed in. Past performance does not predict future
results.
""")
