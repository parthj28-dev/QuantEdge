"""Plotly figure builders — pure functions, no Streamlit.

Every function takes QuantEdge results (pandas objects / dataclasses) and returns a
`plotly.graph_objects.Figure`. Because nothing here touches Streamlit, the same
figures work in a notebook, can be saved with `fig.write_html(...)`, and can be
unit-tested by inspecting their data.

Visual rules (applied consistently):
* Colours come from one validated categorical palette, assigned in a fixed order and
  never cycled. Light and dark modes use separately validated steps.
* Scatter-type charts colour at most 3 series (beyond that, colours become hard to
  tell apart for colour-blind readers); other points are neutral grey with labels.
* One y-axis per chart; percentages formatted as percentages; hover on every mark.
* Backgrounds are transparent so the host page (Streamlit, notebook) shows through.
* Missing or undefined values (e.g. NaN correlation) are shown as gaps and labelled
  "undefined" in the hover — never drawn as 0.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .backtest import BacktestResult
from .frontier import FrontierResult
from .metrics import drawdown_series
from .optimize import OptimizationResult

# --------------------------------------------------------------------------- #
# Theme (validated palette; see docs/ARCHITECTURE.md)
# --------------------------------------------------------------------------- #
THEMES: dict[str, dict] = {
    "light": {
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
        "ink": "#0b0b0b",
        "ink_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "diverging": ["#256abf", "#f0efec", "#e34948"],  # -1 (blue), 0 (neutral grey), +1 (red)
    },
    "dark": {
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
        "ink": "#f0efec",
        "ink_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "diverging": ["#3987e5", "#383835", "#e66767"],
    },
}
MAX_CATEGORICAL = 8          # line / bar / stacked series
MAX_SCATTER_SERIES = 3       # coloured series where every pair must be distinguishable
_FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def _theme(mode: str) -> dict:
    if mode not in THEMES:
        raise ValueError(f"mode must be 'light' or 'dark', got {mode!r}.")
    return THEMES[mode]


def _layout(fig: go.Figure, theme: dict, title: str, x_title: str, y_title: str,
            *, x_pct: bool = False, y_pct: bool = False, legend: bool = True) -> go.Figure:
    axis = dict(gridcolor=theme["grid"], linecolor=theme["axis"], zerolinecolor=theme["axis"],
                tickfont=dict(color=theme["muted"]), title_font=dict(color=theme["ink_secondary"]))
    fig.update_layout(
        title=dict(text=title, font=dict(color=theme["ink"], size=16)),
        font=dict(family=_FONT, color=theme["ink_secondary"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=theme["ink_secondary"])),
        margin=dict(l=60, r=20, t=70, b=50),
        hoverlabel=dict(font_family=_FONT),
    )
    fig.update_xaxes(title_text=x_title, tickformat=".0%" if x_pct else None, **axis)
    fig.update_yaxes(title_text=y_title, tickformat=".0%" if y_pct else None, **axis)
    return fig


def _require(obj, what: str) -> None:
    if obj is None or len(obj) == 0:
        raise ValueError(f"Nothing to plot: {what} is empty.")


# --------------------------------------------------------------------------- #
# Allocations
# --------------------------------------------------------------------------- #
def grouped_bar(frame: pd.DataFrame | pd.Series, *, mode: str = "light", title: str = "",
                y_title: str = "", pct: bool = True) -> go.Figure:
    """Grouped bars: rows on the x-axis, one colour per column."""
    theme = _theme(mode)
    frame = frame.to_frame() if isinstance(frame, pd.Series) else frame
    _require(frame, "values")
    if frame.shape[1] > MAX_CATEGORICAL:
        raise ValueError(f"At most {MAX_CATEGORICAL} series can be compared in one chart.")
    fmt = "%{y:.1%}" if pct else "%{y:.3f}"
    fig = go.Figure()
    for i, column in enumerate(frame.columns):
        fig.add_bar(
            x=[str(a) for a in frame.index], y=frame[column].to_numpy(dtype=float), name=str(column),
            marker=dict(color=theme["series"][i], line=dict(width=0)),
            hovertemplate="%{x}<br>" + f"{column}: " + fmt + "<extra></extra>",
        )
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, barcornerradius=4)
    _layout(fig, theme, title, "", y_title, y_pct=pct, legend=frame.shape[1] > 1)
    return fig


def allocation_bar(weights: pd.DataFrame | pd.Series, *, mode: str = "light",
                   title: str = "Portfolio weights") -> go.Figure:
    """Grouped bars: assets on the x-axis, one colour per portfolio (columns of `weights`)."""
    frame = weights.to_frame() if isinstance(weights, pd.Series) else weights
    _require(frame, "weights")
    if frame.shape[1] > MAX_CATEGORICAL:
        raise ValueError(f"At most {MAX_CATEGORICAL} portfolios can be compared in one chart.")
    fig = grouped_bar(frame, mode=mode, title=title, y_title="Weight")
    fig.update_yaxes(rangemode="tozero")
    return fig


def expected_returns_chart(table: pd.DataFrame, *, mode: str = "light") -> go.Figure:
    """Expected returns of each asset under each estimator (columns), e.g. historical / CAPM / BL."""
    return grouped_bar(table, mode=mode, title="Expected annual return by estimator",
                       y_title="Expected return")


def risk_contribution_chart(contrib: pd.DataFrame, *, mode: str = "light", title: str = "") -> go.Figure:
    """Weight vs share of portfolio risk per asset (from riskbased.risk_contributions)."""
    frame = contrib[["weight", "risk_share"]].rename(columns={"weight": "Weight", "risk_share": "Share of risk"})
    return grouped_bar(frame, mode=mode, title=title or "Weight vs contribution to risk", y_title="Share")


def allocation_pie(weights: pd.Series, *, mode: str = "light", title: str = "Allocation",
                   max_slices: int = 6, min_weight: float = 0.005) -> go.Figure:
    """Donut for part-to-whole at a glance. Small and excess holdings fold into 'Other'."""
    theme = _theme(mode)
    _require(weights, "weights")
    if not 2 <= max_slices <= MAX_CATEGORICAL:
        raise ValueError(f"max_slices must be between 2 and {MAX_CATEGORICAL}.")
    w = weights.astype(float).sort_values(ascending=False)
    main = w[w >= min_weight]
    if len(main) > max_slices:          # leave one slice free for "Other"
        main = main.iloc[: max_slices - 1]
    other = w.drop(main.index)
    labels, values = [str(a) for a in main.index], list(main.to_numpy())
    other = other[other > 1e-6]          # optimiser dust (~1e-10) is not a holding
    if len(other):
        labels.append(f"Other ({len(other)})")
        values.append(float(other.sum()))
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.55, sort=False, direction="clockwise",
        marker=dict(colors=theme["series"][: len(labels)], line=dict(color="rgba(0,0,0,0)", width=2)),
        textinfo="label+percent", texttemplate="%{label}<br>%{value:.1%}",
        hovertemplate="%{label}: %{value:.2%}<extra></extra>",
    ))
    return _layout(fig, theme, title, "", "", legend=False)


def weights_over_time(result: BacktestResult, *, mode: str = "light") -> go.Figure:
    """Stacked step-area of target weights at each rebalance (weights drift in between)."""
    theme = _theme(mode)
    w = result.weights
    _require(w, "rebalance weights")
    if w.shape[1] > MAX_CATEGORICAL:  # keep the 7 largest average holdings, fold the rest
        keep = w.mean().sort_values(ascending=False).index[: MAX_CATEGORICAL - 1]
        w = pd.concat([w[keep], w.drop(columns=keep).sum(axis=1).rename("Other")], axis=1)
    fig = go.Figure()
    for i, asset in enumerate(w.columns):
        fig.add_scatter(
            x=w.index, y=w[asset], name=str(asset), mode="lines", stackgroup="weights",
            line=dict(width=0.5, color=theme["series"][i], shape="hv"),
            hovertemplate=f"{asset}: " + "%{y:.1%}<extra></extra>",
        )
    _layout(fig, theme, f"{result.name}: target weights at each rebalance", "", "Weight", y_pct=True)
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(range=[0, 1])
    return fig


# --------------------------------------------------------------------------- #
# Risk / return
# --------------------------------------------------------------------------- #
def risk_return_scatter(stats: pd.DataFrame, *, mode: str = "light",
                        x: str = "volatility", y: str = "annual_return",
                        title: str = "Risk vs return (historical)") -> go.Figure:
    """One labelled point per row of a performance table (e.g. metrics.performance_summary)."""
    theme = _theme(mode)
    _require(stats, "statistics table")
    fig = go.Figure(go.Scatter(
        x=stats[x], y=stats[y], text=[str(i) for i in stats.index], mode="markers+text",
        textposition="top center", textfont=dict(color=theme["ink_secondary"]),
        marker=dict(size=11, color=theme["series"][0], line=dict(width=2, color="rgba(255,255,255,0.9)")),
        hovertemplate="%{text}<br>Volatility %{x:.1%}<br>Return %{y:.1%}<extra></extra>",
    ))
    return _layout(fig, theme, title, "Annualised volatility", "Annualised return (arithmetic)",
                   x_pct=True, y_pct=True, legend=False)


def frontier_chart(
    frontier: FrontierResult,
    *,
    assets: pd.DataFrame | None = None,
    portfolios: Mapping[str, OptimizationResult] | None = None,
    rf_arith: float | None = None,
    mode: str = "light",
) -> go.Figure:
    """Efficient frontier, individual assets (grey, labelled), highlighted portfolios and
    optionally the capital market line from the risk-free rate through the best-Sharpe portfolio.

    `assets` needs columns 'volatility' and 'expected_return' (same units as the frontier).
    """
    theme = _theme(mode)
    pts = frontier.points
    _require(pts, "frontier points")
    portfolios = dict(portfolios or {})
    if len(portfolios) > MAX_SCATTER_SERIES:
        raise ValueError(f"Highlight at most {MAX_SCATTER_SERIES} portfolios so colours stay distinguishable.")

    fig = go.Figure()
    fig.add_scatter(
        x=pts["volatility"], y=pts["expected_return"], name="Efficient frontier", mode="lines+markers",
        line=dict(width=2, color=theme["ink_secondary"]), marker=dict(size=5, color=theme["ink_secondary"]),
        hovertemplate="Volatility %{x:.2%}<br>Expected return %{y:.2%}<extra>Frontier</extra>",
    )
    if assets is not None and len(assets):
        fig.add_scatter(
            x=assets["volatility"], y=assets["expected_return"], name="Individual assets",
            mode="markers+text", text=[str(a) for a in assets.index], textposition="middle right",
            textfont=dict(color=theme["muted"], size=11),
            marker=dict(size=9, color=theme["muted"], symbol="circle-open", line=dict(width=2)),
            hovertemplate="%{text}<br>Volatility %{x:.2%}<br>Expected return %{y:.2%}<extra></extra>",
        )
    if rf_arith is not None and portfolios:
        best = max(portfolios.values(), key=lambda p: -np.inf if np.isnan(p.sharpe) else p.sharpe)
        if np.isfinite(best.sharpe) and best.sharpe > 0:
            x_end = float(max(pts["volatility"].max(), best.volatility) * 1.05)
            fig.add_scatter(
                x=[0.0, x_end], y=[rf_arith, rf_arith + best.sharpe * x_end], mode="lines",
                name="Capital market line", line=dict(width=1.5, dash="dash", color=theme["muted"]),
                hovertemplate="CML<br>Volatility %{x:.2%}<br>Return %{y:.2%}<extra></extra>",
            )
    for i, (label, p) in enumerate(portfolios.items()):
        fig.add_scatter(
            x=[p.volatility], y=[p.expected_return], name=label, mode="markers+text",
            # Labels sit away from the frontier, which rises up and to the right.
            text=[label], textposition=["middle left", "top left", "bottom right"][i],
            textfont=dict(color=theme["ink"]),
            marker=dict(size=14, color=theme["series"][i], symbol=["diamond", "star", "square"][i],
                        line=dict(width=2, color="rgba(255,255,255,0.9)")),
            hovertemplate=f"{label}<br>Volatility " + "%{x:.2%}<br>Expected return %{y:.2%}"
                          + f"<br>Sharpe {p.sharpe:.2f}<extra></extra>",
        )
    _layout(fig, theme, "Efficient frontier (long-only, capped)", "Annualised volatility",
            "Expected annual return", x_pct=True, y_pct=True)
    fig.update_xaxes(rangemode="tozero")
    return fig


def correlation_heatmap(corr: pd.DataFrame, *, mode: str = "light",
                        title: str = "Correlation of daily returns") -> go.Figure:
    """Diverging blue (-1) / grey (0) / red (+1) heatmap with the values printed in each cell."""
    theme = _theme(mode)
    _require(corr, "correlation matrix")
    labels = [str(c) for c in corr.columns]
    z = corr.to_numpy(dtype=float)
    hover = np.where(np.isnan(z), "undefined (zero variance)", np.char.mod("%.3f", np.nan_to_num(z)))
    neg, mid, pos = theme["diverging"]
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels, zmin=-1, zmax=1, zmid=0,
        colorscale=[[0.0, neg], [0.5, mid], [1.0, pos]],
        texttemplate="%{z:.2f}", textfont=dict(color=theme["ink"]), customdata=hover,
        hovertemplate="%{y} vs %{x}: %{customdata}<extra></extra>",
        colorbar=dict(title=dict(text="ρ"), tickvals=[-1, -0.5, 0, 0.5, 1]),
        xgap=2, ygap=2, hoverongaps=False,
    ))
    _layout(fig, theme, title, "", "", legend=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="bottom")
    fig.update_layout(yaxis_scaleanchor="x")
    return fig


# --------------------------------------------------------------------------- #
# Backtests
# --------------------------------------------------------------------------- #
def _benchmark_path(benchmark: pd.Series, results: Sequence[BacktestResult]) -> pd.Series:
    """Growth of 1 for the benchmark over exactly the backtest's dates (no filling)."""
    index = results[0].returns.index
    aligned = benchmark.reindex(index)
    if aligned.isna().any():
        raise ValueError("Benchmark is missing returns on some backtest dates; align the data first.")
    start = results[0].wealth.index[0]
    return pd.concat([pd.Series([1.0], index=[start]), (1 + aligned).cumprod()])


def equity_curves(results: Sequence[BacktestResult], *, benchmark: pd.Series | None = None,
                  mode: str = "light", log_scale: bool = False) -> go.Figure:
    """Growth of 1 invested at the first allocation date, net of transaction costs."""
    theme = _theme(mode)
    _require(results, "backtest results")
    if len(results) > MAX_CATEGORICAL - 1:
        raise ValueError("Too many strategies for one chart.")
    fig = go.Figure()
    for i, r in enumerate(results):
        fig.add_scatter(x=r.wealth.index, y=r.wealth, name=r.name, mode="lines",
                        line=dict(width=2, color=theme["series"][i]),
                        hovertemplate=f"{r.name}: " + "%{y:.3f}<extra></extra>")
    if benchmark is not None:
        path = _benchmark_path(benchmark, results)
        name = str(benchmark.name or "Benchmark")
        fig.add_scatter(x=path.index, y=path, name=name, mode="lines",
                        line=dict(width=2, dash="dot", color=theme["ink_secondary"]),
                        hovertemplate=f"{name}: " + "%{y:.3f}<extra></extra>")
    _layout(fig, theme, "Growth of 1 (net of costs)", "", "Portfolio value")
    fig.update_layout(hovermode="x unified")
    if log_scale:
        fig.update_yaxes(type="log")
    return fig


def drawdown_chart(results: Sequence[BacktestResult], *, benchmark: pd.Series | None = None,
                   mode: str = "light") -> go.Figure:
    """Running loss from the previous peak for each strategy (0 % = at a new high)."""
    theme = _theme(mode)
    _require(results, "backtest results")
    fig = go.Figure()
    for i, r in enumerate(results):
        dd = drawdown_series(r.returns)
        fig.add_scatter(x=dd.index, y=dd, name=r.name, mode="lines",
                        line=dict(width=2, color=theme["series"][i]),
                        hovertemplate=f"{r.name}: " + "%{y:.1%}<extra></extra>")
    if benchmark is not None:
        _benchmark_path(benchmark, results)  # raises if any backtest date is missing
        dd = drawdown_series(benchmark.reindex(results[0].returns.index))
        name = str(benchmark.name or "Benchmark")
        fig.add_scatter(x=dd.index, y=dd, name=name, mode="lines",
                        line=dict(width=2, dash="dot", color=theme["ink_secondary"]),
                        hovertemplate=f"{name}: " + "%{y:.1%}<extra></extra>")
    _layout(fig, theme, "Drawdown from previous peak", "", "Drawdown", y_pct=True)
    fig.update_layout(hovermode="x unified")
    return fig


# --------------------------------------------------------------------------- #
# Trend and technical analysis (v2.1, v3.0)
# --------------------------------------------------------------------------- #
def trend_chart(prices: pd.Series, smas: Mapping[str, pd.Series], *, events: pd.DataFrame | None = None,
                mode: str = "light", title: str | None = None) -> go.Figure:
    """Price with moving averages; golden crosses as up-triangles, death crosses as down-triangles."""
    theme = _theme(mode)
    _require(prices, "prices")
    fig = go.Figure()
    fig.add_scatter(x=prices.index, y=prices, name=str(prices.name or "Price"), mode="lines",
                    line=dict(width=1.5, color=theme["ink_secondary"]),
                    hovertemplate="Price %{y:,.2f}<extra></extra>")
    for i, (label, line) in enumerate(smas.items()):
        fig.add_scatter(x=line.index, y=line, name=label, mode="lines",
                        line=dict(width=2, color=theme["series"][i]),
                        hovertemplate=f"{label} " + "%{y:,.2f}<extra></extra>")
    if events is not None and len(events):
        for kind, symbol, color in (("golden cross", "triangle-up", theme["series"][2]),
                                    ("death cross", "triangle-down", theme["series"][7])):
            ev = events[events["event"] == kind]
            if len(ev):
                fig.add_scatter(x=ev["date"], y=ev["price"], name=kind.capitalize(), mode="markers",
                                marker=dict(size=12, symbol=symbol, color=color,
                                            line=dict(width=1, color="rgba(255,255,255,0.9)")),
                                hovertemplate=kind + " %{x|%d %b %Y}<extra></extra>")
    _layout(fig, theme, title or f"{prices.name}: price and moving averages", "", "Price")
    fig.update_layout(hovermode="x unified")
    return fig


def technical_chart(prices: pd.Series, *, bands: pd.DataFrame, macd_frame: pd.DataFrame, rsi_line: pd.Series,
                    ema_lines: Mapping[str, pd.Series] | None = None, rsi_levels: tuple[float, float] = (30, 70),
                    events: pd.DataFrame | None = None, mode: str = "light") -> go.Figure:
    """Three panels: price with Bollinger bands and EMAs; MACD with signal and histogram; RSI."""
    theme = _theme(mode)
    _require(prices, "prices")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.25, 0.20],
                        vertical_spacing=0.04)
    s = theme["series"]
    fig.add_scatter(x=bands.index, y=bands["upper"], name="Upper band", mode="lines",
                    line=dict(width=1, color=theme["muted"]), row=1, col=1,
                    hovertemplate="Upper %{y:,.2f}<extra></extra>")
    fig.add_scatter(x=bands.index, y=bands["lower"], name="Lower band", mode="lines", fill="tonexty",
                    fillcolor="rgba(137,135,129,0.12)", line=dict(width=1, color=theme["muted"]), row=1, col=1,
                    hovertemplate="Lower %{y:,.2f}<extra></extra>")
    fig.add_scatter(x=bands.index, y=bands["middle"], name="Middle (SMA)", mode="lines",
                    line=dict(width=1, dash="dot", color=theme["muted"]), row=1, col=1,
                    hovertemplate="SMA %{y:,.2f}<extra></extra>")
    fig.add_scatter(x=prices.index, y=prices, name=str(prices.name or "Price"), mode="lines",
                    line=dict(width=1.8, color=theme["ink_secondary"]), row=1, col=1,
                    hovertemplate="Price %{y:,.2f}<extra></extra>")
    for i, (label, line) in enumerate((ema_lines or {}).items()):
        fig.add_scatter(x=line.index, y=line, name=label, mode="lines", line=dict(width=1.6, color=s[3 + i]),
                        row=1, col=1, hovertemplate=f"{label} " + "%{y:,.2f}<extra></extra>")
    if events is not None and len(events):
        for sig, symbol, color in (("bullish", "triangle-up", s[2]), ("bearish", "triangle-down", s[7])):
            ev = events[(events["signal"] == sig) & (events["indicator"] != "MACD")]
            ev = ev[ev["date"].isin(prices.index)]
            if len(ev):
                fig.add_scatter(x=ev["date"], y=prices.loc[ev["date"]].to_numpy(), mode="markers",
                                name=f"{sig.capitalize()} RSI/band signal", text=ev["event"],
                                marker=dict(size=10, symbol=symbol, color=color), row=1, col=1,
                                hovertemplate="%{text}<extra></extra>")
    hist = macd_frame["histogram"]
    fig.add_bar(x=hist.index, y=hist, name="MACD histogram", row=2, col=1, showlegend=False,
                marker=dict(color=np.where(hist.fillna(0) >= 0, s[2], s[7]), line=dict(width=0)),
                hovertemplate="Histogram %{y:.3f}<extra></extra>")
    fig.add_scatter(x=macd_frame.index, y=macd_frame["macd"], name="MACD", mode="lines",
                    line=dict(width=1.6, color=s[0]), row=2, col=1, hovertemplate="MACD %{y:.3f}<extra></extra>")
    fig.add_scatter(x=macd_frame.index, y=macd_frame["signal"], name="Signal", mode="lines",
                    line=dict(width=1.6, color=s[1]), row=2, col=1, hovertemplate="Signal %{y:.3f}<extra></extra>")
    fig.add_scatter(x=rsi_line.index, y=rsi_line, name="RSI", mode="lines", line=dict(width=1.6, color=s[6]),
                    row=3, col=1, hovertemplate="RSI %{y:.1f}<extra></extra>")
    for level in rsi_levels:
        fig.add_hline(y=level, line=dict(width=1, dash="dash", color=theme["muted"]), row=3, col=1)
    _layout(fig, theme, f"{prices.name}: Bollinger bands, MACD and RSI", "", "")
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    # Many series: legend below the panels so it never collides with the title.
    fig.update_layout(hovermode="x unified", height=800, bargap=0, margin=dict(l=60, r=20, t=60, b=110),
                      legend=dict(orientation="h", yanchor="top", y=-0.06, x=0))
    return fig


def candlestick_chart(ohlc: pd.DataFrame, *, patterns: pd.DataFrame | None = None, mode: str = "light",
                      title: str = "Candlesticks") -> go.Figure:
    """OHLC candles (green = close above open) with detected patterns marked above/below the bar."""
    theme = _theme(mode)
    _require(ohlc, "OHLC data")
    up, down = theme["series"][2], theme["series"][7]
    fig = go.Figure(go.Candlestick(x=ohlc.index, open=ohlc["Open"], high=ohlc["High"], low=ohlc["Low"],
                                   close=ohlc["Close"], name="Price",
                                   increasing=dict(line=dict(color=up), fillcolor=up),
                                   decreasing=dict(line=dict(color=down), fillcolor=down)))
    if patterns is not None and len(patterns):
        for sig, symbol, color, col, shift in (("bullish", "triangle-up", up, "Low", 0.985),
                                               ("bearish", "triangle-down", down, "High", 1.015),
                                               ("neutral", "diamond", theme["muted"], "High", 1.015)):
            p = patterns[patterns["signal"] == sig]
            if len(p):
                y = ohlc.loc[p["date"], col].to_numpy() * shift
                fig.add_scatter(x=p["date"], y=y, mode="markers", name=f"{sig.capitalize()} pattern",
                                text=p["pattern"], marker=dict(size=10, symbol=symbol, color=color),
                                hovertemplate="%{text}<br>%{x|%d %b %Y}<extra></extra>")
    _layout(fig, theme, title, "", "Price")
    fig.update_layout(xaxis_rangeslider_visible=False)
    return fig


# --------------------------------------------------------------------------- #
# Valuation and performance measurement (v3.0)
# --------------------------------------------------------------------------- #
def dcf_chart(schedule: pd.DataFrame, pv_terminal: float, *, mode: str = "light",
              title: str = "Where the intrinsic value comes from") -> go.Figure:
    """Present value of each explicit year's cash flow plus the present value of the terminal value."""
    theme = _theme(mode)
    labels = [f"Year {t}" for t in schedule.index] + ["Terminal value"]
    values = list(schedule["present_value"].astype(float)) + [pv_terminal]
    colors = [theme["series"][0]] * len(schedule) + [theme["series"][1]]
    fig = go.Figure(go.Bar(x=labels, y=values, marker=dict(color=colors, line=dict(width=0)),
                           text=[f"{v:,.2f}" for v in values], textposition="outside",
                           hovertemplate="%{x}: PV %{y:,.2f}<extra></extra>"))
    fig.update_layout(barcornerradius=4)
    return _layout(fig, theme, title, "", "Present value per share", legend=False)


def security_market_line(table: pd.DataFrame, *, rf: float, market_return: float, mode: str = "light",
                         title: str = "Return vs beta (security market line)") -> go.Figure:
    """Portfolios by beta and average return; the SML from rf through the market. Vertical
    distance from the line is Jensen's alpha."""
    theme = _theme(mode)
    _require(table, "performance table")
    x_max = float(max(table["beta"].max(), 1.0) * 1.15)
    fig = go.Figure()
    fig.add_scatter(x=[0, x_max], y=[rf, rf + (market_return - rf) * x_max], mode="lines", name="SML",
                    line=dict(width=1.5, dash="dash", color=theme["muted"]),
                    hovertemplate="SML<br>beta %{x:.2f}<br>return %{y:.2%}<extra></extra>")
    fig.add_scatter(x=table["beta"], y=table["return"], mode="markers+text", name="Portfolios",
                    text=[str(i) for i in table.index], textposition="top center",
                    textfont=dict(color=theme["ink_secondary"]),
                    marker=dict(size=11, color=theme["series"][0], line=dict(width=2, color="rgba(255,255,255,0.9)")),
                    customdata=table["jensen_alpha"],
                    hovertemplate="%{text}<br>beta %{x:.2f}<br>return %{y:.2%}<br>alpha %{customdata:.2%}<extra></extra>")
    _layout(fig, theme, title, "Beta", "Average annual return", y_pct=True, legend=False)
    return fig


def timing_scatter(fund_excess: pd.Series, market_excess: pd.Series, *, alpha: float, beta: float, gamma: float,
                   model: str, mode: str = "light") -> go.Figure:
    """Fund vs market excess returns with the fitted Treynor-Mazuy curve or Henriksson-Merton kink."""
    theme = _theme(mode)
    _require(fund_excess, "returns")
    fig = go.Figure()
    fig.add_scatter(x=market_excess, y=fund_excess, mode="markers", name="Periods",
                    marker=dict(size=5, color=theme["muted"], opacity=0.6),
                    hovertemplate="Market %{x:.2%}<br>Fund %{y:.2%}<extra></extra>")
    grid = np.linspace(float(market_excess.min()), float(market_excess.max()), 200)
    extra = grid**2 if model == "treynor_mazuy" else grid * (grid > 0)
    fig.add_scatter(x=grid, y=alpha + beta * grid + gamma * extra, mode="lines", name="Fitted",
                    line=dict(width=2.5, color=theme["series"][0]),
                    hovertemplate="Fitted %{y:.2%}<extra></extra>")
    label = "Treynor-Mazuy" if model == "treynor_mazuy" else "Henriksson-Merton"
    _layout(fig, theme, f"Market timing ({label})", "Market excess return", "Fund excess return",
            x_pct=True, y_pct=True)
    return fig
