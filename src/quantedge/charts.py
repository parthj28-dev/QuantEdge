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
def allocation_bar(weights: pd.DataFrame | pd.Series, *, mode: str = "light",
                   title: str = "Portfolio weights") -> go.Figure:
    """Grouped bars: assets on the x-axis, one colour per portfolio (columns of `weights`)."""
    theme = _theme(mode)
    frame = weights.to_frame() if isinstance(weights, pd.Series) else weights
    _require(frame, "weights")
    if frame.shape[1] > MAX_CATEGORICAL:
        raise ValueError(f"At most {MAX_CATEGORICAL} portfolios can be compared in one chart.")
    fig = go.Figure()
    for i, column in enumerate(frame.columns):
        fig.add_bar(
            x=[str(a) for a in frame.index], y=frame[column].to_numpy(), name=str(column),
            marker=dict(color=theme["series"][i], line=dict(width=0)),
            hovertemplate="%{x}<br>" + f"{column}: " + "%{y:.1%}<extra></extra>",
        )
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, barcornerradius=4)
    _layout(fig, theme, title, "", "Weight", y_pct=True, legend=frame.shape[1] > 1)
    fig.update_yaxes(rangemode="tozero")
    return fig


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
