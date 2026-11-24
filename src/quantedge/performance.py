"""Portfolio performance measurement (v3.0, SAPM lectures 30-32).

Composite measures (R = portfolio return, RFR = risk-free, M = market/benchmark)
    Sharpe        S = (R - RFR) / sigma                    reward per unit of TOTAL risk
    Treynor       T = (R - RFR) / beta                     reward per unit of SYSTEMATIC risk
    Jensen alpha  R_t - RFR_t = alpha + beta (M_t - RFR_t) + e_t   (average return above CAPM)
    Information   IR = (R - R_b) / sigma(R - R_b)          active return per unit of tracking error
    M-squared     M2 = RFR + S x sigma_M                   the portfolio levered to market risk
    Diversification: systematic variance beta^2 sigma_M^2, unsystematic sigma^2 - beta^2 sigma_M^2,
                     R^2 = systematic / total (1 = completely diversified)

Market timing (excess returns, regressions by OLS)
    Treynor-Mazuy        r_p - r_f = a + b1 (r_M - r_f) + b2 (r_M - r_f)^2 + e;  b2 > 0: timing ability
    Henriksson-Merton    r_p - r_f = a + b1 (r_M - r_f) + b2 (r_M - r_f) D + e,  D = 1 if r_M > r_f

Two ways in:
* `measures_from_summary` takes annual average return, beta and standard deviation per
  portfolio, exactly like the lecture problems.
* `performance_measures` estimates everything from return series (daily by default),
  annualised with the project's conventions (arithmetic mean x N, sigma x sqrt(N)).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .capm import align_returns
from .config import TRADING_DAYS_PER_YEAR, VOLATILITY_EPSILON
from .exceptions import DataError, EstimationError, InsufficientDataError
from .metrics import check_returns

TREYNOR_MAZUY = "treynor_mazuy"
HENRIKSSON_MERTON = "henriksson_merton"
_PERIODS = {"D": TRADING_DAYS_PER_YEAR, "W": 52, "M": 12, "Q": 4}


def _ratio(num: float, den: float) -> float:
    return num / den if abs(den) > 1e-15 else math.nan


def _rank(values: pd.Series) -> pd.Series:
    """1 = best (largest); NaN stays unranked."""
    return values.rank(ascending=False, method="min")


# --------------------------------------------------------------------------- #
# Lecture-style: from summary statistics
# --------------------------------------------------------------------------- #
def measures_from_summary(portfolios: pd.DataFrame, *, market_return: float, market_std: float | None = None,
                          risk_free: float, market_name: str = "Market") -> pd.DataFrame:
    """`portfolios` has columns return, beta and optionally std (annual decimals), one row each.

    Adds the market row (beta 1) and returns Sharpe, Treynor, Jensen alpha, M2, the variance
    split into systematic and unsystematic parts, and Sharpe / Treynor ranks.
    """
    need = {"return", "beta"}
    if not need <= set(portfolios.columns):
        raise EstimationError("Need columns 'return' and 'beta' (and 'std' for Sharpe / M2).")
    table = portfolios.astype(float).copy()
    if "std" not in table:
        table["std"] = np.nan
    if (table["std"] < 0).any():
        raise EstimationError("Standard deviations cannot be negative.")
    market = pd.DataFrame({"return": [market_return], "beta": [1.0],
                           "std": [np.nan if market_std is None else market_std]}, index=[market_name])
    table = pd.concat([table[["return", "beta", "std"]], market])
    excess = table["return"] - risk_free
    out = pd.DataFrame(index=table.index)
    out["return"] = table["return"]
    out["beta"] = table["beta"]
    out["std"] = table["std"]
    out["excess_return"] = excess
    out["sharpe"] = [_ratio(e, s) if pd.notna(s) else math.nan for e, s in zip(excess, table["std"])]
    out["treynor"] = [_ratio(e, b) for e, b in zip(excess, table["beta"])]
    out["jensen_alpha"] = table["return"] - (risk_free + table["beta"] * (market_return - risk_free))
    if market_std is not None:
        out["m_squared"] = risk_free + out["sharpe"] * market_std
        out["m_squared_vs_market"] = out["m_squared"] - market_return
        out["systematic_variance"] = table["beta"] ** 2 * market_std**2
        out["unsystematic_variance"] = table["std"] ** 2 - out["systematic_variance"]
        out["r_squared"] = out["systematic_variance"] / table["std"] ** 2
    out["sharpe_rank"] = _rank(out["sharpe"])
    out["treynor_rank"] = _rank(out["treynor"])
    return out


# --------------------------------------------------------------------------- #
# From return series
# --------------------------------------------------------------------------- #
def periodic_returns(returns: pd.Series | pd.DataFrame, freq: str) -> pd.Series | pd.DataFrame:
    """Compound daily simple returns into weekly ('W'), monthly ('M') or quarterly ('Q') returns.

    Every calendar period with data is compounded, including a partial first or last period;
    'D' returns the input unchanged."""
    if freq == "D":
        return returns
    if freq not in _PERIODS:
        raise ValueError(f"freq must be one of {list(_PERIODS)}.")
    check_returns(returns, min_obs=1)
    rule = {"W": "W-FRI", "M": "ME", "Q": "QE"}[freq]
    grouped = (1 + returns).resample(rule)
    out = grouped.prod() - 1
    counts = returns.resample(rule).count()
    keep = (counts > 0) if isinstance(counts, pd.Series) else (counts > 0).all(axis=1)
    return out[keep]


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Coefficients, standard errors, R^2 for y = X b + e (X includes the constant)."""
    n, k = x.shape
    if n <= k:
        raise InsufficientDataError(f"Regression needs more than {k} observations, got {n}.")
    xtx = x.T @ x
    if np.linalg.matrix_rank(xtx) < k:
        raise EstimationError("Regressors are collinear (e.g. the market never fell below the risk-free "
                              "rate), so the timing coefficient cannot be estimated.")
    coef = np.linalg.solve(xtx, x.T @ y)
    resid = y - x @ coef
    ssr = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    se = np.sqrt(np.diag(np.linalg.inv(xtx)) * ssr / (n - k))
    r2 = 1 - ssr / sst if sst > 0 else math.nan
    return coef, se, r2


def _aligned_excess(returns: pd.Series | pd.DataFrame, benchmark: pd.Series, rf_annual: float,
                    periods_per_year: int, min_observations: int) -> tuple[pd.DataFrame, pd.Series]:
    aligned = align_returns(returns, benchmark, min_observations=min_observations)
    rf_p = (1 + rf_annual) ** (1 / periods_per_year) - 1
    return aligned.assets - rf_p, aligned.benchmark - rf_p


def performance_measures(returns: pd.Series | pd.DataFrame, benchmark: pd.Series, *, rf_annual: float = 0.0,
                         periods_per_year: int = TRADING_DAYS_PER_YEAR,
                         min_observations: int = 24) -> pd.DataFrame:
    """Annualised Sharpe, Treynor, Jensen alpha (with t-statistic), information ratio, tracking
    error, M2 and the diversification split, one row per portfolio plus the benchmark.

    All returns are periodic simple returns over identical dates (no filling). Annual figures:
    mean x P, std x sqrt(P), with P = periods_per_year. The risk-free rate is an annual
    effective rate converted to the period.
    """
    if not math.isfinite(rf_annual) or rf_annual <= -1:
        raise ValueError("rf_annual must be a finite rate above -100 %.")
    ex, mx = _aligned_excess(returns, benchmark, rf_annual, periods_per_year, min_observations)
    P = periods_per_year
    n = len(mx)
    m_sd = float(mx.std(ddof=1))
    if m_sd <= VOLATILITY_EPSILON:
        raise DataError("The benchmark has zero variance over this period; beta is undefined.")
    bench_name = str(benchmark.name or "Benchmark")
    rf_arith = P * ((1 + rf_annual) ** (1 / P) - 1)
    mx_dev = mx - mx.mean()
    sxx = float((mx_dev**2).sum())
    m_vol = m_sd * math.sqrt(P)
    rows = {}
    for col in list(ex.columns) + [bench_name]:
        y = mx if col == bench_name else ex[col]
        mean_ex, sd = float(y.mean()), float(y.std(ddof=1))
        beta = float((y - y.mean()) @ mx_dev) / sxx
        alpha = mean_ex - beta * float(mx.mean())
        resid = (y - y.mean()) - beta * mx_dev
        ssr = float(resid @ resid)
        alpha_se = math.sqrt(ssr / (n - 2) * (1 / n + float(mx.mean()) ** 2 / sxx)) if n > 2 else math.nan
        active = y - mx
        te = float(active.std(ddof=1)) * math.sqrt(P)
        sharpe = _ratio(mean_ex * math.sqrt(P), sd) if sd > VOLATILITY_EPSILON else math.nan
        total_var = (sd**2) * P
        sys_var = beta**2 * m_vol**2
        rows[str(col)] = {
            "annual_return": (mean_ex * P) + rf_arith,
            "annual_excess_return": mean_ex * P,
            "volatility": sd * math.sqrt(P),
            "beta": beta,
            "sharpe": sharpe,
            "treynor": _ratio(mean_ex * P, beta),
            "jensen_alpha": alpha * P,
            "alpha_t_stat": _ratio(alpha, alpha_se) if col != bench_name else math.nan,
            "tracking_error": te,
            "information_ratio": _ratio(float(active.mean()) * P, te) if te > 1e-12 else math.nan,
            "m_squared": rf_arith + sharpe * m_vol if math.isfinite(sharpe) else math.nan,
            "systematic_variance": sys_var,
            "unsystematic_variance": max(total_var - sys_var, 0.0) if col != bench_name else 0.0,
            "r_squared": sys_var / total_var if total_var > 0 else math.nan,
            "observations": n,
        }
    table = pd.DataFrame(rows).T
    table["m_squared_vs_benchmark"] = table["m_squared"] - table.loc[bench_name, "annual_return"]
    table["sharpe_rank"] = _rank(table["sharpe"].astype(float))
    table["treynor_rank"] = _rank(table["treynor"].astype(float))
    return table.astype(float)


@dataclass(frozen=True)
class TimingResult:
    model: str
    table: pd.DataFrame           # one row per portfolio: alpha, beta, gamma (+ t-stats), r_squared, n
    up_periods: int               # periods with market excess return > 0
    down_periods: int


def market_timing(returns: pd.Series | pd.DataFrame, benchmark: pd.Series, *, rf_annual: float = 0.0,
                  model: str = TREYNOR_MAZUY, periods_per_year: int = TRADING_DAYS_PER_YEAR,
                  min_observations: int = 24) -> TimingResult:
    """Estimate the timing coefficient b2 (called gamma here). Alpha is per period.

    gamma > 0 (and significant) suggests the manager raised market exposure before up markets.
    """
    if model not in (TREYNOR_MAZUY, HENRIKSSON_MERTON):
        raise ValueError(f"model must be {TREYNOR_MAZUY!r} or {HENRIKSSON_MERTON!r}.")
    ex, mx = _aligned_excess(returns, benchmark, rf_annual, periods_per_year, min_observations)
    x1 = mx.to_numpy()
    x2 = x1**2 if model == TREYNOR_MAZUY else x1 * (x1 > 0)
    X = np.column_stack([np.ones(len(x1)), x1, x2])
    rows = {}
    for col in ex.columns:
        coef, se, r2 = _ols(ex[col].to_numpy(), X)
        t = [_ratio(c, s) for c, s in zip(coef, se)]
        rows[str(col)] = {"alpha": coef[0], "alpha_t": t[0], "beta": coef[1], "beta_t": t[1],
                          "gamma": coef[2], "gamma_t": t[2], "r_squared": r2, "observations": len(x1)}
        if model == HENRIKSSON_MERTON:
            rows[str(col)]["up_market_beta"] = coef[1] + coef[2]
    up = int((x1 > 0).sum())
    return TimingResult(model, pd.DataFrame(rows).T.astype(float), up, len(x1) - up)


def summary_inputs(stats: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    """Convenience for the dashboard: {'P': {'return': .15, 'beta': 1, 'std': .05}, ...} -> DataFrame."""
    return pd.DataFrame(stats).T
