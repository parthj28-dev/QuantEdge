"""Return and risk statistics.

All functions are pure: pandas in, numbers/pandas out, no I/O, no Streamlit.
Each function accepts either a Series (one asset) or a DataFrame (one column per
asset) of *daily simple returns* and returns a float or a Series respectively.

Conventions (see docs/ARCHITECTURE.md, Section 3):
* simple returns r_t = P_t / P_{t-1} - 1
* annualise with N = TRADING_DAYS_PER_YEAR (252): mean x N, std x sqrt(N)
* risk-free input is an annual effective rate; daily rf = (1 + rf)^(1/N) - 1
* undefined statistics (zero volatility) are NaN — never 0 or infinity
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import TRADING_DAYS_PER_YEAR, VOLATILITY_EPSILON
from .exceptions import DataError, InsufficientDataError

SeriesOrFrame = pd.Series | pd.DataFrame
FloatOrSeries = float | pd.Series


# --------------------------------------------------------------------------- #
# Input checks
# --------------------------------------------------------------------------- #
def _check_prices(prices: SeriesOrFrame) -> None:
    if not isinstance(prices, (pd.Series, pd.DataFrame)):
        raise TypeError("prices must be a pandas Series or DataFrame.")
    if prices.isna().to_numpy().any():
        raise DataError("Prices contain missing values; validate them with data.validate_prices first.")
    if (prices.to_numpy() <= 0).any():
        raise DataError("Prices must be strictly positive to compute returns.")
    if len(prices) < 2:
        raise InsufficientDataError("At least two prices are needed to compute a return.")


def _check_returns(returns: SeriesOrFrame, min_obs: int = 2) -> None:
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("returns must be a pandas Series or DataFrame.")
    values = returns.to_numpy(dtype=float)
    if np.isnan(values).any():
        raise DataError("Returns contain missing values; they will not be filled automatically.")
    if np.isinf(values).any():
        raise DataError("Returns contain infinite values.")
    if (values < -1).any():
        raise DataError("A simple return below -100% is impossible; check the input data.")
    if len(returns) < min_obs:
        raise InsufficientDataError(
            f"At least {min_obs} return observations are required, got {len(returns)}."
        )


# Public name so other QuantEdge modules (capm, optimize, backtest) apply the same rules.
check_returns = _check_returns


def _safe_ratio(numerator: FloatOrSeries, denominator: FloatOrSeries) -> FloatOrSeries:
    """numerator / denominator, but NaN wherever the denominator is ~0."""
    if isinstance(denominator, pd.Series):
        return numerator / denominator.where(denominator.abs() > VOLATILITY_EPSILON)
    if abs(denominator) <= VOLATILITY_EPSILON:
        return math.nan
    return float(numerator / denominator)


# --------------------------------------------------------------------------- #
# Risk-free rate
# --------------------------------------------------------------------------- #
def daily_risk_free(rf_annual: float, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Convert an annual effective rate to the equivalent daily rate.

    (1 + rf_daily)^N = 1 + rf_annual  ->  rf_daily = (1 + rf_annual)^(1/N) - 1
    """
    if not math.isfinite(rf_annual) or rf_annual <= -1:
        raise ValueError(f"Risk-free rate must be a finite number above -100%, got {rf_annual}.")
    return (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0


def annual_arithmetic_risk_free(rf_annual: float) -> float:
    """The risk-free rate in the same units as an arithmetic annual return (N x daily rate).

    Expected returns used by CAPM and the optimiser are arithmetic (mean daily x N),
    so the risk-free rate they are compared with must be too. For rf = 6.5 % this is
    about 6.30 %: the gap is exactly the compounding effect, not an error.
    """
    return TRADING_DAYS_PER_YEAR * daily_risk_free(rf_annual)


# --------------------------------------------------------------------------- #
# Returns
# --------------------------------------------------------------------------- #
def simple_returns(prices: SeriesOrFrame) -> SeriesOrFrame:
    """Daily simple returns r_t = P_t / P_{t-1} - 1 (first date has no return and is removed)."""
    _check_prices(prices)
    returns = prices / prices.shift(1) - 1.0
    return returns.iloc[1:]


def annualized_mean_return(returns: SeriesOrFrame) -> FloatOrSeries:
    """Arithmetic annual return: mean daily return x N. This is the optimiser's mu."""
    _check_returns(returns, min_obs=1)
    return returns.mean() * TRADING_DAYS_PER_YEAR


def cagr(returns: SeriesOrFrame) -> FloatOrSeries:
    """Compound annual growth rate: (prod(1 + r))^(N / T) - 1. What an investor experienced."""
    _check_returns(returns, min_obs=1)
    growth = (1.0 + returns).prod()
    return growth ** (TRADING_DAYS_PER_YEAR / len(returns)) - 1.0


def annualized_volatility(returns: SeriesOrFrame) -> FloatOrSeries:
    """Sample standard deviation of daily returns (ddof=1) x sqrt(N)."""
    _check_returns(returns, min_obs=2)
    return returns.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(returns: SeriesOrFrame, rf_annual: float = 0.0) -> FloatOrSeries:
    """Annualised Sharpe ratio from daily excess returns.

    S = sqrt(N) * mean(r - rf_d) / std(r - rf_d). NaN when volatility is ~0,
    because a ratio with zero risk in the denominator is undefined.
    """
    _check_returns(returns, min_obs=2)
    excess = returns - daily_risk_free(rf_annual)
    return _safe_ratio(excess.mean() * math.sqrt(TRADING_DAYS_PER_YEAR), excess.std(ddof=1))


# --------------------------------------------------------------------------- #
# Drawdowns
# --------------------------------------------------------------------------- #
def wealth_index(returns: SeriesOrFrame, initial: float = 1.0) -> SeriesOrFrame:
    """Value of `initial` invested at the start: W_t = initial * prod_{k<=t}(1 + r_k)."""
    _check_returns(returns, min_obs=1)
    return initial * (1.0 + returns).cumprod()


def drawdown_series(returns: SeriesOrFrame) -> SeriesOrFrame:
    """D_t = W_t / max(W_0, ..., W_t) - 1, with the starting capital W_0 = 1 included.

    Including W_0 matters: a loss on the very first day is a drawdown from the
    money you started with. `clip(lower=1.0)` does exactly that.
    """
    wealth = wealth_index(returns, initial=1.0)
    running_peak = wealth.cummax().clip(lower=1.0)
    return wealth / running_peak - 1.0


def max_drawdown(returns: SeriesOrFrame) -> FloatOrSeries:
    """Largest peak-to-trough loss, as a negative number (0 means it never fell)."""
    return drawdown_series(returns).min()


# --------------------------------------------------------------------------- #
# Co-movement
# --------------------------------------------------------------------------- #
def zero_volatility_assets(returns: pd.DataFrame) -> list[str]:
    """Columns whose daily returns are (numerically) constant."""
    _check_returns(returns, min_obs=2)
    std = returns.std(ddof=1)
    return [str(c) for c in std.index[std.abs() <= VOLATILITY_EPSILON]]


def annualized_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Sample covariance matrix of daily returns x N (the optimiser's Sigma)."""
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("Covariance needs a DataFrame with one column per asset.")
    _check_returns(returns, min_obs=2)
    return returns.cov(ddof=1) * TRADING_DAYS_PER_YEAR


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """rho_ij = cov_ij / (sigma_i * sigma_j). Rows/columns of zero-variance assets are NaN.

    Computed explicitly (rather than with DataFrame.corr) so the formula is visible
    and the zero-variance case is handled deliberately.
    """
    cov = annualized_covariance(returns)
    std = np.sqrt(np.diag(cov.to_numpy()))
    std = np.where(std > VOLATILITY_EPSILON * math.sqrt(TRADING_DAYS_PER_YEAR), std, np.nan)
    corr = cov.to_numpy() / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)  # remove tiny floating-point overshoot like 1.0000000002
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def performance_summary(returns: pd.DataFrame, rf_annual: float = 0.0) -> pd.DataFrame:
    """One row per asset with the headline statistics used in the dashboard."""
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    _check_returns(returns, min_obs=2)
    return pd.DataFrame(
        {
            "annual_return": annualized_mean_return(returns),
            "cagr": cagr(returns),
            "volatility": annualized_volatility(returns),
            "sharpe": sharpe_ratio(returns, rf_annual),
            "max_drawdown": max_drawdown(returns),
            "observations": returns.count(),
        }
    )
