"""CAPM: beta, alpha and R^2 from a regression on excess returns, plus CAPM expected returns.

Model (daily, on dates where BOTH the asset and the benchmark have a return):

    r_i,t - rf_d  =  alpha_i  +  beta_i * (r_m,t - rf_d)  +  e_t

Ordinary least squares with one regressor has a closed form, so we do not need
a statistics library:

    x_t = r_m,t - rf_d            (benchmark excess return)
    y_t = r_i,t - rf_d            (asset excess return)
    beta   = sum((x - x_bar)(y - y_bar)) / sum((x - x_bar)^2)  =  Cov(x, y) / Var(x)
    alpha  = y_bar - beta * x_bar                (daily);  alpha_annual = N * alpha
    R^2    = 1 - SSR / SST                       (share of variance explained by the market)
    SE(beta) = sqrt( SSR / (T - 2) / sum((x - x_bar)^2) )

CAPM expected return (arithmetic, annual — the same units as the optimiser's mu):

    E[R_i] = rf_arith + beta_i * MRP,   rf_arith = N * rf_d

where MRP (market risk premium, E[R_m] - rf) is either the benchmark's historical
annualised excess return or a number you supply.

Edge cases
----------
* Benchmark with zero variance -> beta is undefined for *every* asset, so we raise
  DataError rather than returning a table full of meaningless numbers.
* Asset with zero variance -> beta = 0 (it doesn't move with the market) and R^2 = NaN
  (0 / 0: there is no variance to explain).
* Missing returns are never filled; misaligned dates are dropped and counted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DEFAULT_MIN_OBSERVATIONS, TRADING_DAYS_PER_YEAR, VOLATILITY_EPSILON
from .exceptions import DataError, InsufficientDataError
from .metrics import (
    annual_arithmetic_risk_free,
    annualized_mean_return,
    check_returns,
    daily_risk_free,
)

PREMIUM_HISTORICAL = "historical"
PREMIUM_USER = "user"


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AlignedReturns:
    """Asset and benchmark returns restricted to the dates they have in common."""

    assets: pd.DataFrame
    benchmark: pd.Series
    unmatched_asset_dates: int
    unmatched_benchmark_dates: int


@dataclass(frozen=True)
class CAPMResult:
    """Per-asset regression statistics and CAPM expected returns."""

    table: pd.DataFrame
    risk_free_rate: float          # annual effective rate as entered by the user
    market_premium: float          # arithmetic annual MRP actually used
    premium_source: str            # "historical" or "user"
    n_observations: int
    start: pd.Timestamp
    end: pd.Timestamp
    notes: tuple[str, ...]


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
def align_returns(
    asset_returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series | pd.DataFrame,
    *,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
) -> AlignedReturns:
    """Keep only the dates present in BOTH inputs (exact date match, nothing filled).

    Best practice: fetch the benchmark together with the assets so the *prices* are
    aligned before returns are computed (data.fetch_prices does this). If returns were
    computed on two different calendars (e.g. NSE vs NYSE holidays), a return on a
    shared date can span different numbers of days for the two series; the count of
    unmatched dates is reported so the UI can warn about it.
    """
    assets = asset_returns.to_frame() if isinstance(asset_returns, pd.Series) else asset_returns
    if not isinstance(assets, pd.DataFrame) or assets.shape[1] == 0:
        raise TypeError("asset_returns must be a Series or a DataFrame with at least one column.")
    benchmark = _as_series(benchmark_returns)

    for name, obj in (("Asset", assets), ("Benchmark", benchmark)):
        if obj.index.has_duplicates:
            raise DataError(f"{name} returns contain duplicate dates.")
        check_returns(obj, min_obs=1)  # rejects NaN / inf / < -100 %, never fills

    common = assets.index.intersection(benchmark.index).sort_values()
    if len(common) < min_observations:
        raise InsufficientDataError(
            f"Only {len(common)} date(s) shared by the assets and the benchmark; "
            f"at least {min_observations} are required for a meaningful beta."
        )
    return AlignedReturns(
        assets=assets.loc[common],
        benchmark=benchmark.loc[common],
        unmatched_asset_dates=len(assets.index.difference(common)),
        unmatched_benchmark_dates=len(benchmark.index.difference(common)),
    )


def _as_series(benchmark: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(benchmark, pd.DataFrame):
        if benchmark.shape[1] != 1:
            raise TypeError("The benchmark must be a single return series.")
        benchmark = benchmark.iloc[:, 0]
    if not isinstance(benchmark, pd.Series):
        raise TypeError("benchmark_returns must be a pandas Series.")
    return benchmark


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #
def estimate_beta(
    asset_returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series | pd.DataFrame,
    rf_annual: float = 0.0,
    *,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
) -> pd.DataFrame:
    """OLS of asset excess returns on benchmark excess returns, one row per asset.

    Columns: beta, alpha_daily, alpha_annual, r_squared, beta_std_error,
    residual_volatility (annualised idiosyncratic risk), n_observations.
    """
    if min_observations < 3:
        raise ValueError("min_observations must be at least 3 (the regression uses T - 2).")
    aligned = align_returns(asset_returns, benchmark_returns, min_observations=min_observations)
    rf_d = daily_risk_free(rf_annual)
    x = aligned.benchmark - rf_d                  # benchmark excess return
    y = aligned.assets - rf_d                     # asset excess returns (one column each)
    n_obs = len(x)

    x_dev = x - x.mean()
    sxx = float((x_dev**2).sum())                 # sum of squared deviations of x
    if math.sqrt(sxx / (n_obs - 1)) <= VOLATILITY_EPSILON:
        raise DataError(
            "Benchmark returns have zero variance over this period, so beta is undefined "
            "for every asset. Choose a different benchmark or date range."
        )

    y_dev = y - y.mean()
    beta = y_dev.mul(x_dev, axis=0).sum() / sxx
    alpha_daily = y.mean() - beta * x.mean()

    fitted_dev = pd.DataFrame(np.outer(x_dev, beta), index=y.index, columns=y.columns)
    ssr = ((y_dev - fitted_dev) ** 2).sum()       # residual sum of squares
    sst = (y_dev**2).sum()                        # total sum of squares
    sst_defined = sst.where(np.sqrt(sst / (n_obs - 1)) > VOLATILITY_EPSILON)
    r_squared = (1.0 - ssr / sst_defined).clip(0.0, 1.0)  # NaN for a constant asset

    residual_variance = ssr / (n_obs - 2)         # unbiased: two parameters estimated
    return pd.DataFrame(
        {
            "beta": beta,
            "alpha_daily": alpha_daily,
            "alpha_annual": alpha_daily * TRADING_DAYS_PER_YEAR,
            "r_squared": r_squared,
            "beta_std_error": np.sqrt(residual_variance / sxx),
            "residual_volatility": np.sqrt(residual_variance * TRADING_DAYS_PER_YEAR),
            "n_observations": n_obs,
        }
    )


# --------------------------------------------------------------------------- #
# Expected returns
# --------------------------------------------------------------------------- #
def historical_market_premium(benchmark_returns: pd.Series, rf_annual: float = 0.0) -> float:
    """Arithmetic annual excess return of the benchmark: N * mean(r_m - rf_d)."""
    benchmark = _as_series(benchmark_returns)
    check_returns(benchmark, min_obs=1)
    return float((benchmark - daily_risk_free(rf_annual)).mean() * TRADING_DAYS_PER_YEAR)


def capm_expected_returns(betas: pd.Series, rf_annual: float, market_premium: float) -> pd.Series:
    """E[R_i] = rf_arith + beta_i * MRP  (arithmetic annual returns)."""
    if not math.isfinite(market_premium):
        raise ValueError(f"Market risk premium must be a finite number, got {market_premium}.")
    if betas.isna().any():
        raise DataError("Cannot compute CAPM expected returns from missing betas.")
    expected = annual_arithmetic_risk_free(rf_annual) + betas * market_premium
    return expected.rename("capm_expected_return")


# --------------------------------------------------------------------------- #
# One-call analysis used by the dashboard
# --------------------------------------------------------------------------- #
def capm_analysis(
    asset_returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series | pd.DataFrame,
    rf_annual: float = 0.0,
    market_premium: float | None = None,
    *,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
) -> CAPMResult:
    """Betas, alphas, R^2 and CAPM expected returns on aligned dates.

    market_premium=None uses the benchmark's historical premium over the same dates;
    pass a number (e.g. 0.06) to use a forward-looking assumption instead.
    """
    aligned = align_returns(asset_returns, benchmark_returns, min_observations=min_observations)
    table = estimate_beta(aligned.assets, aligned.benchmark, rf_annual, min_observations=min_observations)

    if market_premium is None:
        premium = historical_market_premium(aligned.benchmark, rf_annual)
        source = PREMIUM_HISTORICAL
    else:
        premium = float(market_premium)
        source = PREMIUM_USER

    table["historical_return"] = annualized_mean_return(aligned.assets)
    table["capm_expected_return"] = capm_expected_returns(table["beta"], rf_annual, premium)

    notes: list[str] = []
    if aligned.unmatched_asset_dates or aligned.unmatched_benchmark_dates:
        notes.append(
            f"Dates were not identical: {aligned.unmatched_asset_dates} asset date(s) and "
            f"{aligned.unmatched_benchmark_dates} benchmark date(s) had no match and were "
            "excluded. If the series come from different exchange calendars, align prices "
            "before computing returns."
        )
    if source == PREMIUM_HISTORICAL and premium < 0:
        notes.append(
            f"The benchmark earned less than the risk-free rate over this period (premium "
            f"{premium:.2%}), so CAPM predicts LOWER returns for higher-beta assets. "
            "Consider entering a forward-looking market risk premium instead."
        )
    low_fit = table.index[table["r_squared"] < 0.1].tolist()
    if low_fit:
        notes.append(
            f"Low R^2 (< 0.10) for {', '.join(map(str, low_fit))}: the benchmark explains "
            "little of their movement, so their betas are weak descriptions of risk."
        )

    return CAPMResult(
        table=table,
        risk_free_rate=rf_annual,
        market_premium=premium,
        premium_source=source,
        n_observations=len(aligned.benchmark),
        start=aligned.benchmark.index[0],
        end=aligned.benchmark.index[-1],
        notes=tuple(notes),
    )
