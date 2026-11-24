"""Robust inputs for optimisation: Ledoit-Wolf covariance and Black-Litterman returns.

Why this module exists
----------------------
Mean-variance optimisation is extremely sensitive to its inputs. The sample
covariance matrix is noisy (and singular when there are more assets than days),
and historical mean returns are noisier still. This module provides two standard
mitigations; see docs/ARCHITECTURE.md, Section 4.7.

Ledoit-Wolf shrinkage
    Sigma_LW = (1 - delta) S + delta F,   F = m I,  m = tr(S) / n
    delta = min(1, phi / (T gamma))  with
    phi   = sum_ij [ (1/T) sum_t x_ti^2 x_tj^2 - S_ij^2 ],   gamma = ||S - F||_F^2
    computed on demeaned daily returns with S = X'X / T, then rescaled to the
    unbiased (T - 1) and annual conventions used everywhere else.

Black-Litterman
    Pi      = delta Sigma w_mkt                               (equilibrium EXCESS returns)
    Omega_k = (1/c_k - 1) (P tau Sigma P')_kk                 (view uncertainty, c in (0, 1])
    mu_BL   = Pi + tau Sigma P' (P tau Sigma P' + Omega)^-1 (Q - P Pi)
    Sig_BL  = Sigma + tau Sigma - tau Sigma P' (P tau Sigma P' + Omega)^-1 P tau Sigma
    Views are entered as TOTAL returns and converted to excess with
    Q - rf_arith * sum_j P_kj, so a relative view (row sums to 0) ignores rf.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import TRADING_DAYS_PER_YEAR, VOLATILITY_EPSILON
from .exceptions import EstimationError
from .metrics import (
    annual_arithmetic_risk_free,
    annualized_covariance,
    annualized_mean_return,
    check_returns,
    daily_risk_free,
)

COV_SAMPLE = "sample"
COV_LEDOIT_WOLF = "ledoit_wolf"
COVARIANCE_METHODS = (COV_SAMPLE, COV_LEDOIT_WOLF)

DEFAULT_TAU = 0.05


# --------------------------------------------------------------------------- #
# Ledoit-Wolf
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LedoitWolfResult:
    covariance: pd.DataFrame      # annual, unbiased scale (same units as annualized_covariance)
    shrinkage: float              # delta in [0, 1]
    target_variance: float        # m = average sample variance (annual), the target's diagonal


def ledoit_wolf(returns: pd.DataFrame) -> LedoitWolfResult:
    """Ledoit-Wolf (2004) shrinkage towards a scaled identity matrix."""
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("Covariance needs a DataFrame with one column per asset.")
    check_returns(returns, min_obs=2)
    x = returns.to_numpy(dtype=float)
    n_obs, n_assets = x.shape
    x = x - x.mean(axis=0)
    s = x.T @ x / n_obs                           # biased (MLE) sample covariance
    m = float(np.trace(s)) / n_assets
    target = m * np.eye(n_assets)
    gamma = float(((s - target) ** 2).sum())
    x2 = x**2
    phi = float((x2.T @ x2 / n_obs - s**2).sum())
    if n_assets == 1 or gamma <= 0.0:
        delta = 0.0                               # nothing to shrink (already the target)
    else:
        delta = min(1.0, max(0.0, phi / (n_obs * gamma)))
    shrunk = (1.0 - delta) * s + delta * target
    scale = TRADING_DAYS_PER_YEAR * n_obs / (n_obs - 1)   # MLE -> unbiased -> annual
    cov = pd.DataFrame(shrunk * scale, index=returns.columns, columns=returns.columns)
    return LedoitWolfResult(cov, delta, m * scale)


def covariance(returns: pd.DataFrame, method: str = COV_SAMPLE) -> pd.DataFrame:
    """Annual covariance by the named method ('sample' or 'ledoit_wolf')."""
    if method == COV_SAMPLE:
        return annualized_covariance(returns)
    if method == COV_LEDOIT_WOLF:
        return ledoit_wolf(returns).covariance
    raise ValueError(f"Unknown covariance method {method!r}; use one of {COVARIANCE_METHODS}.")


def condition_number(cov: pd.DataFrame) -> float:
    """Largest / smallest eigenvalue. inf for a singular matrix (it cannot be inverted safely)."""
    eig = np.linalg.eigvalsh(np.asarray(cov, dtype=float))
    largest, smallest = float(eig.max()), float(eig.min())
    if largest <= 0 or smallest <= 1e-12 * largest:
        return math.inf
    return largest / smallest


# --------------------------------------------------------------------------- #
# Black-Litterman
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class View:
    """One investor view: sum_j weights_j * E[R_j] = value (TOTAL annual return).

    confidence c in (0, 1]: 1 = certain (matched exactly), 0.5 = as uncertain as the
    prior (He-Litterman Omega), smaller = weaker.
    """

    weights: pd.Series
    value: float
    confidence: float = 0.5
    label: str = ""


def absolute_view(asset: str, value: float, confidence: float = 0.5) -> View:
    """'Asset will return `value` per year.'"""
    return View(pd.Series({asset: 1.0}), float(value), confidence, f"{asset} returns {value:.2%}")


def relative_view(outperformer: str, underperformer: str, value: float, confidence: float = 0.5) -> View:
    """'`outperformer` will beat `underperformer` by `value` per year.'"""
    if outperformer == underperformer:
        raise EstimationError("A relative view needs two different assets.")
    return View(pd.Series({outperformer: 1.0, underperformer: -1.0}), float(value), confidence,
                f"{outperformer} beats {underperformer} by {value:.2%}")


@dataclass(frozen=True)
class BlackLittermanResult:
    prior: pd.Series              # equilibrium TOTAL returns (Pi + rf), arithmetic annual
    posterior: pd.Series          # blended TOTAL returns
    posterior_cov: pd.DataFrame   # Sigma_BL
    risk_aversion: float
    tau: float
    views: pd.DataFrame           # one row per view: label, value, confidence, prior_implied, posterior_implied


def implied_returns(cov: pd.DataFrame, market_weights: pd.Series, risk_aversion: float) -> pd.Series:
    """Equilibrium excess returns Pi = delta Sigma w_mkt (reverse optimisation)."""
    w = _market_weights(market_weights, list(cov.index))
    if not math.isfinite(risk_aversion) or risk_aversion <= 0:
        raise EstimationError(f"Risk aversion must be positive, got {risk_aversion}.")
    return pd.Series(risk_aversion * cov.to_numpy(dtype=float) @ w, index=cov.index)


def implied_risk_aversion(benchmark_returns: pd.Series, rf_annual: float = 0.0) -> float:
    """delta = (E[R_m] - rf) / sigma_m^2 from the benchmark's history (arithmetic annual)."""
    check_returns(benchmark_returns, min_obs=2)
    excess = benchmark_returns - daily_risk_free(rf_annual)
    premium = float(excess.mean()) * TRADING_DAYS_PER_YEAR
    variance = float(benchmark_returns.var(ddof=1)) * TRADING_DAYS_PER_YEAR
    if variance <= (VOLATILITY_EPSILON**2) * TRADING_DAYS_PER_YEAR:
        raise EstimationError("Benchmark has zero variance; implied risk aversion is undefined.")
    if premium <= 0:
        raise EstimationError(
            f"The benchmark's historical premium over the risk-free rate is {premium:.2%} (not positive), "
            "so the implied risk aversion would be negative. Enter a risk aversion manually."
        )
    return premium / variance


def black_litterman(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    *,
    risk_aversion: float,
    rf_annual: float = 0.0,
    views: Sequence[View] = (),
    tau: float = DEFAULT_TAU,
) -> BlackLittermanResult:
    """Blend equilibrium returns with investor views (He-Litterman formulation)."""
    assets = [str(a) for a in cov.index]
    sigma = cov.to_numpy(dtype=float)
    if not np.isfinite(sigma).all():
        raise EstimationError("Covariance matrix contains NaN or infinite values.")
    if not math.isfinite(tau) or tau <= 0:
        raise EstimationError(f"tau must be positive, got {tau}.")
    pi = implied_returns(cov, market_weights, risk_aversion).to_numpy()
    rf_arith = annual_arithmetic_risk_free(rf_annual)

    if not views:
        prior = pd.Series(pi + rf_arith, index=cov.index)
        return BlackLittermanResult(prior, prior.copy(), cov * (1.0 + tau), risk_aversion, tau,
                                    pd.DataFrame(columns=["label", "value", "confidence",
                                                          "prior_implied", "posterior_implied"]))

    p = np.vstack([_view_row(v, assets) for v in views])
    confidence = np.array([v.confidence for v in views], dtype=float)
    if not np.all((confidence > 0) & (confidence <= 1)) or not np.isfinite(confidence).all():
        raise EstimationError("Every view confidence must be in (0, 1].")
    if np.any(np.abs(p).sum(axis=1) == 0):
        raise EstimationError("A view has all-zero weights.")
    q_excess = np.array([v.value for v in views]) - rf_arith * p.sum(axis=1)

    ts = tau * sigma
    pts = p @ ts                                   # P tau Sigma
    view_var = pts @ p.T                           # P tau Sigma P'
    omega = np.diag((1.0 / confidence - 1.0) * np.diag(view_var))
    middle = view_var + omega
    if np.linalg.matrix_rank(middle) < len(views) or np.linalg.cond(middle) > 1e12:
        raise EstimationError(
            "The views are contradictory or duplicated (e.g. the same view twice at full confidence), "
            "so they cannot be combined. Remove one or lower its confidence."
        )
    gain = np.linalg.solve(middle, np.eye(len(views)))
    mu_excess = pi + ts @ p.T @ gain @ (q_excess - p @ pi)
    post_cov = sigma + ts - ts @ p.T @ gain @ pts
    post_cov = (post_cov + post_cov.T) / 2.0

    prior = pd.Series(pi + rf_arith, index=cov.index)
    posterior = pd.Series(mu_excess + rf_arith, index=cov.index)
    table = pd.DataFrame({
        "label": [v.label or f"View {k + 1}" for k, v in enumerate(views)],
        "value": [v.value for v in views],
        "confidence": confidence,
        "prior_implied": p @ prior.to_numpy(),
        "posterior_implied": p @ posterior.to_numpy(),
    })
    return BlackLittermanResult(prior, posterior,
                                pd.DataFrame(post_cov, index=cov.index, columns=cov.columns),
                                risk_aversion, tau, table)


def _market_weights(weights: pd.Series, assets: list[str]) -> np.ndarray:
    if not isinstance(weights, pd.Series):
        raise TypeError("market_weights must be a pandas Series.")
    if set(map(str, weights.index)) != set(assets) or weights.index.has_duplicates:
        raise EstimationError(
            f"Market weights {sorted(map(str, weights.index))} must cover exactly the assets {sorted(assets)}."
        )
    w = weights.rename(index=str).reindex(assets).to_numpy(dtype=float)
    if not np.isfinite(w).all() or (w < 0).any():
        raise EstimationError("Market weights must be finite and non-negative.")
    if abs(w.sum() - 1.0) > 1e-6:
        raise EstimationError(f"Market weights must sum to 100 % (got {w.sum():.2%}).")
    return w


def _view_row(view: View, assets: list[str]) -> np.ndarray:
    unknown = [a for a in view.weights.index if a not in assets]
    if unknown:
        raise EstimationError(f"View '{view.label}' refers to unknown asset(s): {', '.join(map(str, unknown))}.")
    if not math.isfinite(view.value):
        raise EstimationError(f"View '{view.label}' has a non-finite value.")
    # Assets not mentioned in a view get a structural 0 in its P row (not a data fill).
    return view.weights.reindex(assets).fillna(0.0).to_numpy(dtype=float)


def historical_mean(returns: pd.DataFrame) -> pd.Series:
    """Alias kept here so every expected-return estimator lives in one module."""
    return annualized_mean_return(returns)
