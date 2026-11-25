"""Risk-based portfolios: they need a covariance matrix but NO expected returns.

Risk contribution (Euler decomposition of volatility):
    sigma_p = sum_i w_i (Sigma w)_i / sigma_p
    share_i = w_i (Sigma w)_i / (w' Sigma w)          (shares sum to 1)

Portfolios
    Inverse volatility     w_i proportional to 1 / sigma_i (closed form; capped weights fixed at the
                           cap and the remainder re-shared in proportion)
    Risk parity (ERC)      every share_i = 1 / n. Solved with Newton's method on the strictly convex
                           f(y) = 1/2 y' Sigma y - (1/n) sum ln y_i, then w = y / sum(y)
                           (Maillard, Roncalli & Teiletche 2010; Spinu 2013).
    Max diversification    maximise w' sigma / sqrt(w' Sigma w) = the max-Sharpe problem with
                           mu = sigma and rf = 0 (Choueifaty & Coignard 2008).

See docs/ARCHITECTURE.md, Section 4.8.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import TRADING_DAYS_PER_YEAR, VOLATILITY_EPSILON
from .exceptions import OptimizationError
from .optimize import Caps, OptimizationResult, _prepare, max_sharpe, portfolio_stats

INVERSE_VOLATILITY = "inverse_volatility"
RISK_PARITY = "risk_parity"
MAX_DIVERSIFICATION = "max_diversification"

_ZERO_VOL = VOLATILITY_EPSILON * math.sqrt(TRADING_DAYS_PER_YEAR)
_NEWTON_DECREMENT_TOL = 1e-8      # below this, take full Newton steps (f is flat to machine precision)
_SHARE_TOLERANCE = 1e-8


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.DataFrame:
    """Per asset: weight, absolute contribution to volatility and share of total risk."""
    w = weights.reindex(cov.index).to_numpy(dtype=float)
    if np.isnan(w).any():
        raise OptimizationError("Weights and covariance matrix refer to different assets.")
    sigma = cov.to_numpy(dtype=float)
    sigma_w = sigma @ w
    variance = float(w @ sigma_w)
    if variance <= _ZERO_VOL**2:
        raise OptimizationError("Portfolio volatility is zero, so risk contributions are undefined.")
    vol = math.sqrt(variance)
    return pd.DataFrame({"weight": w, "risk_contribution": w * sigma_w / vol,
                         "risk_share": w * sigma_w / variance}, index=cov.index)


def diversification_ratio(weights: pd.Series, cov: pd.DataFrame) -> float:
    """Weighted average volatility / portfolio volatility (>= 1 for long-only portfolios)."""
    w = weights.reindex(cov.index).to_numpy(dtype=float)
    sigma = cov.to_numpy(dtype=float)
    vol = math.sqrt(max(float(w @ sigma @ w), 0.0))
    if vol <= _ZERO_VOL:
        return math.nan
    return float(w @ np.sqrt(np.diag(sigma))) / vol


def _check_no_zero_vol(sigma: np.ndarray, assets: list[str], method: str) -> np.ndarray:
    vols = np.sqrt(np.clip(np.diag(sigma), 0.0, None))
    zero = [a for a, v in zip(assets, vols) if v <= _ZERO_VOL]
    if zero:
        raise OptimizationError(
            f"{method} is undefined with zero-volatility asset(s) ({', '.join(zero)}); remove them."
        )
    return vols


def _result(w: np.ndarray, cov: pd.DataFrame, mu: pd.Series | None, rf_annual: float,
            objective: str, iterations: int, notes: tuple[str, ...] = ()) -> OptimizationResult:
    weights = pd.Series(w, index=cov.index, name="weight")
    stats = portfolio_stats(weights, mu, cov, rf_annual)
    return OptimizationResult(weights, stats.expected_return, stats.volatility, stats.sharpe,
                              objective, iterations, notes)


# --------------------------------------------------------------------------- #
# Inverse volatility
# --------------------------------------------------------------------------- #
def _capped_proportional(raw: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """Weights proportional to `raw`; any weight above its cap is fixed at the cap and the
    remaining budget is re-shared among the others in the same proportions (water-filling)."""
    w = np.zeros_like(raw)
    free = np.ones(len(raw), dtype=bool)
    budget = 1.0
    for _ in range(len(raw)):
        w[free] = budget * raw[free] / raw[free].sum()
        over = free & (w > caps + 1e-15)
        if not over.any():
            break
        w[over] = caps[over]
        budget -= caps[over].sum()
        free &= ~over
    return w


def inverse_volatility(cov: pd.DataFrame, *, max_weight: Caps = 1.0, mu: pd.Series | None = None,
                       rf_annual: float = 0.0) -> OptimizationResult:
    p = _prepare(cov, mu, max_weight)
    vols = _check_no_zero_vol(p.cov, p.assets, "Inverse volatility")
    w = _capped_proportional(1.0 / vols, p.caps)
    return _result(w, cov.loc[p.assets, p.assets], mu, rf_annual, INVERSE_VOLATILITY, 0)


# --------------------------------------------------------------------------- #
# Risk parity
# --------------------------------------------------------------------------- #
def _erc_newton(sigma: np.ndarray, max_iter: int = 100) -> tuple[np.ndarray, int]:
    n = sigma.shape[0]
    vols = np.sqrt(np.diag(sigma))
    y = (1.0 / vols) / math.sqrt(float((1.0 / vols) @ sigma @ (1.0 / vols)) * n)  # good scale

    def f(v: np.ndarray) -> float:
        return 0.5 * float(v @ sigma @ v) - float(np.log(v).sum()) / n

    for it in range(1, max_iter + 1):
        grad = sigma @ y - 1.0 / (n * y)
        hess = sigma + np.diag(1.0 / (n * y**2))
        step = np.linalg.solve(hess, grad)
        decrement = float(grad @ step)            # lambda^2 = g' H^-1 g
        if decrement < 1e-30:
            return y, it
        t = 1.0
        while np.any(y - t * step <= 0):           # stay in the domain y > 0
            t *= 0.5
        if decrement > _NEWTON_DECREMENT_TOL:      # damped phase: Armijo backtracking on f
            f0 = f(y)
            while f(y - t * step) > f0 - 0.25 * t * decrement and t > 1e-12:
                t *= 0.5
        y = y - t * step
        if decrement < 1e-24:
            return y, it
    return y, max_iter


def risk_parity(cov: pd.DataFrame, *, max_weight: Caps = 1.0, mu: pd.Series | None = None,
                rf_annual: float = 0.0) -> OptimizationResult:
    """Equal-risk-contribution portfolio; if caps forbid it, the closest capped portfolio."""
    p = _prepare(cov, mu, max_weight)
    _check_no_zero_vol(p.cov, p.assets, "Risk parity")
    sigma, n = p.cov, len(p.assets)
    y, nit = _erc_newton(sigma)
    w = y / y.sum()
    shares = w * (sigma @ w) / float(w @ sigma @ w)
    if not np.allclose(shares, 1.0 / n, atol=1e-6):
        raise OptimizationError("Risk-parity Newton iteration did not converge.")
    notes: tuple[str, ...] = ()
    if (w > p.caps + 1e-10).any():
        w, extra = _capped_risk_parity(sigma, p.caps, w)
        nit += extra
        notes = ("The weight caps make exactly equal risk contributions impossible; these are the "
                 "capped weights whose risk shares are as close to equal as possible.",)
    return _result(w, cov.loc[p.assets, p.assets], mu, rf_annual, RISK_PARITY, nit, notes)


def _project_capped_simplex(x: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """Closest point to x (Euclidean) with sum(w) = 1 and 0 <= w <= caps.

    w = clip(x - t, 0, caps) is monotone in t, so t is found by bisection; the last rounding
    dust goes to assets strictly inside their bounds, so no weight ever exceeds its cap.
    """
    lo, hi = float(x.min() - caps.max() - 1.0), float(x.max() + 1.0)
    for _ in range(200):
        t = 0.5 * (lo + hi)
        if np.clip(x - t, 0.0, caps).sum() > 1.0:
            lo = t
        else:
            hi = t
    w = np.clip(x - 0.5 * (lo + hi), 0.0, caps)
    deficit = 1.0 - w.sum()
    room = (caps - w) if deficit > 0 else w
    if room.sum() > 0:
        w = np.clip(w + deficit * room / room.sum(), 0.0, caps)
    return w


def _capped_risk_parity(sigma: np.ndarray, caps: np.ndarray, start: np.ndarray) -> tuple[np.ndarray, int]:
    """Minimise the squared gaps between risk shares and 1/n under the caps.

    SLSQP can stop early on some platforms (e.g. 'positive directional derivative'), so its
    answer is never trusted as-is: each candidate is projected exactly onto the feasible set
    and the best projected candidate is kept.
    """
    n = len(caps)

    def objective(w: np.ndarray) -> float:
        var = float(w @ sigma @ w)
        return float((((w * (sigma @ w)) / var - 1.0 / n) ** 2).sum())

    starts = [_capped_proportional(start, caps), _capped_proportional(np.ones(n), caps)]
    best, best_score, iterations = None, np.inf, 0
    for x0 in starts:
        res = minimize(objective, x0, method="SLSQP", bounds=list(zip(np.zeros(n), caps)),
                       constraints=({"type": "eq", "fun": lambda w: w.sum() - 1.0},),
                       options={"ftol": 1e-15, "maxiter": 1000})
        iterations += int(res.nit)
        for cand in (np.asarray(res.x, dtype=float), x0):
            w = _project_capped_simplex(cand, caps)
            score = objective(w)
            if np.isfinite(score) and score < best_score:
                best, best_score = w, score
    if best is None:
        raise OptimizationError("Capped risk parity failed to find a feasible portfolio.")
    return best, iterations


# --------------------------------------------------------------------------- #
# Maximum diversification
# --------------------------------------------------------------------------- #
def max_diversification(cov: pd.DataFrame, *, max_weight: Caps = 1.0, mu: pd.Series | None = None,
                        rf_annual: float = 0.0) -> OptimizationResult:
    p = _prepare(cov, mu, max_weight)
    vols = _check_no_zero_vol(p.cov, p.assets, "Maximum diversification")
    square = cov.loc[p.assets, p.assets]
    solved = max_sharpe(pd.Series(vols, index=square.index), square, rf_annual=0.0, max_weight=max_weight)
    return _result(solved.weights.reindex(square.index).to_numpy(), square, mu, rf_annual,
                   MAX_DIVERSIFICATION, solved.iterations)
