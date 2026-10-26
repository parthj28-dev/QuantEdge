"""Long-only portfolio optimisation with weight caps (SciPy SLSQP).

Problems solved (w = weights, mu = annual expected returns, Sigma = annual covariance):

    Minimum variance:   minimise  w' Sigma w
    Maximum Sharpe:     maximise  (w' mu - rf) / sqrt(w' Sigma w)
    Target return:      minimise  w' Sigma w   subject to  w' mu = R   (used by the frontier)

    all subject to      sum(w) = 1,   0 <= w_i <= cap_i

Why SLSQP? It is SciPy's standard method for smooth objectives with both bound
constraints (0 <= w <= cap) and equality constraints (sum to 1). We give it exact
(analytic) gradients so it doesn't have to estimate them numerically.

Defensive design
----------------
1. Inputs are checked *before* solving: labels match, no NaN/inf, Sigma symmetric and
   positive semi-definite, and the caps are feasible (they must add up to >= 100 %).
2. Every solver output is *re-verified independently* (we don't trust res.success
   alone). Numerical dust such as -3e-14 is clipped; any real violation raises
   OptimizationError.
3. Max-Sharpe is not a convex problem, so it is solved from several starting points
   and the best verified solution is kept.

Units: mu and Sigma are arithmetic annual (metrics.annualized_mean_return /
annualized_covariance); the risk-free rate is converted to the same units with
metrics.annual_arithmetic_risk_free.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import TRADING_DAYS_PER_YEAR, VOLATILITY_EPSILON
from .exceptions import OptimizationError
from .metrics import annual_arithmetic_risk_free, annualized_covariance, annualized_mean_return

#: Maximum allowed constraint violation after solving (sum, bounds, target return).
WEIGHT_TOLERANCE = 1e-8
_SOLVER_OPTIONS = {"ftol": 1e-12, "maxiter": 1000}
# An annual volatility below this counts as zero (same threshold as metrics, annualised).
_ZERO_VOL = VOLATILITY_EPSILON * math.sqrt(TRADING_DAYS_PER_YEAR)

Caps = float | pd.Series | Mapping[str, float]

MIN_VARIANCE = "min_variance"
MAX_SHARPE = "max_sharpe"
TARGET_RETURN = "target_return"


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PortfolioStats:
    expected_return: float
    volatility: float
    sharpe: float


@dataclass(frozen=True)
class OptimizationResult:
    """Verified optimal weights plus the portfolio's ex-ante statistics."""

    weights: pd.Series
    expected_return: float     # w' mu (NaN if no mu was supplied)
    volatility: float          # sqrt(w' Sigma w)
    sharpe: float              # (w' mu - rf) / vol, NaN if vol ~ 0 or no mu
    objective: str
    iterations: int
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Problem:
    assets: list[str]
    mu: np.ndarray | None
    cov: np.ndarray
    caps: np.ndarray


# --------------------------------------------------------------------------- #
# Inputs and portfolio statistics
# --------------------------------------------------------------------------- #
def historical_inputs(returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """mu = mean daily return x N, Sigma = sample covariance x N."""
    return annualized_mean_return(returns), annualized_covariance(returns)


def portfolio_stats(
    weights: pd.Series, mu: pd.Series | None, cov: pd.DataFrame, rf_annual: float = 0.0
) -> PortfolioStats:
    """Ex-ante return, volatility and Sharpe ratio of a weight vector."""
    w = weights.reindex(cov.index).to_numpy(dtype=float)
    if np.isnan(w).any():
        raise OptimizationError("Weights and covariance matrix refer to different assets.")
    variance = max(float(w @ cov.to_numpy(dtype=float) @ w), 0.0)  # clip -1e-18 dust
    volatility = math.sqrt(variance)
    if mu is None:
        return PortfolioStats(math.nan, volatility, math.nan)
    expected = float(w @ mu.reindex(cov.index).to_numpy(dtype=float))
    excess = expected - annual_arithmetic_risk_free(rf_annual)
    sharpe = excess / volatility if volatility > _ZERO_VOL else math.nan
    return PortfolioStats(expected, volatility, sharpe)


def _prepare(cov: pd.DataFrame, mu: pd.Series | None, max_weight: Caps) -> _Problem:
    """Validate everything up front so the solver only ever sees a well-posed problem."""
    if not isinstance(cov, pd.DataFrame):
        raise TypeError("cov must be a pandas DataFrame.")
    if cov.shape[0] == 0 or cov.shape[0] != cov.shape[1] or not cov.index.equals(cov.columns):
        raise OptimizationError("Covariance matrix must be square with identical row and column labels.")
    assets = [str(a) for a in cov.index]
    sigma = cov.to_numpy(dtype=float)
    if not np.isfinite(sigma).all():
        raise OptimizationError("Covariance matrix contains NaN or infinite values.")
    scale = float(np.abs(sigma).max()) or 1.0
    if not np.allclose(sigma, sigma.T, rtol=0.0, atol=1e-10 * scale):
        raise OptimizationError("Covariance matrix is not symmetric.")
    sigma = (sigma + sigma.T) / 2.0
    smallest_eigenvalue = float(np.linalg.eigvalsh(sigma).min())
    if smallest_eigenvalue < -1e-10 * scale:
        raise OptimizationError(
            f"Covariance matrix is not positive semi-definite (smallest eigenvalue "
            f"{smallest_eigenvalue:.3g}); it cannot describe real portfolio variance."
        )

    mu_arr = None
    if mu is not None:
        if not isinstance(mu, pd.Series):
            raise TypeError("mu must be a pandas Series.")
        if set(map(str, mu.index)) != set(assets) or mu.index.has_duplicates:
            raise OptimizationError(
                f"Expected returns {sorted(map(str, mu.index))} and covariance {sorted(assets)} "
                "must cover exactly the same assets."
            )
        mu_arr = mu.rename(index=str).reindex(assets).to_numpy(dtype=float)
        if not np.isfinite(mu_arr).all():
            raise OptimizationError("Expected returns contain NaN or infinite values.")

    return _Problem(assets, mu_arr, sigma, _caps_array(max_weight, assets))


def _caps_array(max_weight: Caps, assets: list[str]) -> np.ndarray:
    if isinstance(max_weight, (pd.Series, Mapping)):
        caps_series = pd.Series(max_weight, dtype=float).rename(index=str)
        missing = [a for a in assets if a not in caps_series.index]
        if missing:
            raise OptimizationError(f"No maximum weight given for: {', '.join(missing)}.")
        caps = caps_series.reindex(assets).to_numpy(dtype=float)
        if not np.isfinite(caps).all() or (caps < 0).any() or (caps > 1).any():
            raise OptimizationError("Each maximum weight must be between 0 and 1.")
    else:
        cap = float(max_weight)
        if not math.isfinite(cap) or not 0.0 < cap <= 1.0:
            raise OptimizationError(f"Maximum weight must be in (0, 1], got {max_weight}.")
        caps = np.full(len(assets), cap)

    total = float(caps.sum())
    if total < 1.0 - 1e-12:
        raise OptimizationError(
            f"Infeasible constraints: with {len(assets)} asset(s) the maximum weights add up to "
            f"only {total:.1%}, so the weights cannot sum to 100%. Raise the cap to at least "
            f"{1 / len(assets):.1%} per asset or add more assets."
        )
    return caps


# --------------------------------------------------------------------------- #
# Exact helpers for the box-and-budget constraint set
# --------------------------------------------------------------------------- #
def _greedy(values: np.ndarray, caps: np.ndarray, *, descending: bool) -> np.ndarray:
    """Fill assets in order of `values` up to their caps until 100 % is allocated.

    For a linear objective (like w' mu) over {sum w = 1, 0 <= w <= cap} this greedy
    fill is the exact optimum — no solver needed.
    """
    order = np.argsort(-values if descending else values, kind="stable")
    w = np.zeros_like(caps)
    remaining = 1.0
    for i in order:
        take = min(caps[i], remaining)
        w[i] = take
        remaining -= take
        if remaining <= 1e-15:
            break
    return w


def _feasible_start(caps: np.ndarray) -> np.ndarray:
    """As close to equal weights as the caps allow ("water-filling")."""
    w = np.zeros_like(caps)
    remaining = 1.0
    active = caps > 0
    while remaining > 1e-15 and active.any():
        add = np.minimum(remaining / active.sum(), caps - w) * active
        w += add
        remaining -= add.sum()
        active = (caps - w) > 1e-15
    return w


def achievable_return_range(mu: pd.Series, *, max_weight: Caps = 1.0) -> tuple[float, float]:
    """Lowest and highest w' mu attainable under the long-only, capped constraints."""
    assets = [str(a) for a in mu.index]
    mu_arr = mu.to_numpy(dtype=float)
    caps = _caps_array(max_weight, assets)
    return (float(mu_arr @ _greedy(mu_arr, caps, descending=False)),
            float(mu_arr @ _greedy(mu_arr, caps, descending=True)))


# --------------------------------------------------------------------------- #
# Solver wrapper
# --------------------------------------------------------------------------- #
def _verify(raw: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """Independently check a solver's weights; clean numerical dust or raise."""
    if not np.isfinite(raw).all():
        raise OptimizationError("Solver returned non-finite weights.")
    worst_low = float((-raw).max())
    worst_high = float((raw - caps).max())
    budget_error = abs(float(raw.sum()) - 1.0)
    if max(worst_low, worst_high, budget_error) > WEIGHT_TOLERANCE:
        raise OptimizationError(
            f"Solver result violates constraints (sum error {budget_error:.2e}, "
            f"below-zero {max(worst_low, 0):.2e}, above-cap {max(worst_high, 0):.2e})."
        )
    w = np.clip(raw, 0.0, caps)
    return w / w.sum()


def _solve(
    name: str,
    objective: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    problem: _Problem,
    starts: list[np.ndarray],
    extra_constraints: tuple[dict, ...] = (),
    accept: Callable[[np.ndarray], None] | None = None,
) -> tuple[np.ndarray, int]:
    """Run SLSQP from each start; keep the best solution that passes verification."""
    constraints = [
        {"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones_like(w)},
        *extra_constraints,
    ]
    bounds = list(zip(np.zeros(len(problem.caps)), problem.caps))
    best: tuple[float, np.ndarray, int] | None = None
    failures: list[str] = []

    for x0 in starts:
        res = minimize(objective, x0, jac=gradient, method="SLSQP", bounds=bounds,
                       constraints=constraints, options=_SOLVER_OPTIONS)
        if not res.success:
            failures.append(str(res.message))
            continue
        try:
            w = _verify(np.asarray(res.x, dtype=float), problem.caps)
            if accept is not None:
                accept(w)
        except OptimizationError as exc:
            failures.append(str(exc))
            continue
        value = objective(w)
        if best is None or value < best[0]:
            best = (value, w, int(res.nit))

    if best is None:
        reasons = "; ".join(dict.fromkeys(failures)) or "no starting point succeeded"
        raise OptimizationError(f"{name} optimisation failed: {reasons}")
    return best[1], best[2]


def _result(w, problem, rf_annual, objective, iterations, notes=()) -> OptimizationResult:
    weights = pd.Series(w, index=problem.assets, name="weight")
    cov_df = pd.DataFrame(problem.cov, index=problem.assets, columns=problem.assets)
    mu_s = None if problem.mu is None else pd.Series(problem.mu, index=problem.assets)
    stats = portfolio_stats(weights, mu_s, cov_df, rf_annual)
    return OptimizationResult(weights, stats.expected_return, stats.volatility, stats.sharpe,
                              objective, iterations, tuple(notes))


# --------------------------------------------------------------------------- #
# Public optimisers
# --------------------------------------------------------------------------- #
def min_variance(
    cov: pd.DataFrame,
    *,
    mu: pd.Series | None = None,
    max_weight: Caps = 1.0,
    rf_annual: float = 0.0,
) -> OptimizationResult:
    """Long-only, capped global minimum-variance portfolio.

    Convex problem (Sigma is positive semi-definite), so any local minimum is global.
    mu is optional and only used to report the portfolio's expected return / Sharpe.
    """
    p = _prepare(cov, mu, max_weight)
    sigma = p.cov
    w, nit = _solve(
        "Minimum-variance",
        objective=lambda w: float(w @ sigma @ w),
        gradient=lambda w: 2.0 * sigma @ w,
        problem=p,
        starts=[_feasible_start(p.caps), _greedy(np.diag(sigma), p.caps, descending=False)],
    )
    notes = []
    if math.sqrt(max(float(w @ sigma @ w), 0.0)) <= _ZERO_VOL:
        notes.append("This portfolio has (numerically) zero volatility; its Sharpe ratio is undefined.")
    return _result(w, p, rf_annual, MIN_VARIANCE, nit, notes)


def max_sharpe(
    mu: pd.Series,
    cov: pd.DataFrame,
    *,
    rf_annual: float = 0.0,
    max_weight: Caps = 1.0,
) -> OptimizationResult:
    """Long-only, capped portfolio with the highest (w' mu - rf) / sqrt(w' Sigma w)."""
    p = _prepare(cov, mu, max_weight)
    sigma, excess = p.cov, p.mu - annual_arithmetic_risk_free(rf_annual)

    zero_vol = [a for a, v in zip(p.assets, np.diag(sigma)) if math.sqrt(max(v, 0.0)) <= _ZERO_VOL]
    if zero_vol:
        raise OptimizationError(
            f"Maximum-Sharpe is undefined with zero-volatility asset(s) ({', '.join(zero_vol)}): "
            "the ratio would divide by zero. The risk-free rate is already modelled separately — "
            "remove these assets or use minimum variance."
        )

    # If ANY allowed portfolio has zero risk, the Sharpe ratio is unbounded. The minimum-
    # variance problem is convex, so checking its solution detects this case exactly.
    gmv = min_variance(cov, mu=mu, max_weight=max_weight, rf_annual=rf_annual)
    if gmv.volatility <= _ZERO_VOL:
        raise OptimizationError(
            "A zero-risk combination of these assets exists (perfectly offsetting returns), "
            "so the Sharpe ratio is unbounded. Check the inputs for duplicated or inverse assets."
        )
    # Floor for intermediate solver iterates, which may briefly leave the feasible set.
    vol_floor = gmv.volatility * 1e-3

    def neg_sharpe(w: np.ndarray) -> float:
        vol = max(math.sqrt(max(float(w @ sigma @ w), 0.0)), vol_floor)
        return -float(w @ excess) / vol

    def neg_sharpe_grad(w: np.ndarray) -> np.ndarray:
        # d/dw [ (w'e) / s ] = e / s - (w'e) * Sigma w / s^3,  with s = sqrt(w' Sigma w)
        sigma_w = sigma @ w
        vol = max(math.sqrt(max(float(w @ sigma_w), 0.0)), vol_floor)
        return -(excess / vol - float(w @ excess) * sigma_w / vol**3)

    starts = [
        _feasible_start(p.caps),
        gmv.weights.reindex(p.assets).to_numpy(),
        _greedy(p.mu, p.caps, descending=True),
        _greedy(excess / np.sqrt(np.diag(sigma)), p.caps, descending=True),  # best stand-alone Sharpe
    ]
    w, nit = _solve("Maximum-Sharpe", neg_sharpe, neg_sharpe_grad, p, starts)

    result = _result(w, p, rf_annual, MAX_SHARPE, nit)
    notes: list[str] = []
    if result.sharpe <= 0:
        notes.append(
            "No allowed portfolio is expected to beat the risk-free rate, so this 'maximum-Sharpe' "
            "portfolio has a non-positive Sharpe ratio and is not a meaningful tangency portfolio."
        )
    return replace(result, notes=tuple(notes))


def min_variance_for_target(
    mu: pd.Series,
    cov: pd.DataFrame,
    target_return: float,
    *,
    max_weight: Caps = 1.0,
    rf_annual: float = 0.0,
    x0: np.ndarray | None = None,
) -> OptimizationResult:
    """Lowest-variance portfolio whose expected return equals `target_return` exactly."""
    p = _prepare(cov, mu, max_weight)
    w_lo = _greedy(p.mu, p.caps, descending=False)
    w_hi = _greedy(p.mu, p.caps, descending=True)
    r_lo, r_hi = float(p.mu @ w_lo), float(p.mu @ w_hi)
    tol = WEIGHT_TOLERANCE * max(1.0, abs(target_return))
    if not math.isfinite(target_return) or not r_lo - tol <= target_return <= r_hi + tol:
        raise OptimizationError(
            f"Target return {target_return:.2%} is outside the achievable range "
            f"{r_lo:.2%} to {r_hi:.2%} under these constraints."
        )
    target = min(max(target_return, r_lo), r_hi)

    # A guaranteed-feasible start: the mix of the min- and max-return corners that hits the target.
    share = 0.0 if r_hi - r_lo <= 1e-15 else (target - r_lo) / (r_hi - r_lo)
    starts = [] if x0 is None else [np.asarray(x0, dtype=float)]
    starts.append((1 - share) * w_lo + share * w_hi)

    def hits_target(w: np.ndarray) -> None:
        if abs(float(w @ p.mu) - target) > tol:
            raise OptimizationError(f"Solution misses the target return by {abs(w @ p.mu - target):.2e}.")

    sigma = p.cov
    w, nit = _solve(
        f"Target-return ({target:.2%})",
        objective=lambda w: float(w @ sigma @ w),
        gradient=lambda w: 2.0 * sigma @ w,
        problem=p,
        starts=starts,
        extra_constraints=({"type": "eq", "fun": lambda w: w @ p.mu - target, "jac": lambda w: p.mu},),
        accept=hits_target,
    )
    return _result(w, p, rf_annual, TARGET_RETURN, nit)
