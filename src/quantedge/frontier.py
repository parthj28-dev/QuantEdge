"""Efficient frontier: the lowest achievable risk for each level of expected return.

Algorithm
---------
1. Solve the global minimum-variance (GMV) portfolio. Its return R_min is the
   bottom of the *efficient* frontier: below it, you could get more return for
   less risk, so those portfolios are inefficient.
2. Find the highest return any allowed portfolio can reach, R_max. Because the
   constraints are only "sum to 1, 0 <= w <= cap", this is exact without a solver:
   fill the highest-return assets up to their caps (optimize.achievable_return_range).
3. Pick n targets evenly spaced in [R_min, R_max] and, for each, solve
       minimise w' Sigma w   s.t.  w' mu = target,  sum(w) = 1,  0 <= w <= cap.
4. Warm-start each solve from the previous point's weights (neighbouring points
   have similar weights, so the solver converges faster and more reliably).
5. A target that fails is recorded in `failed_targets` — never silently skipped
   or interpolated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .exceptions import OptimizationError
from .optimize import (
    Caps,
    OptimizationResult,
    achievable_return_range,
    min_variance,
    min_variance_for_target,
)

# Volatility may dip by at most this much between consecutive points before we flag it.
_MONOTONIC_TOLERANCE = 1e-6


@dataclass(frozen=True)
class FrontierResult:
    points: pd.DataFrame    # target_return, expected_return, volatility, sharpe (one row per point)
    weights: pd.DataFrame   # same rows, one column per asset
    min_variance: OptimizationResult
    failed_targets: tuple[float, ...]
    notes: tuple[str, ...]


def efficient_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    *,
    n_points: int = 25,
    max_weight: Caps = 1.0,
    rf_annual: float = 0.0,
) -> FrontierResult:
    """Trace the long-only, capped efficient frontier with `n_points` constrained solves."""
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")

    gmv = min_variance(cov, mu=mu, max_weight=max_weight, rf_annual=rf_annual)
    r_min = gmv.expected_return
    r_max = achievable_return_range(mu, max_weight=max_weight)[1]

    notes: list[str] = []
    if r_max - r_min <= 1e-12 * max(1.0, abs(r_max)):
        notes.append(
            "The minimum-variance portfolio already has the highest achievable return, "
            "so the efficient frontier is a single point."
        )
        return _assemble([(r_min, gmv)], gmv, [], notes)

    targets = np.linspace(r_min, r_max, n_points)
    solved: list[tuple[float, OptimizationResult]] = [(float(targets[0]), gmv)]
    failed: list[float] = []
    previous = gmv.weights.to_numpy()
    for target in targets[1:]:
        try:
            point = min_variance_for_target(mu, cov, float(target), max_weight=max_weight,
                                            rf_annual=rf_annual, x0=previous)
        except OptimizationError:
            failed.append(float(target))
            continue
        solved.append((float(target), point))
        previous = point.weights.to_numpy()

    if len(solved) < 2:
        raise OptimizationError("Efficient frontier failed: no target-return problem could be solved.")
    if failed:
        notes.append(
            f"{len(failed)} of {n_points} frontier points could not be solved and are omitted "
            f"(targets: {', '.join(f'{t:.2%}' for t in failed)})."
        )
    return _assemble(solved, gmv, failed, notes)


def _assemble(solved, gmv, failed, notes) -> FrontierResult:
    points = pd.DataFrame(
        {
            "target_return": [t for t, _ in solved],
            "expected_return": [r.expected_return for _, r in solved],
            "volatility": [r.volatility for _, r in solved],
            "sharpe": [r.sharpe for _, r in solved],
        }
    )
    weights = pd.DataFrame([r.weights for _, r in solved]).reset_index(drop=True)

    dips = np.diff(points["volatility"].to_numpy())
    if (dips < -_MONOTONIC_TOLERANCE).any():
        notes.append(
            "Warning: volatility decreases between some consecutive frontier points, which "
            "indicates solver inaccuracy; treat the affected points with caution."
        )
    return FrontierResult(points, weights, gmv, tuple(failed), tuple(notes))
