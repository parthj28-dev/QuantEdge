# QuantEdge — Architecture, Formulas & Build Plan

This document is the design contract for the project. Every module, formula and
convention used in the code is defined here first, so the code can be checked
against it (and so you can explain it in an interview without reading code).

---

## 1. Design principles

1. **Layered, one-directional dependencies.** Each layer only imports from
   layers below it. The math never imports Streamlit or Plotly.

   ```
   app/streamlit_app.py   (UI: widgets, layout, error display)
           │
   charts.py              (Plotly figure builders — pure functions, no Streamlit)
           │
   backtest.py            (walk-forward simulation)
           │
   optimize.py, frontier.py, capm.py   (portfolio math)
           │
   metrics.py             (returns, risk, performance statistics)
           │
   data.py                (fetch + validate prices)
           │
   config.py, exceptions.py (constants and error types)
   ```

2. **Pure functions in, DataFrames out.** Calculation functions take pandas
   objects and plain numbers and return pandas objects or dataclasses. No global
   state, no hidden I/O. This makes them testable with tiny hand-made inputs.

3. **Fail loudly, never fabricate.** Missing or invalid data is *reported* or
   *raised*, never forward-filled, zero-filled or interpolated behind your back.
   Undefined statistics (e.g. a Sharpe ratio when volatility is zero) are
   returned as `NaN` — "undefined" — never as `0` or `inf`.

4. **Dependency injection for the network.** `fetch_prices` accepts a
   `downloader` argument. Production uses yfinance; tests pass a fake that
   returns a DataFrame shaped like yfinance's output. The test suite therefore
   never touches the internet and is fully reproducible.

5. **One set of conventions** (Section 3), defined once in `config.py`.

---

## 2. File tree

```
quantedge/
├── README.md                 setup, usage, formulas summary, limitations
├── pyproject.toml            package metadata, pinned deps, pytest config
├── requirements.txt          exact pinned versions for a fresh environment
├── .gitignore
├── docs/
│   └── ARCHITECTURE.md       this file
├── src/quantedge/
│   ├── __init__.py
│   ├── config.py             constants: 252 trading days, tolerances, defaults
│   ├── exceptions.py         DataError, InsufficientDataError, OptimizationError
│   ├── data.py               [Stage 1] ticker parsing, yfinance fetch, CSV load, validation
│   ├── metrics.py            [Stage 1] returns, annualisation, vol, Sharpe, drawdown, corr/cov
│   ├── capm.py               [Stage 2] beta / alpha regression, CAPM expected returns
│   ├── optimize.py           [Stage 3] min-variance and max-Sharpe (SciPy SLSQP)
│   ├── frontier.py           [Stage 3] efficient frontier by repeated constrained solves
│   ├── backtest.py           [Stage 4] walk-forward backtest, costs, drift, no look-ahead
│   └── charts.py             [Stage 5] Plotly figures (allocation, frontier, heatmap, equity)
├── app/
│   └── streamlit_app.py      [Stage 5] dashboard — orchestration and display only
└── tests/
    ├── conftest.py           deterministic synthetic fixtures (clearly test-only)
    ├── test_data.py
    ├── test_metrics.py
    ├── test_capm.py
    ├── test_optimize.py
    ├── test_frontier.py
    └── test_backtest.py
```

The `src/` layout forces the package to be *installed* (`pip install -e .`)
before tests run, which catches packaging mistakes that a flat layout hides.

---

## 3. Conventions (used everywhere)

| Item | Convention |
|---|---|
| Price series | Adjusted close (splits **and** dividends), daily |
| Return type | **Simple** returns $r_t = P_t / P_{t-1} - 1$ |
| Trading days per year | $N = 252$ (`config.TRADING_DAYS_PER_YEAR`) |
| Risk-free input | Annual **effective** rate, e.g. `0.065` for 6.5 % |
| Daily risk-free | $r_f^{d} = (1 + r_f)^{1/N} - 1$ |
| Expected return (μ) for optimisation | Arithmetic mean × $N$ |
| Realised growth for reporting | CAGR (geometric) |
| Volatility | Sample std (ddof = 1) × $\sqrt{N}$ |
| Covariance | Sample covariance × $N$ |
| Missing prices | Dates where any selected series is missing are **dropped and reported**; never filled |

**Why simple returns, not log returns?** A portfolio's return is a weighted sum
of *simple* asset returns: $r_p = \sum_i w_i r_i = w^\top r$. That linearity is
what makes $w^\top \mu$ and $w^\top \Sigma w$ correct, so mean–variance
optimisation must use simple returns. (Log returns add up nicely *over time*
but not *across assets*.)

**Why two annual return numbers?** The arithmetic mean × 252 is the right input
for the optimiser (it's the expectation of a one-period return). CAGR is what
an investor actually experienced. CAGR ≤ arithmetic mean, roughly
CAGR ≈ μ − σ²/2 — so a volatile asset can have a high average return and a poor
compounded return. The dashboard shows both and labels them.

---

## 4. Formulas

### 4.1 Returns and risk (`metrics.py`)

| Quantity | Formula |
|---|---|
| Daily simple return | $r_t = \dfrac{P_t}{P_{t-1}} - 1$ |
| Annualised arithmetic return | $\mu = N \cdot \bar r$, where $\bar r = \frac1T\sum_t r_t$ |
| CAGR | $\left(\prod_{t=1}^{T}(1+r_t)\right)^{N/T} - 1$ |
| Annualised volatility | $\sigma = \sqrt{N}\cdot s$, $s = \sqrt{\frac{1}{T-1}\sum_t (r_t-\bar r)^2}$ |
| Sharpe ratio | $S = \sqrt{N}\cdot \dfrac{\overline{r - r_f^{d}}}{s(r - r_f^{d})}$ — `NaN` if $s \approx 0$ |
| Wealth index | $W_0 = 1,\; W_t = \prod_{k\le t}(1+r_k)$ |
| Drawdown | $D_t = W_t / \max_{k \le t} W_k - 1$ (with $W_0=1$ included) |
| Maximum drawdown | $\text{MDD} = \min_t D_t$ (≤ 0) |
| Covariance (annual) | $\Sigma = N \cdot \widehat{\text{Cov}}(r)$ |
| Correlation | $\rho_{ij} = \Sigma_{ij} / (\sigma_i \sigma_j)$ — `NaN` for a zero-variance asset |

Including $W_0 = 1$ in the running peak matters: if the asset falls on day 1,
that loss is a drawdown from the starting capital.

### 4.2 CAPM (`capm.py`, Stage 2)

Regression on aligned **excess** returns (same dates for asset and benchmark):

$$ r_{i,t} - r_f^{d} = \alpha_i + \beta_i\,(r_{m,t} - r_f^{d}) + \varepsilon_t $$

$$ \beta_i = \frac{\text{Cov}(r_i - r_f,\; r_m - r_f)}{\text{Var}(r_m - r_f)}, \qquad
\alpha_i^{\text{annual}} = N \cdot \alpha_i $$

CAPM expected return (annual):

$$ E[R_i] = r_f + \beta_i \,\big(E[R_m] - r_f\big) $$

The market risk premium $E[R_m]-r_f$ is configurable: either the benchmark's
historical annualised excess return, or a user-supplied number (e.g. 6 %).
The regression also reports $R^2$ and the number of aligned observations.

### 4.3 Optimisation (`optimize.py`, Stage 3)

Let $w \in \mathbb{R}^n$, $\mu$ annual expected returns, $\Sigma$ annual covariance.
Common constraints (long-only, fully invested, capped):

$$ \sum_i w_i = 1, \qquad 0 \le w_i \le w_{\max} $$

Feasibility check before solving: need $n \cdot w_{\max} \ge 1$ (otherwise the
weights cannot sum to 1 and we raise a clear error).

* **Minimum variance:** $\min_w\; w^\top \Sigma w$
* **Maximum Sharpe:** $\max_w\; \dfrac{w^\top\mu - r_f^{\text{ann}}}{\sqrt{w^\top\Sigma w}}$,
  with $r_f^{\text{ann}} = N \cdot r_f^{d}$ so the units match the arithmetic μ.

Solved with SciPy `minimize(method="SLSQP")` using analytic gradients. After
solving, constraints are **re-verified** independently; tiny numerical noise
(|negative weight| < 1e-9) is cleaned and reported, anything larger raises
`OptimizationError`.

### 4.4 Efficient frontier (`frontier.py`, Stage 3)

1. Solve min-variance → lowest frontier return $R_{\min}$.
2. Highest achievable return under the caps, $R_{\max}$: fill the
   highest-μ assets up to $w_{\max}$ until the budget is used (exact for this
   constraint set).
3. For $K$ targets $R_k \in [R_{\min}, R_{\max}]$ solve
   $\min_w w^\top\Sigma w$ s.t. $w^\top\mu = R_k$ plus the common constraints.
4. Warm-start each solve from the previous solution; record (not hide)
   any target that fails.

### 4.5 Backtest (`backtest.py`, Stage 4)

Timeline rule (**no look-ahead**): on a rebalance date $t$ the strategy may use
only returns dated $\le t$ (i.e. prices up to the close of $t$). New weights
earn returns from $t+1$ onward.

* **Estimation window:** the last $L$ daily returns up to and including $t$.
* **Rebalance schedule:** every $H$ trading days (or month-end), configurable.
* **Weight drift between rebalances:**
  $w_{i,t+1} = \dfrac{w_{i,t}(1 + r_{i,t+1})}{1 + r_{p,t+1}}$
* **Transaction cost:** turnover $\tau_t = \sum_i |w^{\text{target}}_{i} - w^{\text{drift}}_{i}|$,
  cost $= c \cdot \tau_t$ deducted from portfolio value at $t$
  ($c$ in basis points, one-way).
* **Strategies compared:** equal weight, min-variance, max-Sharpe, and
  buy-and-hold benchmark.
* **Automated look-ahead test:** changing prices *after* date $t$ must not
  change the weights chosen at $t$.

---

## 5. Error handling contract

| Situation | Behaviour |
|---|---|
| Ticker returns no data | Excluded, listed in `report.missing_tickers` |
| All tickers fail | `DataError` |
| NaN inside a series | Those dates dropped, count reported per ticker |
| Price ≤ 0 or ±inf | `DataError` naming ticker and date |
| Duplicate dates | `DataError` |
| Too few observations | `InsufficientDataError` naming the limiting ticker |
| Zero-volatility asset | Sharpe/correlation = `NaN`; flagged by `zero_volatility_assets()` |
| Infeasible caps | `OptimizationError` before solving |
| Solver fails / violates constraints | `OptimizationError` with SciPy's message |
| Network failure | `DataError` wrapping the original exception |

The UI catches `QuantEdgeError` subclasses and shows the message; it never
shows a chart built on partial, silently-repaired data.

---

## 6. Implementation stages

| Stage | Deliverable | Tests |
|---|---|---|
| 1 | Scaffold, pinned deps, `config`, `exceptions`, `data`, `metrics` | data validation, every metric vs hand-computed values |
| 2 | `capm` | β recovers a known slope, alignment, zero-variance benchmark |
| 3 | `optimize`, `frontier` | constraints hold, caps respected, infeasible caps, 2-asset analytic min-var |
| 4 | `backtest` | costs, drift, rebalancing dates, no look-ahead |
| 5 | `charts`, Streamlit app | figure builders return valid figures; app smoke test |
| 6 | README, CI workflow, fresh-environment check | full suite in a clean venv |

---

## 7. Known limitations (carried into README)

* Historical estimates of μ are extremely noisy; optimiser output is highly
  sensitive to them ("error maximisation"). Weight caps and CAPM-based μ are
  partial mitigations, not cures.
* Sample covariance is unstable when the number of assets is large relative to
  the number of observations. (Shrinkage estimators are a natural extension.)
* yfinance is an unofficial interface to Yahoo Finance; data can change,
  be revised, or be unavailable. For reproducible research, save the validated
  prices to CSV and reload them.
* Mixing exchanges with different holiday calendars (e.g. NSE and NYSE)
  drops non-common dates; returns on the next common date then span the gap.
* Backtests assume execution at the closing price, no slippage beyond the
  fixed cost rate, no taxes, and fractional shares.
* Past performance is research data, not a forecast.
