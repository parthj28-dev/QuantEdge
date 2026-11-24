# QuantEdge v3.0 — Architecture, Formulas & Design Decisions

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
   valuation.py, technical.py, performance.py   (v3: FA, TA, portfolio performance measurement)
           │
   backtest.py            (walk-forward simulation)
           │
   estimators.py, riskbased.py, frontier.py   (v2: robust inputs, risk-based portfolios)
           │
   optimize.py, capm.py   (constrained optimisation, CAPM regression)
           │
   indicators.py          (v2.1: simple moving averages, trend signals)
           │
   metrics.py             (returns, risk, performance statistics)
           │
   data.py                (fetch + validate prices; v3: OHLCV for one ticker)
           │
   config.py, exceptions.py (constants and error types)
   ```

2. **Pure functions in, DataFrames out.** Calculation functions take pandas
   objects and plain numbers and return pandas objects or dataclasses. No global
   state, no hidden I/O. This makes them testable with tiny hand-made inputs.

3. **Fail loudly, never fabricate.** Missing or invalid data is *reported* or
   *raised*, never forward-filled, zero-filled or interpolated behind your back.
   Undefined statistics (e.g. a Sharpe ratio when volatility is zero, RSI over a
   flat window, an implied price from a negative metric) are returned as `NaN` —
   "undefined" — never as `0` or `inf`.

4. **Dependency injection for the network.** `fetch_prices` and `fetch_ohlcv`
   accept a `downloader` argument. Production uses yfinance; tests pass a fake
   that returns a DataFrame shaped like yfinance's output. The test suite therefore
   never touches the internet and is fully reproducible.

5. **One set of conventions** (Section 3), defined once in `config.py`.

6. **Textbook-checked.** The v3 modules (valuation, technical, performance) are
   tested against the worked examples in the SAPM course lectures, so every
   formula reproduces a number a student can check by hand.

---

## 2. File tree

```
quantedge/
├── README.md                 setup, usage, formulas summary, limitations
├── CHANGELOG.md              release notes
├── LICENSE                   MIT
├── pyproject.toml            package metadata, pinned deps, pytest config
├── requirements.txt          exact pinned versions (+ the package itself, for app hosts)
├── .github/workflows/tests.yml   CI: install from requirements.txt, run pytest
├── .streamlit/config.toml    usage telemetry off by default
├── docs/
│   └── ARCHITECTURE.md       this file
├── src/quantedge/
│   ├── config.py             constants: 252 trading days, tolerances, defaults
│   ├── exceptions.py         DataError, InsufficientDataError, OptimizationError, EstimationError
│   ├── data.py               ticker parsing, yfinance fetch, CSV load, validation; OHLCV (v3)
│   ├── metrics.py            returns, annualisation, vol, Sharpe, drawdown, corr/cov
│   ├── indicators.py         [v2.1] SMA, trend status, golden/death crosses
│   ├── capm.py               beta / alpha regression, CAPM expected returns
│   ├── optimize.py           min-variance and max-Sharpe (SciPy SLSQP)
│   ├── estimators.py         [v2] Ledoit-Wolf covariance, Black-Litterman returns
│   ├── riskbased.py          [v2] risk contributions, inverse vol, risk parity, max diversification
│   ├── frontier.py           efficient frontier by repeated constrained solves
│   ├── backtest.py           walk-forward backtest, costs, drift, cash, strategy registry
│   ├── valuation.py          [v3] DDM, PVGO, FCFF/FCFE DCF, justified P/E, peer multiples
│   ├── technical.py          [v3] EMA, Bollinger, MACD, RSI, signals, candlesticks, TRIN, volume
│   ├── performance.py        [v3] Sharpe, Treynor, Jensen, IR, M², diversification, market timing
│   └── charts.py             Plotly figures
├── app/
│   └── streamlit_app.py      dashboard — orchestration and display only (11 tabs)
└── tests/                    pytest suite (one file per module + headless app tests)
```

The `src/` layout forces the package to be *installed* (`pip install -e .`)
before tests run, which catches packaging mistakes that a flat layout hides.

---

## 3. Conventions (used everywhere)

| Item | Convention |
|---|---|
| Price series | Adjusted close (splits **and** dividends), daily. Candlesticks use the unadjusted session OHLC. |
| Return type | **Simple** returns $r_t = P_t / P_{t-1} - 1$ |
| Trading days per year | $N = 252$ (`config.TRADING_DAYS_PER_YEAR`) |
| Risk-free input | Annual **effective** rate, e.g. `0.065` for 6.5 % |
| Daily risk-free | $r_f^{d} = (1 + r_f)^{1/N} - 1$ |
| Risk-free in arithmetic annual units | $r_f^{\text{ann}} = N \cdot r_f^{d}$ (used wherever rf is compared with μ) |
| Expected return (μ) for optimisation | Arithmetic mean × $N$ |
| Realised growth for reporting | CAGR (geometric) |
| Volatility | Sample std (ddof = 1) × $\sqrt{N}$ |
| Covariance | Sample covariance × $N$ (or Ledoit-Wolf, same units) |
| Other frequencies (performance tab) | Daily returns compounded to W/M; annualise with $P$ = 252 / 52 / 12 |
| Missing prices | Dates where any selected series is missing are **dropped and reported**; never filled |
| Valuation inputs | Per-share amounts and decimal rates; no market data needed |

**Why simple returns, not log returns?** A portfolio's return is a weighted sum
of *simple* asset returns: $r_p = \sum_i w_i r_i = w^\top r$. That linearity is
what makes $w^\top \mu$ and $w^\top \Sigma w$ correct, so mean–variance
optimisation must use simple returns.

**Why two annual return numbers?** The arithmetic mean × 252 is the right input
for the optimiser. CAGR is what an investor actually experienced;
roughly CAGR ≈ μ − σ²/2. The dashboard shows both and labels them.

---

## 4. Formulas

### 4.1 Returns and risk (`metrics.py`)

| Quantity | Formula |
|---|---|
| Daily simple return | $r_t = \dfrac{P_t}{P_{t-1}} - 1$ |
| Annualised arithmetic return | $\mu = N \cdot \bar r$ |
| CAGR | $\left(\prod_{t=1}^{T}(1+r_t)\right)^{N/T} - 1$ |
| Annualised volatility | $\sigma = \sqrt{N}\cdot s$, $s$ = sample std of daily returns |
| Sharpe ratio | $S = \sqrt{N}\cdot \dfrac{\overline{r - r_f^{d}}}{s(r - r_f^{d})}$ — `NaN` if $s \approx 0$ |
| Wealth index | $W_0 = 1,\; W_t = \prod_{k\le t}(1+r_k)$ |
| Drawdown | $D_t = W_t / \max_{k \le t} W_k - 1$ (with $W_0=1$ included) |
| Maximum drawdown | $\text{MDD} = \min_t D_t$ (≤ 0) |
| Correlation | $\rho_{ij} = \Sigma_{ij} / (\sigma_i \sigma_j)$ — `NaN` for a zero-variance asset |

### 4.2 CAPM (`capm.py`)

OLS on aligned **excess** returns: $r_{i,t} - r_f^{d} = \alpha_i + \beta_i\,(r_{m,t} - r_f^{d}) + \varepsilon_t$.
Outputs: $\hat\beta$, $\alpha^{\text{ann}} = N\hat\alpha$, $R^2$, std. error of beta,
residual volatility $\sqrt{N \cdot \text{SSR}/(T-2)}$. CAPM expected return
$E[R_i] = r_f^{\text{ann}} + \beta_i \cdot \text{MRP}$, with MRP historical or user-supplied.
With the historical premium, *historical mean = CAPM expected return + annual alpha* exactly (tested).

### 4.3 Optimisation (`optimize.py`)

Long-only, fully invested, capped: $\sum_i w_i = 1,\ 0 \le w_i \le c_i$ (feasible only if $\sum c_i \ge 1$).
Minimum variance $\min w^\top\Sigma w$; maximum Sharpe $\max (w^\top\mu - r_f^{\text{ann}})/\sqrt{w^\top\Sigma w}$
(four starting points, non-convex); target return for the frontier. SciPy SLSQP with analytic gradients;
every solution re-verified independently of `res.success`.

### 4.4 Efficient frontier (`frontier.py`)

Min-variance return to the highest return achievable under the caps; $K$ target-return solves,
warm-started; failed targets recorded, not hidden.

### 4.5 Backtest (`backtest.py`)

* **No look-ahead:** on rebalance date $t$ the strategy receives only a copy of the
  last $L$ returns up to and including $t$; new weights earn from $t+1$.
* **Drift:** $w_{i,t+1} = w_{i,t}(1 + r_{i,t+1}) / (1 + r_{p,t+1})$.
* **Costs:** turnover $\tau_t = \sum_{i \in \text{assets}} |w^{\text{target}}_{i} - w^{\text{drift}}_{i}|$, value $\times (1 - c\,\tau_t)$.
* **Cash (v2.1):** reserved key `CASH` earns $r_f^d$ from `cash_rate`; turnover counts risky assets
  only, so the initial purchase costs $c \times$ the invested fraction.
* **Strategies (8):** equal weight, min variance, max Sharpe, inverse volatility, risk parity,
  max diversification (each with sample or Ledoit-Wolf covariance), equal weight + SMA filter,
  risk parity + SMA filter. Registry `STRATEGY_FACTORIES[name](rf, cap, cov_method, sma_window)`.
* **Look-ahead tests:** spy strategy, poisoned future, same-day trap.

### 4.6 Charts and dashboard (`charts.py`, `app/`)

Pure Plotly figure builders with one validated palette; scatter charts colour ≤ 3 series; every chart
has a table next to it in the app. The dashboard only orchestrates; each section catches
`QuantEdgeError` and shows the message while the rest of the page renders. A test enforces that
importing any package module never imports Streamlit.

Tabs: Overview · Risk · CAPM · Estimation · Optimisation · Backtest · Trend · Technical · Valuation ·
Performance · Methodology. Valuation and the summary-statistics mode of Performance work without
loading any market data.

### 4.7 Robust estimation (`estimators.py`, v2)

**Ledoit-Wolf shrinkage.** Demeaned daily returns $X$ ($T \times n$), $S = X^\top X / T$,
target $F = m I$, $m = \operatorname{tr}(S)/n$:

$$ \Sigma_{LW} = (1-\delta) S + \delta F, \qquad
\delta = \min\!\Big(1,\ \frac{\phi}{T\,\gamma}\Big),\quad
\phi = \sum_{ij}\Big[\tfrac1T \sum_t x_{ti}^2 x_{tj}^2 - S_{ij}^2\Big],\quad
\gamma = \lVert S - F \rVert_F^2 $$

rescaled to the unbiased and annual convention. Intensity verified against scikit-learn
reference values (frozen in the tests).

**Black-Litterman.** $\Pi = \delta\,\Sigma\, w_{mkt}$; views $P\mu = Q$ with confidence $c_k \in (0,1]$,
$\Omega_{kk} = (1/c_k - 1)(P\tau\Sigma P^\top)_{kk}$;
$\mu_{BL} = \Pi + \tau\Sigma P^\top (P\tau\Sigma P^\top + \Omega)^{-1}(Q - P\Pi)$,
$\Sigma_{BL} = \Sigma + \tau\Sigma - \tau\Sigma P^\top (P\tau\Sigma P^\top + \Omega)^{-1} P \tau\Sigma$.
Views are total returns converted with $Q - r_f^{\text{ann}}\sum_j P_{kj}$. Risk aversion manual or
implied $\delta = (E[R_m]-r_f)/\sigma_m^2$ (must be positive; the app falls back to manual).
Tested: no views ⇒ prior and $(1+\tau)\Sigma$; max-Sharpe on the prior recovers $w_{mkt}$; $c=1$ matches
a view exactly, $c=0.5$ moves halfway; duplicate certain views raise `EstimationError`.

### 4.8 Risk-based portfolios (`riskbased.py`, v2)

Risk share$_i = w_i(\Sigma w)_i / w^\top\Sigma w$.
Inverse volatility (closed form, water-filled caps); risk parity by Newton's method on
$\tfrac12 y^\top\Sigma y - \tfrac1n\sum\ln y_i$ (damped while the Newton decrement exceeds $10^{-8}$,
then full steps; a 100-problem stress test guards convergence), with a capped least-squares fallback
and a note when caps make equal shares impossible; maximum diversification as max-Sharpe with
$\mu = \sigma$, $r_f = 0$. Identities tested: two assets ⇒ ERC = inverse vol; equal correlations ⇒
all three coincide; uncorrelated ⇒ max diversification = inverse vol.

### 4.9 Moving averages and the trend filter (`indicators.py`, v2.1)

$\text{SMA}_n(t) = \frac1n\sum_{k=0}^{n-1} P_{t-k}$; first $n-1$ values undefined. Above trend means
$P_t > \text{SMA}_n(t)$ strictly. Golden cross: fast SMA moves from $\le$ to $>$ the slow SMA.
Inside the backtest $P_t/\text{SMA}_n(t)$ is computed exactly from relative prices over the last
$n-1$ returns. `sma_trend_filter(base, n)` keeps each asset's base weight while above trend and moves
the rest to cash (not redistributed).

### 4.10 Fundamental analysis (`valuation.py`, v3, lectures 22–25)

| Model | Formula |
|---|---|
| Holding-period return | $HPR = (D_1 + P_1 - P_0)/P_0$; signal Buy if $HPR > k$ |
| Required return | $k = r_f + \beta\,\text{MRP}$, or implied $k = D_1/P_0 + g$ |
| Sustainable growth | $g = b \times ROE$ |
| Zero growth | $V_0 = D/k$ |
| Gordon | $V_0 = D_1/(k-g)$, needs $k > g$ |
| Multistage DDM | $V_0 = \sum_{t=1}^{T} D_t/(1+k)^t + V_T/(1+k)^T$, $V_T = D_{T+1}/(k - g)$ |
| Two-stage closed form | $\frac{D_1}{k-g_1}\Big[1-\big(\frac{1+g_1}{1+k}\big)^N\Big] + \frac{D_{N+1}}{(k-g_2)(1+k)^N}$ |
| Payout/ROE-driven DDM | $E_t = E_{t-1}(1 + b_{t-1}ROE_{t-1})$, $D_t = (1-b_t)E_t$ |
| PVGO | $P_0 = E_1/k + PVGO$ |
| FCFF | $EBIT(1-t) + \text{Dep} - \text{Capex} - \Delta NWC$ |
| FCFE | $FCFF - \text{Int}(1-t) + \text{net borrowing}$ $= NI - (\text{Capex}-\text{Dep}) - \Delta WC + \text{new debt} - \text{repaid}$ |
| Equity from firm value | $(V_{firm} - \text{net debt})/\text{shares}$ |
| Justified P/E | forward $\frac{1-b}{k - b\,ROE}$, trailing $\frac{(1-b)(1+g)}{k-g}$, FCFE $\frac{FCFE_1/E_1}{k-g}$ |
| PEG | $(P/E)/(100\,g)$ |
| Peer multiples | implied price = multiple × own metric; EV/EBITDA via the equity bridge |

The payout/ROE model makes a payout change move the dividend immediately and the growth rate one year
later (lecture 22, problem 2). Lecture examples reproduced exactly in the tests: 3.92, 25, 54, 28.89,
~30.42, ₹168 (sell), 50.07 (sell), PVGO −11.11, P/E 11.43 and value 31.14.

### 4.11 Technical analysis (`technical.py`, v3, lectures 26–29)

| Indicator | Definition |
|---|---|
| EMA | $EMA_t = EMA_{t-1} + \frac{2}{n+1}(P_t - EMA_{t-1})$, seeded with the first full SMA |
| Bollinger | $SMA_n \pm k\,\sigma_n$ (population σ), $n=20, k=2$; %B and bandwidth |
| MACD | $EMA_{12} - EMA_{26}$; signal = 9-day **SMA** of MACD (lecture; EMA optional); histogram |
| RSI | $100 - 100/(1+RS)$, RS = average gain / average loss over 14 changes (simple; Wilder optional) |
| Support / resistance | min / max of the **previous** $n$ prices (no look-ahead) |
| TRIN | $(A/D \text{ issues}) / (A/D \text{ volume})$ |

Signals: MACD crossing its signal line or zero; RSI entering overbought (> 70) or oversold (< 30);
close breaking outside the bands; band squeeze flagged when bandwidth is in its lowest decile.
Candlesticks (hammer, hanging man, doji, engulfing, harami, morning/evening star) are detected on
validated OHLC with a prior-trend filter; single-candle patterns are marked as needing confirmation.

### 4.12 Portfolio performance measurement (`performance.py`, v3, lectures 30–32)

| Measure | Formula |
|---|---|
| Sharpe | $(\bar R - \bar R_f)/\sigma$ (σ of excess returns in series mode, as in `metrics`) |
| Treynor | $(\bar R - \bar R_f)/\beta$ |
| Jensen alpha | intercept of $R_t - R_{f,t} = \alpha + \beta(R_{m,t} - R_{f,t}) + e_t$, with t-statistic |
| Information ratio | $(\bar R - \bar R_b)/\sigma(R - R_b)$; tracking error $\sigma(R - R_b)$ |
| M² | $R_f + S \times \sigma_M$ |
| Diversification | systematic $\beta^2\sigma_M^2$, unsystematic $\sigma^2 - \beta^2\sigma_M^2$, $R^2$ |
| Treynor-Mazuy | $r_p - r_f = a + b_1 x + b_2 x^2 + e$, $x = r_M - r_f$ |
| Henriksson-Merton | $r_p - r_f = a + b_1 x + b_2 x D + e$, $D = 1$ if $x > 0$ |

Summary mode takes return/beta/σ per portfolio (the lecture problems). Series mode works at daily,
weekly or monthly frequency over dates shared with the benchmark. Tested: lecture Treynor and Sharpe
examples, P/Q/R/S exercise, beta/alpha/Sharpe identical to `capm` and `metrics`, Henriksson-Merton
recovers both betas exactly on noiseless data, collinear regressors raise `EstimationError`.

---

## 5. Error handling contract

| Situation | Behaviour |
|---|---|
| Ticker returns no data | Excluded, listed in `report.missing_tickers` |
| All tickers fail / network failure | `DataError` |
| NaN inside a series | Those dates dropped, count reported per ticker |
| Price ≤ 0 or ±inf, duplicate dates | `DataError` |
| Too few observations | `InsufficientDataError` saying how many are needed |
| Zero-volatility asset | Sharpe/correlation = `NaN`; max-Sharpe and risk-based methods raise `OptimizationError` |
| Infeasible caps / solver failure | `OptimizationError` |
| Invalid BL views, market weights, δ ≤ 0 | `EstimationError` naming the problem |
| Caps make equal risk contributions impossible | Closest feasible solution + note |
| SMA window longer than the data / lookback | `InsufficientDataError` |
| Data column named `Cash` in a backtest | `DataError` (reserved name) |
| A strategy cannot start in the app | That strategy is reported; the others run |
| $k \le g$ in any growing-perpetuity model | `EstimationError` (value would be infinite) |
| Non-numeric / non-finite valuation input, payout outside 0–1 | `EstimationError` |
| Multiple whose metric is missing or ≤ 0 | Implied price `NaN` with a note (not 0) |
| Flat RSI window | `NaN` (undefined), not 50 |
| OHLC with High < max(O, C), Low > min(O, C), gaps in prices | `DataError`; incomplete candles dropped and counted when downloading |
| Market never below rf (Henriksson-Merton) | `EstimationError` (regressors collinear) |

---

## 6. Releases

| Release | Deliverable | Tests |
|---|---|---|
| v1.0.0 | Data, metrics, CAPM, optimisation, frontier, backtest, charts, dashboard | 211 |
| v2.0 / v2.1 | Ledoit-Wolf, Black-Litterman, risk-based portfolios, SMA indicators, cash, trend filter | — |
| **v3.0.0** | v2.x features rebuilt from this specification (the v2.x code was never published), plus valuation, technical analysis, performance measurement, OHLCV data, 11-tab dashboard | 369 |

---

## 7. Known limitations (carried into README)

* Historical estimates of μ are extremely noisy; optimiser output is highly sensitive to them.
  Caps, CAPM, Black-Litterman and shrinkage are mitigations, not cures.
* Black-Litterman's equilibrium is only as good as the market weights supplied; the dashboard's
  equal-weight default is a placeholder.
* Ledoit-Wolf shrinks towards a scaled identity only.
* yfinance is an unofficial interface to Yahoo Finance; save validated prices to CSV for reproducibility.
* Mixing exchanges with different holiday calendars drops non-common dates.
* Backtests assume execution at the close, a fixed cost rate, no taxes and fractional shares.
* Technical-analysis signals are descriptive; they are not backtested here and are not evidence of an edge.
  Candlestick rules use fixed thresholds that analysts apply with judgement.
* Valuation outputs depend entirely on the growth, payout and discount-rate assumptions entered; the
  terminal value usually dominates.
* Market-timing regressions need many periods and both up and down markets; daily data adds noise.
* Past performance is research data, not a forecast.
