# QuantEdge — Portfolio Risk & Optimization Lab

[![tests](https://github.com/parthj28-dev/QuantEdge/actions/workflows/tests.yml/badge.svg)](https://github.com/parthj28-dev/QuantEdge/actions/workflows/tests.yml)

An interactive lab for security analysis and portfolio management, built with Python,
SciPy and Streamlit. It downloads market data, measures risk, builds optimised portfolios,
tests them with a walk-forward backtest, values companies from fundamentals, reads technical
indicators and evaluates portfolio performance. Every number comes from a tested library;
the dashboard only displays it.

**Research and educational tool only. Historical data is not a forecast and nothing here is
investment advice.**

## What it does

| Area | Features |
|---|---|
| Risk & return | CAGR vs arithmetic return, volatility, Sharpe, max drawdown, correlation heatmap |
| CAPM | Beta, alpha, R², residual risk, CAPM expected returns |
| Estimation | Ledoit-Wolf shrinkage covariance, Black-Litterman returns with absolute / relative views |
| Optimisation | Min variance, max Sharpe, inverse volatility, risk parity, max diversification, efficient frontier, risk contributions |
| Backtest | Walk-forward with no look-ahead, drift, turnover costs, cash, 8 strategies incl. 200-day SMA trend filters |
| **Fundamental analysis** (v3) | Zero-growth, Gordon, multistage and payout/ROE-driven DDM, PVGO, FCFF/FCFE DCF, justified P/E, PEG, peer multiples, buy/sell signal |
| **Technical analysis** (v3) | SMA/EMA, Bollinger bands, MACD, RSI, signal log, support/resistance, candlestick patterns, volume confirmation, TRIN |
| **Performance measurement** (v3) | Sharpe, Treynor, Jensen's alpha (t-stat), information ratio, tracking error, M², systematic vs unsystematic risk, Treynor-Mazuy and Henriksson-Merton market timing |

The fundamental analysis, technical analysis and performance modules follow the formulas of
the Security Analysis & Portfolio Management course (BITS Pilani, K K Birla Goa Campus) and are
tested against its worked examples: for instance the Gordon model gives 3.92 and 54, the
two-stage DDM 28.89, and the Treynor ratios 4.4 %, 7.6 % and 8.3 %.

## Quick start

Requires Python 3.12 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest                             # 369 tests, fully offline
streamlit run app/streamlit_app.py
```

In the sidebar, enter Yahoo Finance symbols (e.g. `RELIANCE.NS, TCS.NS` with benchmark `^NSEI`)
or upload a CSV of adjusted prices (first column `Date` as YYYY-MM-DD, one column per ticker).
The **Valuation** tab and the summary-statistics mode of **Performance** work without any data.

### Using the library directly

```python
from quantedge.valuation import multistage_ddm, valuation_signal
res = multistage_ddm(d0=6, growth_rates=[0.08] * 3, terminal_growth=0.06, k=0.10)
print(round(res.value, 2), valuation_signal(res.value, market_price=200))   # 167.84 Sell

from quantedge.performance import measures_from_summary
import pandas as pd
funds = pd.DataFrame({"return": [0.12, 0.16, 0.18], "beta": [0.9, 1.05, 1.2]}, index=["W", "X", "Y"])
print(measures_from_summary(funds, market_return=0.14, risk_free=0.08)[["treynor", "treynor_rank"]])
```

## Deploying the dashboard

The app is a Streamlit server, so it is hosted on **Streamlit Community Cloud** (free), not on a
static host such as Vercel. On [share.streamlit.io](https://share.streamlit.io): *Create app* →
this repository, branch `main`, main file `app/streamlit_app.py`, Python 3.12 or 3.13. The app
adds `src/` to its import path when the package isn't installed, so `requirements.txt` is enough.

## Design

* **Layered:** `data → metrics → indicators → capm/optimize/estimators/riskbased → frontier →
  backtest → valuation/technical/performance → charts → app`. The library never imports Streamlit
  (enforced by a test).
* **Never fabricate data:** missing prices are dropped and reported, never filled; undefined
  statistics are `NaN`, never 0.
* **Tested:** 369 offline tests, including three look-ahead-bias checks, a Ledoit-Wolf cross-check
  against scikit-learn reference values, lecture examples for every v3 formula, and headless
  tests that drive the real dashboard.

Formulas, conventions and decisions: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Limitations

* Expected returns estimated from history are very noisy and optimised weights are sensitive to them.
* yfinance is an unofficial data source; save the validated prices (download button) to reproduce results.
* Backtests ignore taxes, slippage beyond a fixed cost rate, and liquidity.
* Technical signals are shown for learning, not as evidence of an edge.
* Valuations are only as good as the growth and discount-rate assumptions typed in.

## Licence

MIT — see [`LICENSE`](LICENSE).
