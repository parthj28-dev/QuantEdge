# Changelog

All notable changes are documented here. Versions follow [Semantic Versioning](https://semver.org/).

## [3.0.0] — 2026-11-24

### Added — security analysis and performance measurement (SAPM lectures 22–32)
- **Fundamental analysis** (`valuation`): holding-period return and CAPM required return with
  buy/sell/hold signals; zero-growth, Gordon, multistage and payout/ROE-driven dividend discount
  models (with a two-stage closed form); PVGO; FCFF and FCFE (both routes) with DCF valuation and
  the equity bridge; justified forward, trailing and FCFE P/E; PEG; peer-multiple valuation
  (P/E, P/B, P/S, P/CF, EV/EBITDA).
- **Technical analysis** (`technical`): EMA, Bollinger bands, MACD (lecture SMA signal line or EMA),
  RSI (simple or Wilder), support/resistance, dated signal log, latest-readings table, candlestick
  patterns (hammer, hanging man, doji, engulfing, harami, morning/evening star), volume confirmation, TRIN.
- **Performance measurement** (`performance`): Sharpe, Treynor, Jensen's alpha with t-statistic,
  information ratio, tracking error, M², systematic/unsystematic risk and rankings, from return series
  (daily/weekly/monthly) or from summary statistics; Treynor-Mazuy and Henriksson-Merton market timing.
- **Data:** daily OHLCV for one ticker from Yahoo Finance or CSV (`fetch_ohlcv`, `load_ohlcv_csv`).
- **Charts:** technical three-panel chart, candlesticks with patterns, SMA trend chart, DCF value
  breakdown, security market line, market-timing scatter, expected-returns and risk-contribution bars.
- **Dashboard:** Estimation, Trend, Technical, Valuation and Performance tabs (11 in total); Valuation
  and lecture-style Performance work without loading data; backtest strategies feed the Performance tab.
- Deployment notes for Streamlit Community Cloud; the app finds the package in `src/` when it is not installed.

### Rebuilt
- The v2.0 and v2.1 features (Ledoit-Wolf, Black-Litterman, inverse volatility, risk parity, maximum
  diversification, SMA indicators, cash in backtests, SMA trend-filter strategies) existed only in a
  development workspace and were never published. They are re-implemented here from the design
  specification in `docs/ARCHITECTURE.md`, with new tests (including the scikit-learn reference values).

### Changed
- Backtest strategies take a covariance method; `STRATEGY_FACTORIES` registry with 8 strategies.
- `BacktestConfig.cash_rate`; turnover counts risky assets only.
- In the app, a strategy that cannot start (e.g. SMA window longer than the lookback) is reported and
  the others still run.

### Compatibility
- All v1 public functions keep their signatures and default behaviour (the v1 test suite passes unchanged
  apart from the app tests, which now expect the new tabs).

## [1.0.0] — 2026-10-26

### Added
- Validated data layer (yfinance or CSV; missing prices dropped and reported, never filled).
- Risk metrics, CAPM regression, long-only capped min-variance / max-Sharpe optimisers,
  efficient frontier, walk-forward backtest with drift and transaction costs.
- Plotly charts and a Streamlit dashboard; 211 tests.
