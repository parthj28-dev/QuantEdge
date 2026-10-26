"""Project-wide constants and conventions.

Every annualisation, tolerance and default lives here so that the whole code base
uses one consistent set of assumptions. See docs/ARCHITECTURE.md, Section 3.
"""

from __future__ import annotations

# Number of trading days used to annualise daily statistics.
# 252 is the common convention (US markets). Indian markets trade ~248-250 days;
# the difference changes annualised numbers by well under 1 %.
TRADING_DAYS_PER_YEAR: int = 252

# Minimum number of daily *return* observations required before we compute
# statistics for the dashboard. Mathematically 2 is enough for a standard
# deviation, but estimates from a few weeks of data are meaningless.
DEFAULT_MIN_OBSERVATIONS: int = 60

# A daily standard deviation below this is treated as zero (a constant series).
# Floating-point arithmetic rarely gives exactly 0.0, hence a tolerance.
VOLATILITY_EPSILON: float = 1e-12

# Default annual risk-free rate (effective annual yield, as a decimal).
DEFAULT_RISK_FREE_RATE: float = 0.0

# Price column requested from Yahoo Finance: adjusted for splits and dividends.
ADJUSTED_PRICE_FIELD: str = "Adj Close"
