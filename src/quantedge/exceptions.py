"""Error types.

Having our own exception classes lets the UI catch *expected* problems
(bad tickers, too little data, infeasible constraints) and show a friendly
message, while genuine programming bugs still surface as ordinary exceptions.
"""

from __future__ import annotations


class QuantEdgeError(Exception):
    """Base class for all expected, user-facing errors in QuantEdge."""


class DataError(QuantEdgeError):
    """Price data is missing, malformed or invalid (e.g. negative prices)."""


class InsufficientDataError(DataError):
    """Data is valid but too short for the requested calculation."""


class OptimizationError(QuantEdgeError):
    """An optimisation problem is infeasible or the solver failed."""
