"""QuantEdge — Portfolio Risk & Optimization Lab.

Layered package; lower layers never import higher ones:
    config / exceptions -> data -> metrics -> indicators -> capm / optimize / estimators / riskbased
    -> frontier -> backtest -> valuation / technical / performance -> charts
The Streamlit UI lives outside the package in app/ and only orchestrates.
"""

__version__ = "3.0.0"
