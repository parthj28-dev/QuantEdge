"""QuantEdge — Portfolio Risk & Optimization Lab.

Layered package; lower layers never import higher ones:
    config / exceptions -> data -> metrics -> capm / optimize / frontier -> backtest -> charts
The Streamlit UI lives outside the package in app/ and only orchestrates.
"""

__version__ = "0.1.0"
