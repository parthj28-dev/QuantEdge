"""Fundamental analysis: intrinsic value of equity (v3.0, SAPM lectures 22-25).

Every model discounts expected cash flows at a required rate k, or compares price
multiples with peers. Inputs are plain numbers (per-share amounts, rates as decimals),
so the functions work for any company and can be checked against textbook problems.

Absolute (DCF) models
    Zero growth             V0 = D / k
    Gordon (constant g)     V0 = D1 / (k - g),           D1 = D0 (1 + g),   needs k > g
    Multistage DDM          V0 = sum_{t=1}^{T} D_t / (1+k)^t  +  V_T / (1+k)^T,
                            V_T = D_{T+1} / (k - g_terminal)
    Earnings-driven DDM     E_t = E_{t-1} (1 + b_{t-1} ROE_{t-1}),  D_t = (1 - b_t) E_t
    PVGO                    P0 = E1 / k + PVGO
    FCFF                    EBIT (1 - t) + depreciation - capex - increase in NWC
    FCFE                    FCFF - interest (1 - t) + net borrowing
                            = NI - (capex - depreciation) - increase in WC + new debt - debt repaid
    FCFE / FCFF DCF         same discounting as the DDM (FCFE at cost of equity k,
                            FCFF at WACC, then subtract net debt for equity value)

Relative valuation
    Forward P/E   P0/E1 = (1 - b) / (k - b ROE)          Trailing P/E   P0/E0 = (1 - b)(1 + g) / (k - g)
    FCFE P/E      P0/E1 = (FCFE1/E1) / (k - g)           PEG = (P/E) / (100 g)
    Peer multiples: implied price = peer multiple x the company's own metric
    (EV/EBITDA: implied EV minus net debt, per share).

Signals (lecture convention): IV > price -> Buy, IV < price -> Sell, equal -> Hold.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from .exceptions import EstimationError

BUY, SELL, HOLD = "Buy", "Sell", "Hold"


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise EstimationError(f"{name} must be a finite number, got {value!r}.")
    return float(value)


def _rate(name: str, value: float) -> float:
    value = _finite(name, value)
    if value <= -1:
        raise EstimationError(f"{name} must be above -100 %, got {value:.2%}.")
    return value


def _k_above_g(k: float, g: float, what: str = "growth") -> None:
    if k <= g:
        raise EstimationError(
            f"The required return ({k:.2%}) must exceed the perpetual {what} rate ({g:.2%}); "
            "otherwise the present value of the growing perpetuity is infinite."
        )


# --------------------------------------------------------------------------- #
# Returns, required return and signals
# --------------------------------------------------------------------------- #
def holding_period_return(price_now: float, expected_dividend: float, expected_price: float) -> float:
    """HPR = (D1 + P1 - P0) / P0 = dividend yield + capital gains yield."""
    p0 = _finite("Current price", price_now)
    if p0 <= 0:
        raise EstimationError("Current price must be positive.")
    return (_finite("Dividend", expected_dividend) + _finite("Expected price", expected_price) - p0) / p0


def capm_required_return(rf: float, beta: float, market_premium: float) -> float:
    """k = rf + beta (E[R_m] - rf)."""
    return _rate("Risk-free rate", rf) + _finite("Beta", beta) * _finite("Market risk premium", market_premium)


def implied_required_return(price: float, next_dividend: float, growth: float) -> float:
    """From Gordon: k = D1 / P0 + g (dividend yield + capital gains yield)."""
    p0 = _finite("Price", price)
    if p0 <= 0:
        raise EstimationError("Price must be positive.")
    return _finite("Dividend", next_dividend) / p0 + _rate("Growth", growth)


def sustainable_growth(retention_ratio: float, roe: float) -> float:
    """g = b x ROE (plowback ratio times return on retained earnings)."""
    b = _finite("Retention ratio", retention_ratio)
    if not 0 <= b <= 1:
        raise EstimationError(f"Retention ratio must be between 0 and 1, got {b}.")
    return b * _finite("ROE", roe)


def valuation_signal(intrinsic_value: float, market_price: float, tolerance: float = 0.0) -> str:
    """Buy if IV > price, Sell if IV < price, Hold if equal (within an optional relative tolerance)."""
    iv, mp = _finite("Intrinsic value", intrinsic_value), _finite("Market price", market_price)
    if mp <= 0:
        raise EstimationError("Market price must be positive.")
    gap = iv / mp - 1
    if abs(gap) <= tolerance:
        return HOLD
    return BUY if gap > 0 else SELL


def return_signal(expected_return: float, required_return: float, tolerance: float = 0.0) -> str:
    """Buy if expected HPR > k, Sell if below, Hold if equal (fairly priced)."""
    gap = _finite("Expected return", expected_return) - _finite("Required return", required_return)
    if abs(gap) <= tolerance:
        return HOLD
    return BUY if gap > 0 else SELL


# --------------------------------------------------------------------------- #
# Dividend discount models
# --------------------------------------------------------------------------- #
def zero_growth_value(dividend: float, k: float) -> float:
    """Perpetuity: V0 = D / k (e.g. a preferred share)."""
    k = _rate("Required return", k)
    _k_above_g(k, 0.0)
    return _finite("Dividend", dividend) / k


def gordon_value(k: float, growth: float, *, d0: float | None = None, d1: float | None = None) -> float:
    """Constant-growth DDM: V0 = D1 / (k - g). Give the dividend just paid (d0) OR the next one (d1)."""
    k, g = _rate("Required return", k), _rate("Growth", growth)
    _k_above_g(k, g)
    if (d0 is None) == (d1 is None):
        raise EstimationError("Give exactly one of d0 (just paid) or d1 (next year's dividend).")
    next_div = _finite("D1", d1) if d1 is not None else _finite("D0", d0) * (1 + g)
    return next_div / (k - g)


@dataclass(frozen=True)
class DCFResult:
    """Explicit-period cash flows, terminal value and their present values."""

    value: float
    schedule: pd.DataFrame        # index t = 1..T: cash_flow, growth, discount_factor, present_value
    terminal_value: float         # V_T at the end of year T
    pv_terminal: float
    terminal_cash_flow: float     # CF_{T+1}
    terminal_growth: float
    discount_rate: float

    @property
    def pv_explicit(self) -> float:
        return float(self.schedule["present_value"].sum())

    @property
    def terminal_share(self) -> float:
        """Fraction of the value that comes from the terminal value."""
        return self.pv_terminal / self.value if self.value else math.nan

    def value_at(self, year: int) -> float:
        """Intrinsic value at the end of `year` (0 <= year <= T): PV at that date of all later cash flows."""
        T = len(self.schedule)
        if not 0 <= year <= T:
            raise EstimationError(f"year must be between 0 and {T}.")
        k = self.discount_rate
        later = self.schedule.loc[year + 1:, "cash_flow"]
        pv = sum(cf / (1 + k) ** (t - year) for t, cf in later.items())
        return pv + self.terminal_value / (1 + k) ** (T - year)


def _discount(cash_flows: Sequence[float], growth: Sequence[float], k: float,
              terminal_cf: float, terminal_g: float) -> DCFResult:
    _k_above_g(k, terminal_g)
    T = len(cash_flows)
    t = pd.RangeIndex(1, T + 1, name="year")
    df = pd.Series([(1 + k) ** -i for i in t], index=t)
    schedule = pd.DataFrame({"cash_flow": list(cash_flows), "growth": list(growth),
                             "discount_factor": df}, index=t)
    schedule["present_value"] = schedule["cash_flow"] * schedule["discount_factor"]
    terminal_value = terminal_cf / (k - terminal_g)
    pv_terminal = terminal_value * (1 + k) ** -T if T else terminal_value
    value = float(schedule["present_value"].sum()) + pv_terminal
    return DCFResult(value, schedule, terminal_value, pv_terminal, terminal_cf, terminal_g, k)


def multistage_ddm(d0: float, growth_rates: Sequence[float], terminal_growth: float, k: float) -> DCFResult:
    """D_t = D_{t-1} (1 + g_t) for each listed year, then constant growth forever.

    Two-stage: growth_rates = [g1] * N. growth_rates = [] reduces to the Gordon model.
    """
    d = _finite("D0", d0)
    k, g_term = _rate("Required return", k), _rate("Terminal growth", terminal_growth)
    dividends, growth = [], []
    for i, g in enumerate(growth_rates, start=1):
        g = _rate(f"Growth in year {i}", g)
        d *= 1 + g
        dividends.append(d)
        growth.append(g)
    return _discount(dividends, growth, k, d * (1 + g_term), g_term)


def two_stage_closed_form(d0: float, g1: float, years: int, g2: float, k: float) -> float:
    """Growing annuity for N years plus a growing perpetuity from N+1 (lecture formula):

        V0 = D1/(k-g1) [1 - ((1+g1)/(1+k))^N] + D_{N+1} / ((k-g2)(1+k)^N)

    Needs k != g1 (for k = g1 the annuity term is N D0 / (1 + k)).
    """
    d0, g1, g2, k = _finite("D0", d0), _rate("g1", g1), _rate("g2", g2), _rate("k", k)
    _k_above_g(k, g2)
    if abs(k - g1) < 1e-14:
        annuity = years * d0
    else:
        annuity = d0 * (1 + g1) / (k - g1) * (1 - ((1 + g1) / (1 + k)) ** years)
    return annuity + d0 * (1 + g1) ** years * (1 + g2) / ((k - g2) * (1 + k) ** years)


@dataclass(frozen=True)
class EarningsStage:
    """`years` years with this payout ratio and return on equity (on retained earnings)."""

    years: int
    payout: float
    roe: float


def earnings_ddm(eps0: float, stages: Sequence[EarningsStage], terminal_payout: float, terminal_roe: float,
                 k: float, *, payout0: float | None = None) -> DCFResult:
    """DDM driven by payout and ROE: earnings grow by what was retained the year before.

        E_t = E_{t-1} (1 + b_{t-1} ROE_{t-1}),   D_t = (1 - b_t) E_t,   b = 1 - payout
        terminal:  g = b_terminal x ROE_terminal from year T+1 on

    `payout0` is the payout ratio applied at t = 0 (defaults to the first stage's payout),
    which sets the growth from year 0 to year 1. A payout change therefore changes the
    dividend immediately and the growth rate one year later, as in lecture 22, problem 2.
    """
    e = _finite("EPS0", eps0)
    k = _rate("Required return", k)
    plan = [(s.payout, s.roe) for s in stages for _ in range(int(s.years))]
    for pay in [p for p, _ in plan] + [terminal_payout] + ([] if payout0 is None else [payout0]):
        if not 0 <= _finite("Payout ratio", pay) <= 1:
            raise EstimationError(f"Payout ratios must be between 0 and 1, got {pay}.")
    if payout0 is not None:
        prev_payout = payout0
    else:
        prev_payout = plan[0][0] if plan else terminal_payout
    prev_roe = plan[0][1] if plan else terminal_roe
    dividends, growth = [], []
    for pay, roe in plan:
        g = (1 - prev_payout) * prev_roe
        e *= 1 + g
        dividends.append(pay * e)
        growth.append(g)
        prev_payout, prev_roe = pay, roe
    e_next = e * (1 + (1 - prev_payout) * prev_roe)
    g_term = sustainable_growth(1 - terminal_payout, terminal_roe)
    return _discount(dividends, growth, k, terminal_payout * e_next, g_term)


def pvgo(price: float, next_eps: float, k: float) -> float:
    """Present value of growth opportunities: PVGO = P0 - E1 / k (negative when ROE < k)."""
    k = _rate("Required return", k)
    _k_above_g(k, 0.0)
    return _finite("Price", price) - _finite("E1", next_eps) / k


# --------------------------------------------------------------------------- #
# Free cash flows
# --------------------------------------------------------------------------- #
def fcff(ebit: float, tax_rate: float, depreciation: float, capex: float, increase_nwc: float) -> float:
    """Free cash flow to the firm = EBIT (1 - t) + depreciation - capex - increase in NWC."""
    t = _finite("Tax rate", tax_rate)
    if not 0 <= t < 1:
        raise EstimationError("Tax rate must be between 0 and 100 %.")
    return (_finite("EBIT", ebit) * (1 - t) + _finite("Depreciation", depreciation)
            - _finite("Capex", capex) - _finite("Increase in NWC", increase_nwc))


def fcfe_from_fcff(fcff_value: float, interest: float, tax_rate: float, net_borrowing: float) -> float:
    """FCFE = FCFF - interest (1 - t) + increase in net debt."""
    t = _finite("Tax rate", tax_rate)
    return _finite("FCFF", fcff_value) - _finite("Interest", interest) * (1 - t) + _finite("Net borrowing", net_borrowing)


def fcfe_from_net_income(net_income: float, capex: float, depreciation: float, increase_wc: float,
                         new_debt: float = 0.0, debt_repaid: float = 0.0) -> float:
    """FCFE = NI - (capex - depreciation) - increase in WC + new debt issued - debt repaid."""
    return (_finite("Net income", net_income) - (_finite("Capex", capex) - _finite("Depreciation", depreciation))
            - _finite("Increase in WC", increase_wc) + _finite("New debt", new_debt) - _finite("Debt repaid", debt_repaid))


def dcf_value(cash_flows: Sequence[float], terminal_growth: float, discount_rate: float) -> DCFResult:
    """Discount explicit cash flows CF_1..CF_T and a terminal value CF_T (1 + g) / (r - g).

    With FCFE use the cost of equity; with FCFF use the WACC (then `equity_from_firm_value`).
    For a pure constant-growth valuation with no explicit period use `constant_growth_dcf`.
    """
    if not cash_flows:
        raise EstimationError("Give at least one explicit-period cash flow.")
    r, g = _rate("Discount rate", discount_rate), _rate("Terminal growth", terminal_growth)
    cfs = [_finite(f"Cash flow {i}", c) for i, c in enumerate(cash_flows, start=1)]
    growth = [math.nan] + [b / a - 1 if a else math.nan for a, b in zip(cfs[:-1], cfs[1:])]
    return _discount(cfs, growth, r, cfs[-1] * (1 + g), g)


def constant_growth_dcf(next_cash_flow: float, growth: float, discount_rate: float) -> float:
    """V0 = CF_1 / (r - g) (FCFE at cost of equity, or FCFF at WACC)."""
    r, g = _rate("Discount rate", discount_rate), _rate("Growth", growth)
    _k_above_g(r, g)
    return _finite("CF1", next_cash_flow) / (r - g)


def equity_from_firm_value(firm_value: float, net_debt: float, shares: float) -> float:
    """Per-share equity value = (firm value - net debt) / shares outstanding."""
    n = _finite("Shares outstanding", shares)
    if n <= 0:
        raise EstimationError("Shares outstanding must be positive.")
    return (_finite("Firm value", firm_value) - _finite("Net debt", net_debt)) / n


# --------------------------------------------------------------------------- #
# Relative valuation
# --------------------------------------------------------------------------- #
def forward_pe(retention_ratio: float, roe: float, k: float) -> float:
    """Justified P0/E1 = (1 - b) / (k - b ROE)."""
    g = sustainable_growth(retention_ratio, roe)
    k = _rate("Required return", k)
    _k_above_g(k, g)
    return (1 - retention_ratio) / (k - g)


def trailing_pe(retention_ratio: float, roe: float, k: float) -> float:
    """Justified P0/E0 = (1 - b)(1 + g) / (k - g), g = b ROE."""
    g = sustainable_growth(retention_ratio, roe)
    return forward_pe(retention_ratio, roe, k) * (1 + g)


def fcfe_pe(fcfe_to_earnings: float, growth: float, k: float) -> float:
    """Justified P0/E1 from the FCFE model: (FCFE1 / E1) / (k - g)."""
    k, g = _rate("Required return", k), _rate("Growth", growth)
    _k_above_g(k, g)
    return _finite("FCFE/E", fcfe_to_earnings) / (k - g)


def peg_ratio(pe: float, eps_growth: float) -> float:
    """PEG = P/E divided by expected EPS growth in percent (e.g. 20 / 15 = 1.33)."""
    g = _finite("EPS growth", eps_growth)
    if g <= 0:
        raise EstimationError("PEG is undefined for zero or negative growth.")
    return _finite("P/E", pe) / (g * 100)


#: multiple name -> the per-share (or firm-level for EV) metric it multiplies
MULTIPLES = {
    "P/E": "eps",
    "P/B": "book_value_per_share",
    "P/S": "sales_per_share",
    "P/CF": "cash_flow_per_share",
    "EV/EBITDA": "ebitda",
}


def relative_valuation(company: Mapping[str, float], peer_multiples: Mapping[str, float],
                       market_price: float | None = None) -> pd.DataFrame:
    """Implied price per share from each peer (or industry) multiple.

    `company` holds the metrics named in MULTIPLES; EV/EBITDA also needs `net_debt` and
    `shares` (implied price = (multiple x EBITDA - net debt) / shares). Multiples whose
    metric is missing or non-positive are reported as not applicable, never filled.
    """
    rows = {}
    for name, multiple in peer_multiples.items():
        if name not in MULTIPLES:
            raise EstimationError(f"Unknown multiple {name!r}; use one of {list(MULTIPLES)}.")
        metric = company.get(MULTIPLES[name])
        m = _finite(f"{name} multiple", multiple)
        if metric is None or not math.isfinite(metric) or metric <= 0:
            rows[name] = {"peer_multiple": m, "company_metric": metric, "implied_price": math.nan,
                          "note": f"needs a positive {MULTIPLES[name]}"}
            continue
        if name == "EV/EBITDA":
            if company.get("shares") is None or company.get("net_debt") is None:
                raise EstimationError("EV/EBITDA needs the company's net_debt and shares.")
            price = equity_from_firm_value(m * metric, company["net_debt"], company["shares"])
        else:
            price = m * metric
        rows[name] = {"peer_multiple": m, "company_metric": metric, "implied_price": price, "note": ""}
    table = pd.DataFrame(rows).T
    table["implied_price"] = table["implied_price"].astype(float)
    if market_price is not None:
        mp = _finite("Market price", market_price)
        table["upside"] = table["implied_price"] / mp - 1
        table["signal"] = [valuation_signal(p, mp) if math.isfinite(p) else None for p in table["implied_price"]]
    return table
