"""Tests for quantedge.valuation, checked against the worked examples in SAPM lectures 22-25."""

from __future__ import annotations

import math

import pytest

from quantedge.exceptions import EstimationError
from quantedge.valuation import (
    BUY,
    HOLD,
    SELL,
    EarningsStage,
    capm_required_return,
    constant_growth_dcf,
    dcf_value,
    earnings_ddm,
    equity_from_firm_value,
    fcfe_from_fcff,
    fcfe_from_net_income,
    fcfe_pe,
    fcff,
    forward_pe,
    gordon_value,
    holding_period_return,
    implied_required_return,
    multistage_ddm,
    peg_ratio,
    pvgo,
    relative_valuation,
    return_signal,
    sustainable_growth,
    trailing_pe,
    two_stage_closed_form,
    valuation_signal,
    zero_growth_value,
)


# --------------------------------------------------------------------------- #
# Lecture examples
# --------------------------------------------------------------------------- #
def test_preferred_stock_zero_growth():                      # L23: $2 at 8 % -> $25
    assert zero_growth_value(2, 0.08) == pytest.approx(25.0)


def test_big_d_constant_growth():                            # L22: D0 0.50, g 2 %, R 15 % -> 3.92
    assert gordon_value(0.15, 0.02, d0=0.50) == pytest.approx(3.923, abs=1e-3)


def test_constant_growth_capm_rate():                        # L23: D0 3, g 8 %, k 14 % -> 54
    assert gordon_value(0.14, 0.08, d0=3) == pytest.approx(54.0)
    assert gordon_value(0.14, 0.08, d1=3.24) == pytest.approx(54.0)


def test_differential_growth_example_and_closed_form():      # L22: $2, 8 % x3, then 4 %, 12 % -> 28.89
    res = multistage_ddm(2, [0.08] * 3, 0.04, 0.12)
    assert res.value == pytest.approx(28.89, abs=0.01)
    assert two_stage_closed_form(2, 0.08, 3, 0.04, 0.12) == pytest.approx(res.value, rel=1e-12)
    assert res.pv_explicit == pytest.approx(5.58, abs=0.01) and res.pv_terminal == pytest.approx(23.31, abs=0.01)


def test_high_growth_then_mature():                          # L23: D0 2, 20 % x3, then 5 %, 15 % -> ~30.42
    res = multistage_ddm(2.0, [0.20] * 3, 0.05, 0.15)
    assert list(res.schedule["cash_flow"].round(2)) == [2.40, 2.88, 3.46]
    assert res.value == pytest.approx(30.42, abs=0.03)       # slide rounds dividends to cents


def test_problem_1_reema():                                  # L22 P1: answer (c) ~168, sell
    res = multistage_ddm(6.0, [0.08] * 3, 0.06, 0.10)
    assert round(res.value) == 168
    assert valuation_signal(res.value, 200) == SELL


def test_problem_2_payout_change():                          # L22 P2: payout 25 % -> 50 % after year 4
    res = earnings_ddm(20, [EarningsStage(4, 0.25, 0.20)], terminal_payout=0.50, terminal_roe=0.20, k=0.30)
    d = list(res.schedule["cash_flow"])
    assert d[0] == pytest.approx(5 * 1.15)                    # g = 0.75 x 20 % for years 1-4
    assert res.terminal_cash_flow == pytest.approx(0.5 * 20 * 1.15**5)   # D5 = 50 % of E5
    assert res.terminal_growth == pytest.approx(0.10)
    assert res.value == pytest.approx(50.07, abs=0.01)
    assert valuation_signal(res.value, 80) == SELL


def test_earnings_ddm_with_constant_policy_equals_gordon():
    res = earnings_ddm(20, [EarningsStage(3, 0.25, 0.2)], 0.25, 0.2, 0.30)
    assert res.value == pytest.approx(gordon_value(0.30, 0.15, d0=5))


def test_pvgo_example():                                     # L23: 60 % at ROE 10 %, k 15 %, D1 2, E1 5
    g = sustainable_growth(0.6, 0.10)
    price = gordon_value(0.15, g, d1=2)
    assert price == pytest.approx(22.22, abs=0.005)
    assert pvgo(price, 5, 0.15) == pytest.approx(-11.11, abs=0.005)


def test_pe_example():                                       # L25: ROE 15 %, b 60 %, EPS 2.5, k 12.5 %
    assert forward_pe(0.6, 0.15, 0.125) == pytest.approx(11.43, abs=0.005)
    g = sustainable_growth(0.6, 0.15)
    assert gordon_value(0.125, g, d1=2.5 * (1 + g) * 0.4) == pytest.approx(31.14, abs=0.005)
    assert trailing_pe(0.6, 0.15, 0.125) == pytest.approx(forward_pe(0.6, 0.15, 0.125) * 1.09)


def test_dividend_yield_plus_growth_and_hpr():
    assert implied_required_return(54, 3.24, 0.08) == pytest.approx(0.14)
    assert holding_period_return(100, 4, 110) == pytest.approx(0.14)
    assert capm_required_return(0.06, 1.2, 0.07) == pytest.approx(0.144)
    assert return_signal(0.14, 0.12) == BUY and return_signal(0.10, 0.12) == SELL
    assert return_signal(0.12, 0.12) == HOLD


# --------------------------------------------------------------------------- #
# Identities and edge cases
# --------------------------------------------------------------------------- #
def test_multistage_with_no_high_growth_is_gordon():
    assert multistage_ddm(3, [], 0.05, 0.11).value == pytest.approx(gordon_value(0.11, 0.05, d0=3))


def test_value_at_future_date_grows_at_k_minus_dividend():
    res = multistage_ddm(2, [0.2, 0.15, 0.1], 0.04, 0.12)
    v0, v1 = res.value, res.value_at(1)
    d1 = res.schedule.loc[1, "cash_flow"]
    assert (v1 + d1) / v0 == pytest.approx(1.12)              # total return = k
    assert res.value_at(3) == pytest.approx(res.terminal_value)


def test_k_must_exceed_g():
    with pytest.raises(EstimationError, match="must exceed"):
        gordon_value(0.08, 0.08, d0=1)
    with pytest.raises(EstimationError, match="must exceed"):
        multistage_ddm(1, [0.3], 0.12, 0.10)


@pytest.mark.parametrize("kwargs", [{}, {"d0": 1, "d1": 1}])
def test_gordon_needs_exactly_one_dividend(kwargs):
    with pytest.raises(EstimationError):
        gordon_value(0.1, 0.05, **kwargs)


@pytest.mark.parametrize("bad", [math.nan, math.inf, "5", None])
def test_non_numbers_rejected(bad):
    with pytest.raises(EstimationError):
        zero_growth_value(bad, 0.1)


def test_two_stage_k_equal_g1():
    v = two_stage_closed_form(2, 0.10, 4, 0.03, 0.10)
    assert v == pytest.approx(multistage_ddm(2, [0.10] * 4, 0.03, 0.10).value)


def test_signal_tolerance():
    assert valuation_signal(101, 100, tolerance=0.02) == HOLD
    assert valuation_signal(103, 100, tolerance=0.02) == BUY


# --------------------------------------------------------------------------- #
# Free cash flow
# --------------------------------------------------------------------------- #
def test_fcff_and_both_fcfe_routes_agree():
    ebit, t, dep, capex, dnwc, interest = 1000, 0.25, 200, 350, 50, 100
    ff = fcff(ebit, t, dep, capex, dnwc)
    assert ff == pytest.approx(750 + 200 - 350 - 50)
    net_borrowing = 80
    fe = fcfe_from_fcff(ff, interest, t, net_borrowing)
    ni = (ebit - interest) * (1 - t)
    assert fcfe_from_net_income(ni, capex, dep, dnwc, new_debt=120, debt_repaid=40) == pytest.approx(fe)


def test_dcf_value_terminal_and_equity_bridge():
    res = dcf_value([100, 110, 121], terminal_growth=0.03, discount_rate=0.10)
    manual = 100 / 1.1 + 110 / 1.1**2 + 121 / 1.1**3 + 121 * 1.03 / 0.07 / 1.1**3
    assert res.value == pytest.approx(manual)
    assert 0 < res.terminal_share < 1
    assert equity_from_firm_value(res.value, 200, 10) == pytest.approx((manual - 200) / 10)
    assert constant_growth_dcf(100, 0.03, 0.10) == pytest.approx(100 / 0.07)


def test_fcfe_pe_and_peg():
    assert fcfe_pe(0.6, 0.05, 0.12) == pytest.approx(0.6 / 0.07)
    assert peg_ratio(20, 0.15) == pytest.approx(20 / 15)
    with pytest.raises(EstimationError):
        peg_ratio(20, 0)


def test_relative_valuation_table():
    company = {"eps": 10, "book_value_per_share": 50, "sales_per_share": 0, "ebitda": 500,
               "net_debt": 1000, "shares": 100}
    t = relative_valuation(company, {"P/E": 15, "P/B": 2.5, "P/S": 1.2, "EV/EBITDA": 8}, market_price=140)
    assert t.loc["P/E", "implied_price"] == pytest.approx(150)
    assert t.loc["P/B", "implied_price"] == pytest.approx(125)
    assert math.isnan(t.loc["P/S", "implied_price"])          # zero sales: not applicable, not 0
    assert t.loc["EV/EBITDA", "implied_price"] == pytest.approx((8 * 500 - 1000) / 100)
    assert t.loc["P/E", "signal"] == BUY and t.loc["P/B", "signal"] == SELL
    with pytest.raises(EstimationError, match="Unknown multiple"):
        relative_valuation(company, {"P/X": 3})


def test_retention_ratio_bounds():
    with pytest.raises(EstimationError):
        sustainable_growth(1.2, 0.1)


def test_earnings_ddm_stage_change_hand_computed():
    # E0 = 10, payout0 = 20 %. Year 1: payout 20 %, ROE 20 %. Year 2: payout 60 %, ROE 10 %.
    # Terminal: payout 50 %, ROE 8 %  ->  g = 4 %.
    res = earnings_ddm(10, [EarningsStage(1, 0.2, 0.2), EarningsStage(1, 0.6, 0.1)], 0.5, 0.08, 0.12)
    e1 = 10 * (1 + 0.8 * 0.2)          # 11.6, grown by year-0 retention
    e2 = e1 * (1 + 0.8 * 0.2)          # year-1 retention (80 %) at year-1 ROE (20 %)
    e3 = e2 * (1 + 0.4 * 0.1)          # year-2 retention (40 %) at year-2 ROE (10 %)
    assert list(res.schedule["cash_flow"]) == pytest.approx([0.2 * e1, 0.6 * e2])
    assert res.terminal_cash_flow == pytest.approx(0.5 * e3)
    expected = 0.2 * e1 / 1.12 + 0.6 * e2 / 1.12**2 + 0.5 * e3 / (0.12 - 0.04) / 1.12**2
    assert res.value == pytest.approx(expected)
