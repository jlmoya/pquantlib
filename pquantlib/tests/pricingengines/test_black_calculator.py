"""Tests for BlackCalculator (Black 1976 Greeks calculator).

# C++ parity: ql/pricingengines/blackcalculator.{hpp,cpp} @ v1.42.1.

Cross-validated against the textbook BSM parameters embedded in
``migration-harness/references/cluster/l3d.json`` (``analytic_european``
section — the BlackCalculator inputs are the same as the engine's).
"""

from __future__ import annotations

import math
from math import erfc
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.payoffs import (
    AssetOrNothingPayoff,
    CashOrNothingPayoff,
    GapPayoff,
    OptionType,
    PlainVanillaPayoff,
    SuperSharePayoff,
)
from pquantlib.pricingengines.black_calculator import BlackCalculator
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import tight


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/l3d")


# Textbook BSM @ (S=100, K=100, r=5%, q=2%, sigma=20%, T=1) — these
# inputs are what the AnalyticEuropeanEngine probe uses internally.
# forward = S * exp((r-q)*T) = 100 * exp(0.03)
# discount = exp(-r*T) = exp(-0.05)
# stdDev = sigma * sqrt(T) = 0.20


def _make_call_calculator() -> BlackCalculator:
    spot = 100.0
    strike = 100.0
    r = 0.05
    q = 0.02
    sigma = 0.20
    t = 1.0
    forward = spot * math.exp((r - q) * t)
    df = math.exp(-r * t)
    payoff = PlainVanillaPayoff(OptionType.Call, strike)
    return BlackCalculator(payoff, forward, sigma * math.sqrt(t), df)


def _make_put_calculator() -> BlackCalculator:
    spot = 100.0
    strike = 100.0
    r = 0.05
    q = 0.02
    sigma = 0.20
    t = 1.0
    forward = spot * math.exp((r - q) * t)
    df = math.exp(-r * t)
    payoff = PlainVanillaPayoff(OptionType.Put, strike)
    return BlackCalculator(payoff, forward, sigma * math.sqrt(t), df)


# --- value -----------------------------------------------------------------


def test_call_value_matches_textbook(reference_data: dict[str, Any]) -> None:
    bc = _make_call_calculator()
    tight(bc.value(), float(reference_data["analytic_european"]["call_npv"]))


def test_put_value_matches_textbook(reference_data: dict[str, Any]) -> None:
    bc = _make_put_calculator()
    tight(bc.value(), float(reference_data["analytic_european"]["put_npv"]))


# --- delta -----------------------------------------------------------------


def test_call_delta_matches_textbook(reference_data: dict[str, Any]) -> None:
    bc = _make_call_calculator()
    tight(bc.delta(100.0), float(reference_data["analytic_european"]["call_delta"]))


def test_put_delta_matches_textbook(reference_data: dict[str, Any]) -> None:
    bc = _make_put_calculator()
    tight(bc.delta(100.0), float(reference_data["analytic_european"]["put_delta"]))


# --- gamma -----------------------------------------------------------------


def test_call_gamma_matches_textbook(reference_data: dict[str, Any]) -> None:
    bc = _make_call_calculator()
    tight(bc.gamma(100.0), float(reference_data["analytic_european"]["call_gamma"]))


def test_put_gamma_matches_textbook(reference_data: dict[str, Any]) -> None:
    bc = _make_put_calculator()
    # gamma is the same for call and put.
    tight(bc.gamma(100.0), float(reference_data["analytic_european"]["put_gamma"]))


# --- vega ------------------------------------------------------------------


def test_call_vega_matches_textbook(reference_data: dict[str, Any]) -> None:
    bc = _make_call_calculator()
    tight(bc.vega(1.0), float(reference_data["analytic_european"]["call_vega"]))


def test_put_vega_matches_textbook(reference_data: dict[str, Any]) -> None:
    bc = _make_put_calculator()
    # vega is the same for call and put.
    tight(bc.vega(1.0), float(reference_data["analytic_european"]["put_vega"]))


# --- rho / dividend_rho ----------------------------------------------------


def test_call_rho_matches_textbook(reference_data: dict[str, Any]) -> None:
    bc = _make_call_calculator()
    tight(bc.rho(1.0), float(reference_data["analytic_european"]["call_rho"]))


def test_put_rho_matches_textbook(reference_data: dict[str, Any]) -> None:
    bc = _make_put_calculator()
    tight(bc.rho(1.0), float(reference_data["analytic_european"]["put_rho"]))


def test_call_dividend_rho_matches_textbook(reference_data: dict[str, Any]) -> None:
    bc = _make_call_calculator()
    tight(bc.dividend_rho(1.0), float(reference_data["analytic_european"]["call_dividend_rho"]))


def test_put_dividend_rho_matches_textbook(reference_data: dict[str, Any]) -> None:
    bc = _make_put_calculator()
    tight(bc.dividend_rho(1.0), float(reference_data["analytic_european"]["put_dividend_rho"]))


# --- theta -----------------------------------------------------------------


def test_call_theta_matches_textbook(reference_data: dict[str, Any]) -> None:
    bc = _make_call_calculator()
    tight(bc.theta(100.0, 1.0), float(reference_data["analytic_european"]["call_theta"]))


def test_put_theta_matches_textbook(reference_data: dict[str, Any]) -> None:
    bc = _make_put_calculator()
    tight(bc.theta(100.0, 1.0), float(reference_data["analytic_european"]["put_theta"]))


# --- itm cash probability --------------------------------------------------


def test_call_itm_cash_probability_matches_textbook(reference_data: dict[str, Any]) -> None:
    bc = _make_call_calculator()
    tight(
        bc.itm_cash_probability(),
        float(reference_data["analytic_european"]["call_itm_cash_probability"]),
    )


# --- payoff dispatch (Visitor → isinstance) --------------------------------


def test_cash_or_nothing_call_value_is_cash_times_n_d2() -> None:
    """CashOrNothing call: value = discount * cash * N(d2)."""
    payoff = CashOrNothingPayoff(OptionType.Call, 100.0, 5.0)
    forward = 100.0 * math.exp(0.03)
    df = math.exp(-0.05)
    bc = BlackCalculator(payoff, forward, 0.20, df)
    # We verify alpha was zeroed by checking value uses x=5.0 (cash payoff).
    # value = df * (forward * 0 + cash * cum_d2). The shape is right;
    # exact reference comes from the formula.
    # cum_d2 for our setup is in the call_itm_cash_probability.
    expected_cum_d2 = 0.5199388058383726  # from probe JSON
    tight(bc.value(), df * 5.0 * expected_cum_d2)


def test_asset_or_nothing_call_value() -> None:
    """AssetOrNothing call: value = discount * forward * N(d1)."""
    payoff = AssetOrNothingPayoff(OptionType.Call, 100.0)
    forward = 100.0 * math.exp(0.03)
    df = math.exp(-0.05)
    bc = BlackCalculator(payoff, forward, 0.20, df)
    # value = df * forward * cum_d1.
    # d1 = ln(forward/strike)/stdDev + 0.5*stdDev
    d1 = math.log(forward / 100.0) / 0.20 + 0.1
    cum_d1 = 0.5 * erfc(-d1 / math.sqrt(2.0))
    tight(bc.value(), df * forward * cum_d1)


def test_gap_payoff_uses_second_strike() -> None:
    """Gap payoff: value uses second_strike as x — discount * (forward * alpha + x * beta)."""
    payoff = GapPayoff(OptionType.Call, 100.0, 110.0)
    forward = 100.0 * math.exp(0.03)
    df = math.exp(-0.05)
    bc = BlackCalculator(payoff, forward, 0.20, df)
    # Reference: plain-vanilla call at strike=100 with the strike replaced
    # by second_strike=110 in the K * N(d2) term.
    d1 = math.log(forward / 100.0) / 0.20 + 0.1
    d2 = d1 - 0.20
    cum_d1 = 0.5 * erfc(-d1 / math.sqrt(2.0))
    cum_d2 = 0.5 * erfc(-d2 / math.sqrt(2.0))
    expected = df * (forward * cum_d1 - 110.0 * cum_d2)
    tight(bc.value(), expected)


# --- factory ---------------------------------------------------------------


def test_from_type_strike_classmethod() -> None:
    """``BlackCalculator.from_type_strike`` builds the same calculator as
    the payoff-based constructor."""
    forward = 100.0 * math.exp(0.03)
    df = math.exp(-0.05)
    bc_factory = BlackCalculator.from_type_strike(OptionType.Call, 100.0, forward, 0.20, df)
    bc_direct = _make_call_calculator()
    tight(bc_factory.value(), bc_direct.value())
    tight(bc_factory.delta(100.0), bc_direct.delta(100.0))


# --- edge cases ------------------------------------------------------------


def test_zero_volatility_call_at_atm() -> None:
    """At zero vol and ATM, a call has delta = discount * 0.5 (spot delta)."""
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    df = math.exp(-0.05)
    bc = BlackCalculator(payoff, 100.0, 0.0, df)
    tight(bc.delta(100.0), df * 0.5)
    # Gamma is zero with no volatility.
    tight(bc.gamma(100.0), 0.0)
    # Vega is zero.
    tight(bc.vega(1.0), 0.0)


def test_negative_spot_raises() -> None:
    """delta(spot <= 0) must raise."""
    bc = _make_call_calculator()
    with pytest.raises(LibraryException, match="positive spot value required"):
        bc.delta(-1.0)


def test_negative_maturity_raises() -> None:
    """vega / rho / dividend_rho with negative maturity must raise."""
    bc = _make_call_calculator()
    with pytest.raises(LibraryException, match="negative maturity not allowed"):
        bc.vega(-1.0)


def test_unsupported_payoff_raises() -> None:
    """A payoff that's not one of the four supported types must raise."""
    payoff = SuperSharePayoff(strike=90.0, second_strike=110.0, cash_payoff=1.0)
    forward = 100.0 * math.exp(0.03)
    df = math.exp(-0.05)
    with pytest.raises(LibraryException, match="unsupported payoff type"):
        BlackCalculator(payoff, forward, 0.20, df)


# --- vanna / volga ---------------------------------------------------------


def test_vanna_atm_call_is_finite() -> None:
    bc = _make_call_calculator()
    v = bc.vanna(100.0, 1.0)
    assert math.isfinite(v)


def test_volga_atm_call_is_finite() -> None:
    bc = _make_call_calculator()
    v = bc.volga(1.0)
    assert math.isfinite(v)


# --- forward / asset probability accessors --------------------------------


def test_call_itm_asset_probability_is_n_d1() -> None:
    """itm_asset_probability == cum_d1 (call). Loosely verified by
    checking it's > cum_d2 for the standard ITM-call setup."""
    bc = _make_call_calculator()
    assert bc.itm_asset_probability() > bc.itm_cash_probability()


# --- alpha / beta ----------------------------------------------------------


def test_call_alpha_is_positive() -> None:
    """For Call: alpha = N(d1) >= 0, beta = -N(d2) <= 0."""
    bc = _make_call_calculator()
    assert bc.alpha() > 0
    assert bc.beta() < 0


def test_put_alpha_is_negative() -> None:
    """For Put: alpha = N(d1) - 1 <= 0, beta = 1 - N(d2) >= 0."""
    bc = _make_put_calculator()
    assert bc.alpha() < 0
    assert bc.beta() > 0
