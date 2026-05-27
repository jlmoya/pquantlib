"""Tests for AnalyticEuropeanEngine (closed-form European vanilla).

# C++ parity: ql/pricingengines/vanilla/analyticeuropeanengine.{hpp,cpp}
# @ v1.42.1.

Cross-validates against ``analytic_european`` section of
``migration-harness/references/cluster/l3d.json``.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, EuropeanExercise
from pquantlib.instruments.european_option import EuropeanOption
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.vanilla.analytic_european_engine import (
    AnalyticEuropeanEngine,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import tight
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/l3d")


def _build_process_and_dates() -> tuple[GeneralizedBlackScholesProcess, Date]:
    """Build the textbook BSM process and the 1-year expiry date."""
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.June, 2026)
    expiry = ref + 365  # 1y under Actual/365 Fixed

    spot_q = SimpleQuote(100.0)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.02, day_counter=dc)
    vol = BlackConstantVol(reference_date=ref, calendar=cal, day_counter=dc, volatility=0.20)
    process = GeneralizedBlackScholesProcess(
        x0=spot_q, dividend_ts=div, risk_free_ts=rf, black_vol_ts=vol
    )
    return process, expiry


# --- call --------------------------------------------------------------------


def test_call_npv_matches_textbook(reference_data: dict[str, Any]) -> None:
    process, expiry = _build_process_and_dates()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = EuropeanOption(payoff, exercise)
    opt.set_pricing_engine(AnalyticEuropeanEngine(process))
    tight(opt.npv(), float(reference_data["analytic_european"]["call_npv"]))


def test_call_delta_matches_textbook(reference_data: dict[str, Any]) -> None:
    process, expiry = _build_process_and_dates()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = EuropeanOption(payoff, exercise)
    opt.set_pricing_engine(AnalyticEuropeanEngine(process))
    tight(opt.delta(), float(reference_data["analytic_european"]["call_delta"]))


def test_call_gamma_matches_textbook(reference_data: dict[str, Any]) -> None:
    process, expiry = _build_process_and_dates()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = EuropeanOption(payoff, exercise)
    opt.set_pricing_engine(AnalyticEuropeanEngine(process))
    tight(opt.gamma(), float(reference_data["analytic_european"]["call_gamma"]))


def test_call_vega_matches_textbook(reference_data: dict[str, Any]) -> None:
    process, expiry = _build_process_and_dates()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = EuropeanOption(payoff, exercise)
    opt.set_pricing_engine(AnalyticEuropeanEngine(process))
    tight(opt.vega(), float(reference_data["analytic_european"]["call_vega"]))


def test_call_theta_matches_textbook(reference_data: dict[str, Any]) -> None:
    process, expiry = _build_process_and_dates()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = EuropeanOption(payoff, exercise)
    opt.set_pricing_engine(AnalyticEuropeanEngine(process))
    tight(opt.theta(), float(reference_data["analytic_european"]["call_theta"]))


def test_call_rho_matches_textbook(reference_data: dict[str, Any]) -> None:
    process, expiry = _build_process_and_dates()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = EuropeanOption(payoff, exercise)
    opt.set_pricing_engine(AnalyticEuropeanEngine(process))
    tight(opt.rho(), float(reference_data["analytic_european"]["call_rho"]))


def test_call_dividend_rho_matches_textbook(reference_data: dict[str, Any]) -> None:
    process, expiry = _build_process_and_dates()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = EuropeanOption(payoff, exercise)
    opt.set_pricing_engine(AnalyticEuropeanEngine(process))
    tight(opt.dividend_rho(), float(reference_data["analytic_european"]["call_dividend_rho"]))


def test_call_itm_cash_probability_matches_textbook(reference_data: dict[str, Any]) -> None:
    process, expiry = _build_process_and_dates()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = EuropeanOption(payoff, exercise)
    opt.set_pricing_engine(AnalyticEuropeanEngine(process))
    tight(
        opt.itm_cash_probability(),
        float(reference_data["analytic_european"]["call_itm_cash_probability"]),
    )


# --- put ---------------------------------------------------------------------


def test_put_npv_matches_textbook(reference_data: dict[str, Any]) -> None:
    process, expiry = _build_process_and_dates()
    payoff = PlainVanillaPayoff(OptionType.Put, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = EuropeanOption(payoff, exercise)
    opt.set_pricing_engine(AnalyticEuropeanEngine(process))
    tight(opt.npv(), float(reference_data["analytic_european"]["put_npv"]))


def test_put_delta_matches_textbook(reference_data: dict[str, Any]) -> None:
    process, expiry = _build_process_and_dates()
    payoff = PlainVanillaPayoff(OptionType.Put, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = EuropeanOption(payoff, exercise)
    opt.set_pricing_engine(AnalyticEuropeanEngine(process))
    tight(opt.delta(), float(reference_data["analytic_european"]["put_delta"]))


def test_put_gamma_matches_textbook(reference_data: dict[str, Any]) -> None:
    process, expiry = _build_process_and_dates()
    payoff = PlainVanillaPayoff(OptionType.Put, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = EuropeanOption(payoff, exercise)
    opt.set_pricing_engine(AnalyticEuropeanEngine(process))
    tight(opt.gamma(), float(reference_data["analytic_european"]["put_gamma"]))


def test_put_theta_matches_textbook(reference_data: dict[str, Any]) -> None:
    process, expiry = _build_process_and_dates()
    payoff = PlainVanillaPayoff(OptionType.Put, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = EuropeanOption(payoff, exercise)
    opt.set_pricing_engine(AnalyticEuropeanEngine(process))
    tight(opt.theta(), float(reference_data["analytic_european"]["put_theta"]))


def test_put_rho_matches_textbook(reference_data: dict[str, Any]) -> None:
    process, expiry = _build_process_and_dates()
    payoff = PlainVanillaPayoff(OptionType.Put, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = EuropeanOption(payoff, exercise)
    opt.set_pricing_engine(AnalyticEuropeanEngine(process))
    tight(opt.rho(), float(reference_data["analytic_european"]["put_rho"]))


def test_put_dividend_rho_matches_textbook(reference_data: dict[str, Any]) -> None:
    process, expiry = _build_process_and_dates()
    payoff = PlainVanillaPayoff(OptionType.Put, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = EuropeanOption(payoff, exercise)
    opt.set_pricing_engine(AnalyticEuropeanEngine(process))
    tight(opt.dividend_rho(), float(reference_data["analytic_european"]["put_dividend_rho"]))


# --- additional results -----------------------------------------------------


def test_additional_results_present() -> None:
    process, expiry = _build_process_and_dates()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = EuropeanOption(payoff, exercise)
    opt.set_pricing_engine(AnalyticEuropeanEngine(process))
    opt.npv()  # trigger calculation
    ar = opt.additional_results()
    # Mirror the C++ keys exactly.
    assert "spot" in ar
    assert "dividendDiscount" in ar
    assert "riskFreeDiscount" in ar
    assert "forward" in ar
    assert "strike" in ar
    assert "volatility" in ar
    assert "timeToExpiry" in ar
    tight(float(ar["spot"]), 100.0)
    tight(float(ar["strike"]), 100.0)
    tight(float(ar["volatility"]), 0.20)
    tight(float(ar["timeToExpiry"]), 1.0)


# --- error paths ------------------------------------------------------------


def test_american_exercise_rejected() -> None:
    """Engine should reject a VanillaOption with American exercise."""
    process, expiry = _build_process_and_dates()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    earliest = Date.from_ymd(15, Month.June, 2026)
    exercise = AmericanExercise(earliest, expiry)
    opt = VanillaOption(payoff, exercise)
    opt.set_pricing_engine(AnalyticEuropeanEngine(process))
    with pytest.raises(LibraryException, match="not a European option"):
        opt.npv()


def test_negative_spot_raises() -> None:
    """An invalid spot quote should yield an underlying error from value()."""
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.June, 2026)
    expiry = ref + 365
    # Construct with positive spot, then mutate to negative — BlackCalculator
    # will fail at value() with "forward must be positive".
    spot_q = SimpleQuote(100.0)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.02, day_counter=dc)
    vol = BlackConstantVol(reference_date=ref, calendar=cal, day_counter=dc, volatility=0.20)
    process = GeneralizedBlackScholesProcess(
        x0=spot_q, dividend_ts=div, risk_free_ts=rf, black_vol_ts=vol
    )
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = EuropeanOption(payoff, exercise)
    opt.set_pricing_engine(AnalyticEuropeanEngine(process))
    spot_q.set_value(-1.0)
    with pytest.raises(LibraryException, match="negative or null underlying"):
        opt.npv()


# --- separate discount curve ------------------------------------------------


def test_separate_discount_curve(reference_data: dict[str, Any]) -> None:
    """Passing a separate discount curve should reproduce the basic case
    when the curve matches the process's risk-free."""
    process, expiry = _build_process_and_dates()
    dc = Actual365Fixed()
    ref = Date.from_ymd(15, Month.June, 2026)
    discount = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = EuropeanOption(payoff, exercise)
    opt.set_pricing_engine(AnalyticEuropeanEngine(process, discount_curve=discount))
    tight(opt.npv(), float(reference_data["analytic_european"]["call_npv"]))
