"""Tests for AnalyticContinuousGeometricAveragePriceAsianEngine.

# C++ parity: ql/pricingengines/asian/analytic_cont_geom_av_price.{hpp,cpp}
# @ v1.42.1.

Cross-validates against ``analytic_continuous_geometric_asian`` section
of ``migration-harness/references/cluster/l5e.json``.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, EuropeanExercise
from pquantlib.instruments.asian_option import (
    AverageType,
    ContinuousAveragingAsianOption,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.asian.analytic_continuous_geometric_average_price_engine import (
    AnalyticContinuousGeometricAveragePriceAsianEngine,
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
    return load_reference("cluster/l5e")


def _build_process_and_dates() -> tuple[GeneralizedBlackScholesProcess, Date]:
    """Build the textbook BSM process for the L5-E probe:
    spot=100, r=5%, q=0%, sigma=30%, T=1y under Actual/365 Fixed.
    """
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.June, 2026)
    expiry = ref + 365

    spot_q = SimpleQuote(100.0)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.00, day_counter=dc)
    vol = BlackConstantVol(reference_date=ref, calendar=cal, day_counter=dc, volatility=0.30)
    process = GeneralizedBlackScholesProcess(
        x0=spot_q, dividend_ts=div, risk_free_ts=rf, black_vol_ts=vol
    )
    return process, expiry


# --- Call NPV + Greeks vs Kemna-Vorst reference ---------------------------


def test_call_npv_matches_kemna_vorst(reference_data: dict[str, Any]) -> None:
    process, expiry = _build_process_and_dates()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = ContinuousAveragingAsianOption(AverageType.Geometric, payoff, exercise)
    opt.set_pricing_engine(
        AnalyticContinuousGeometricAveragePriceAsianEngine(process)
    )
    tight(
        opt.npv(),
        float(reference_data["analytic_continuous_geometric_asian"]["call_npv"]),
    )


def test_call_delta_matches_reference(reference_data: dict[str, Any]) -> None:
    process, expiry = _build_process_and_dates()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = ContinuousAveragingAsianOption(AverageType.Geometric, payoff, exercise)
    opt.set_pricing_engine(
        AnalyticContinuousGeometricAveragePriceAsianEngine(process)
    )
    tight(
        opt.delta(),
        float(reference_data["analytic_continuous_geometric_asian"]["call_delta"]),
    )


def test_call_gamma_matches_reference(reference_data: dict[str, Any]) -> None:
    process, expiry = _build_process_and_dates()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = ContinuousAveragingAsianOption(AverageType.Geometric, payoff, exercise)
    opt.set_pricing_engine(
        AnalyticContinuousGeometricAveragePriceAsianEngine(process)
    )
    tight(
        opt.gamma(),
        float(reference_data["analytic_continuous_geometric_asian"]["call_gamma"]),
    )


def test_call_vega_matches_reference(reference_data: dict[str, Any]) -> None:
    process, expiry = _build_process_and_dates()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = ContinuousAveragingAsianOption(AverageType.Geometric, payoff, exercise)
    opt.set_pricing_engine(
        AnalyticContinuousGeometricAveragePriceAsianEngine(process)
    )
    tight(
        opt.vega(),
        float(reference_data["analytic_continuous_geometric_asian"]["call_vega"]),
    )


def test_call_rho_matches_reference(reference_data: dict[str, Any]) -> None:
    process, expiry = _build_process_and_dates()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = ContinuousAveragingAsianOption(AverageType.Geometric, payoff, exercise)
    opt.set_pricing_engine(
        AnalyticContinuousGeometricAveragePriceAsianEngine(process)
    )
    tight(
        opt.rho(),
        float(reference_data["analytic_continuous_geometric_asian"]["call_rho"]),
    )


def test_call_dividend_rho_matches_reference(reference_data: dict[str, Any]) -> None:
    process, expiry = _build_process_and_dates()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = ContinuousAveragingAsianOption(AverageType.Geometric, payoff, exercise)
    opt.set_pricing_engine(
        AnalyticContinuousGeometricAveragePriceAsianEngine(process)
    )
    tight(
        opt.dividend_rho(),
        float(reference_data["analytic_continuous_geometric_asian"]["call_dividend_rho"]),
    )


# --- error paths -----------------------------------------------------------


def test_engine_rejects_arithmetic_average() -> None:
    process, expiry = _build_process_and_dates()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = ContinuousAveragingAsianOption(AverageType.Arithmetic, payoff, exercise)
    opt.set_pricing_engine(
        AnalyticContinuousGeometricAveragePriceAsianEngine(process)
    )
    with pytest.raises(LibraryException, match="not a geometric average"):
        opt.npv()


def test_engine_rejects_american_exercise() -> None:
    process, expiry = _build_process_and_dates()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    earliest = Date.from_ymd(15, Month.June, 2026)
    exercise = AmericanExercise(earliest, expiry)
    opt = ContinuousAveragingAsianOption(AverageType.Geometric, payoff, exercise)
    opt.set_pricing_engine(
        AnalyticContinuousGeometricAveragePriceAsianEngine(process)
    )
    with pytest.raises(LibraryException, match="not an European"):
        opt.npv()


def test_engine_rejects_seasoned_option() -> None:
    process, expiry = _build_process_and_dates()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)
    start = Date.from_ymd(1, Month.January, 2026)
    opt = ContinuousAveragingAsianOption(
        AverageType.Geometric, payoff, exercise, start_date=start
    )
    opt.set_pricing_engine(
        AnalyticContinuousGeometricAveragePriceAsianEngine(process)
    )
    with pytest.raises(LibraryException, match="seasoned"):
        opt.npv()
