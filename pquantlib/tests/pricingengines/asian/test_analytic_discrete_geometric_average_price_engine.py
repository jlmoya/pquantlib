"""Tests for AnalyticDiscreteGeometricAveragePriceAsianEngine.

# C++ parity:
# ql/pricingengines/asian/analytic_discr_geom_av_price.{hpp,cpp}
# @ v1.42.1.

Cross-validates against ``analytic_discrete_geometric_asian`` section
of ``migration-harness/references/cluster/l5e.json``.

Probe configuration: 12 monthly fixings at ref + i*365/12 (i in 1..12),
running accumulator = 1.0, past fixings = 0.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, EuropeanExercise
from pquantlib.instruments.asian_option import (
    AverageType,
    DiscreteAveragingAsianOption,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.asian.analytic_discrete_geometric_average_price_engine import (
    AnalyticDiscreteGeometricAveragePriceAsianEngine,
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


def _build_process_dates_fixings() -> tuple[
    GeneralizedBlackScholesProcess, Date, list[Date]
]:
    """Build the textbook process (spot=100, r=5%, q=0%, sigma=30%, T=1y)
    + 12 monthly fixings on integer-day offsets ref + i*365/12.
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
    # Mirror C++ probe: 12 monthly fixings at ref + i * (365 / 12)
    # (integer division — same value as C++).
    fixings = [ref + (i * 365 // 12) for i in range(1, 13)]
    return process, expiry, fixings


# --- Call NPV + Greeks vs Levy 1997 reference ----------------------------


def test_call_npv_matches_levy(reference_data: dict[str, Any]) -> None:
    process, expiry, fixings = _build_process_dates_fixings()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = DiscreteAveragingAsianOption(
        AverageType.Geometric, 1.0, 0, fixings, payoff, exercise
    )
    opt.set_pricing_engine(
        AnalyticDiscreteGeometricAveragePriceAsianEngine(process)
    )
    tight(
        opt.npv(),
        float(reference_data["analytic_discrete_geometric_asian"]["call_npv"]),
    )


def test_call_delta_matches_reference(reference_data: dict[str, Any]) -> None:
    process, expiry, fixings = _build_process_dates_fixings()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = DiscreteAveragingAsianOption(
        AverageType.Geometric, 1.0, 0, fixings, payoff, exercise
    )
    opt.set_pricing_engine(
        AnalyticDiscreteGeometricAveragePriceAsianEngine(process)
    )
    tight(
        opt.delta(),
        float(reference_data["analytic_discrete_geometric_asian"]["call_delta"]),
    )


def test_call_gamma_matches_reference(reference_data: dict[str, Any]) -> None:
    process, expiry, fixings = _build_process_dates_fixings()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)
    opt = DiscreteAveragingAsianOption(
        AverageType.Geometric, 1.0, 0, fixings, payoff, exercise
    )
    opt.set_pricing_engine(
        AnalyticDiscreteGeometricAveragePriceAsianEngine(process)
    )
    tight(
        opt.gamma(),
        float(reference_data["analytic_discrete_geometric_asian"]["call_gamma"]),
    )


# --- error paths -----------------------------------------------------------


def test_engine_rejects_american_exercise() -> None:
    process, expiry, fixings = _build_process_dates_fixings()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    earliest = Date.from_ymd(15, Month.June, 2026)
    exercise = AmericanExercise(earliest, expiry)
    opt = DiscreteAveragingAsianOption(
        AverageType.Geometric, 1.0, 0, fixings, payoff, exercise
    )
    opt.set_pricing_engine(
        AnalyticDiscreteGeometricAveragePriceAsianEngine(process)
    )
    with pytest.raises(LibraryException, match="not an European"):
        opt.npv()
