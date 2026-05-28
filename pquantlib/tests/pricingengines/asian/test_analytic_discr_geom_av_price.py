"""Tests for ``AnalyticDiscreteGeometricAveragePriceAsianEngine``.

# C++ parity: ql/pricingengines/asian/analytic_discr_geom_av_price.{hpp,cpp}
# (v1.42.1).

Cross-validates against the ``analytic_discr_geom_av_call`` section
of ``migration-harness/references/cluster/l5c.json``: 12 monthly
fixings, 1y maturity, BSM textbook scenario.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.asian_option import DiscreteAveragingAsianOption
from pquantlib.instruments.average_type import AverageType
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.asian.analytic_discr_geom_av_price import (
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
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def reference_data() -> dict[str, Any]:
    return reference_reader.load("cluster/l5c")


def _build_textbook_asian() -> tuple[
    GeneralizedBlackScholesProcess, DiscreteAveragingAsianOption
]:
    """Build the 12-monthly-fixing geometric Asian call from the probe."""
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.May, 2026)
    spot = SimpleQuote(100.0)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.0, day_counter=dc)
    vol = BlackConstantVol(
        reference_date=ref, calendar=cal, day_counter=dc, volatility=0.20
    )
    process = GeneralizedBlackScholesProcess(
        x0=spot, dividend_ts=div, risk_free_ts=rf, black_vol_ts=vol
    )
    fixings = [ref + m * 30 for m in range(1, 13)]
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(fixings[-1])
    opt = DiscreteAveragingAsianOption(
        AverageType.Geometric, 1.0, 0, fixings, payoff, exercise
    )
    return process, opt


def test_call_npv_matches_cpp(reference_data: dict[str, Any]) -> None:
    """Closed-form geometric-average Asian call vs the C++ probe."""
    process, opt = _build_textbook_asian()
    engine = AnalyticDiscreteGeometricAveragePriceAsianEngine(process)
    opt.set_pricing_engine(engine)
    npv = opt.npv()
    expected = float(reference_data["analytic_discr_geom_av_call"]["npv"])
    tight(npv, expected)
