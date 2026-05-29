"""Tests for AnalyticSimpleChooserEngine.

Cross-validates against ``migration-harness/references/cluster/w4b.json``.

C++ parity:
ql/pricingengines/exotic/analyticsimplechooserengine.{hpp,cpp}
@ v1.42.1 (099987f0).

Tolerance: **TIGHT** — pure closed-form (no solver inside). Both
implementations apply identical CDF expressions to the same
``CumulativeNormalDistribution``; agreement should be at float64
machine precision.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.european_option import EuropeanOption
from pquantlib.instruments.simple_chooser_option import SimpleChooserOption
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.exotic.analytic_simple_chooser_engine import (
    AnalyticSimpleChooserEngine,
)
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
def refs() -> dict[str, Any]:
    return load_reference("cluster/w4b")


def test_simple_chooser_matches_cpp(refs: dict[str, Any]) -> None:
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.June, 2026)
    choosing = ref + 91
    expiry = ref + 182

    spot = SimpleQuote(50.0)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.08, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    vol = BlackConstantVol(reference_date=ref, calendar=cal, day_counter=dc, volatility=0.25)
    process = GeneralizedBlackScholesProcess(
        x0=spot, dividend_ts=div, risk_free_ts=rf, black_vol_ts=vol
    )

    opt = SimpleChooserOption(
        choosing_date=choosing,
        strike=50.0,
        exercise=EuropeanExercise(expiry),
    )
    opt.set_pricing_engine(AnalyticSimpleChooserEngine(process))

    tight(opt.npv(), float(refs["simple_chooser"]["npv"]))


def test_chooser_call_floor_consistency() -> None:
    """At the choosing date a chooser must be worth at least max(call, put).

    This is a sanity check, not a probe-cross-validation. We construct
    a chooser plus a separate call+put with the same strike + maturity,
    price them, and assert chooser_NPV >= max(call_NPV, put_NPV).
    """
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.June, 2026)
    choosing = ref + 91
    expiry = ref + 182

    spot = SimpleQuote(50.0)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.08, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    vol = BlackConstantVol(reference_date=ref, calendar=cal, day_counter=dc, volatility=0.25)
    process = GeneralizedBlackScholesProcess(
        x0=spot, dividend_ts=div, risk_free_ts=rf, black_vol_ts=vol
    )

    chooser = SimpleChooserOption(choosing, 50.0, EuropeanExercise(expiry))
    chooser.set_pricing_engine(AnalyticSimpleChooserEngine(process))
    chooser_npv = chooser.npv()

    # Vanilla call.
    call = EuropeanOption(
        PlainVanillaPayoff(OptionType.Call, 50.0), EuropeanExercise(expiry)
    )
    call.set_pricing_engine(AnalyticEuropeanEngine(process))

    # Vanilla put.
    put = EuropeanOption(
        PlainVanillaPayoff(OptionType.Put, 50.0), EuropeanExercise(expiry)
    )
    put.set_pricing_engine(AnalyticEuropeanEngine(process))

    assert chooser_npv >= max(call.npv(), put.npv()) - 1e-12, (
        f"chooser NPV {chooser_npv} must be >= max(call={call.npv()}, put={put.npv()})"
    )
