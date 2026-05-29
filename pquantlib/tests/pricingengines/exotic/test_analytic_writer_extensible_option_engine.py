"""Tests for AnalyticWriterExtensibleOptionEngine.

Cross-validates against ``migration-harness/references/cluster/w4b.json``.

C++ parity:
ql/pricingengines/exotic/analyticwriterextensibleoptionengine.{hpp,cpp}
@ v1.42.1 (099987f0).

Tolerance: **TIGHT** — pure closed-form, no solver inside. Both
implementations apply the same BSM + bivariate-normal expression to
identical inputs; agreement should be at float64 machine precision.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.writer_extensible_option import WriterExtensibleOption
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.exotic.analytic_writer_extensible_option_engine import (
    AnalyticWriterExtensibleOptionEngine,
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


def _setup() -> tuple[GeneralizedBlackScholesProcess, Date, Date]:
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.June, 2026)
    t1 = ref + 182
    t2 = ref + 365

    spot = SimpleQuote(80.0)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.08, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.00, day_counter=dc)
    vol = BlackConstantVol(reference_date=ref, calendar=cal, day_counter=dc, volatility=0.30)
    process = GeneralizedBlackScholesProcess(
        x0=spot, dividend_ts=div, risk_free_ts=rf, black_vol_ts=vol
    )
    return process, t1, t2


def test_call_matches_cpp(refs: dict[str, Any]) -> None:
    process, t1, t2 = _setup()
    payoff1 = PlainVanillaPayoff(OptionType.Call, 90.0)
    payoff2 = PlainVanillaPayoff(OptionType.Call, 82.0)
    opt = WriterExtensibleOption(
        payoff1, EuropeanExercise(t1), payoff2, EuropeanExercise(t2)
    )
    opt.set_pricing_engine(AnalyticWriterExtensibleOptionEngine(process))

    tight(opt.npv(), float(refs["writer_extensible"]["call_npv"]))


def test_put_matches_cpp(refs: dict[str, Any]) -> None:
    process, t1, t2 = _setup()
    payoff1 = PlainVanillaPayoff(OptionType.Put, 90.0)
    payoff2 = PlainVanillaPayoff(OptionType.Put, 82.0)
    opt = WriterExtensibleOption(
        payoff1, EuropeanExercise(t1), payoff2, EuropeanExercise(t2)
    )
    opt.set_pricing_engine(AnalyticWriterExtensibleOptionEngine(process))

    tight(opt.npv(), float(refs["writer_extensible"]["put_npv"]))
