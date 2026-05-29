"""Tests for ContinuousArithmeticAsianLevyEngine.

Cross-validates against ``migration-harness/references/cluster/w4b.json``.

C++ parity:
ql/pricingengines/asian/continuousarithmeticasianlevyengine.{hpp,cpp}
@ v1.42.1 (099987f0).

Tolerance: **TIGHT** — closed-form expression with no solver inside.
Both implementations apply the same Levy 1992 formula to identical
inputs; agreement at machine precision is expected.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.asian_option import (
    AverageType,
    ContinuousAveragingAsianOption,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.asian.continuous_arithmetic_asian_levy_engine import (
    ContinuousArithmeticAsianLevyEngine,
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


def _setup() -> tuple[
    GeneralizedBlackScholesProcess, Date, Date, SimpleQuote
]:
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.June, 2026)
    start = ref - 91
    expiry = ref + 274

    spot = SimpleQuote(100.0)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.02, day_counter=dc)
    vol = BlackConstantVol(reference_date=ref, calendar=cal, day_counter=dc, volatility=0.20)
    process = GeneralizedBlackScholesProcess(
        x0=spot, dividend_ts=div, risk_free_ts=rf, black_vol_ts=vol
    )
    return process, start, expiry, SimpleQuote(100.0)


def test_call_matches_cpp(refs: dict[str, Any]) -> None:
    process, start, expiry, avg = _setup()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    opt = ContinuousAveragingAsianOption(
        AverageType.Arithmetic, payoff, EuropeanExercise(expiry), start_date=start
    )
    opt.set_pricing_engine(
        ContinuousArithmeticAsianLevyEngine(process, avg)
    )

    tight(opt.npv(), float(refs["asian_levy"]["call_npv"]))


def test_put_matches_cpp(refs: dict[str, Any]) -> None:
    process, start, expiry, avg = _setup()
    payoff = PlainVanillaPayoff(OptionType.Put, 100.0)
    opt = ContinuousAveragingAsianOption(
        AverageType.Arithmetic, payoff, EuropeanExercise(expiry), start_date=start
    )
    opt.set_pricing_engine(
        ContinuousArithmeticAsianLevyEngine(process, avg)
    )

    tight(opt.npv(), float(refs["asian_levy"]["put_npv"]))
