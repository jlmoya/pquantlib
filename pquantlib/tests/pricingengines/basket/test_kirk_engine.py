"""Tests for KirkEngine — Kirk 1995 spread option engine.

Cross-validates against ``migration-harness/references/cluster/w4b.json``.

C++ parity:
ql/pricingengines/basket/kirkengine.{hpp,cpp} @ v1.42.1 (099987f0).

Tolerance: **TIGHT** — pure closed-form, no solver inside. Both
implementations use the same BlackCalculator on identical scaled
forward + variance inputs; agreement should be at machine precision.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.basket_option import BasketOption, SpreadBasketPayoff
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.basket.kirk_engine import KirkEngine
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
    GeneralizedBlackScholesProcess,
    GeneralizedBlackScholesProcess,
    Date,
]:
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.June, 2026)
    expiry = ref + 365

    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.00, day_counter=dc)
    vol1 = BlackConstantVol(reference_date=ref, calendar=cal, day_counter=dc, volatility=0.20)
    vol2 = BlackConstantVol(reference_date=ref, calendar=cal, day_counter=dc, volatility=0.25)

    process1 = GeneralizedBlackScholesProcess(
        x0=SimpleQuote(100.0), dividend_ts=div, risk_free_ts=rf, black_vol_ts=vol1
    )
    process2 = GeneralizedBlackScholesProcess(
        x0=SimpleQuote(90.0), dividend_ts=div, risk_free_ts=rf, black_vol_ts=vol2
    )
    return process1, process2, expiry


def test_call_matches_cpp(refs: dict[str, Any]) -> None:
    process1, process2, expiry = _setup()
    payoff = SpreadBasketPayoff(PlainVanillaPayoff(OptionType.Call, 5.0))
    opt = BasketOption(payoff, EuropeanExercise(expiry))
    opt.set_pricing_engine(KirkEngine(process1, process2, 0.5))

    tight(opt.npv(), float(refs["kirk_spread"]["call_npv"]))


def test_put_matches_cpp(refs: dict[str, Any]) -> None:
    process1, process2, expiry = _setup()
    payoff = SpreadBasketPayoff(PlainVanillaPayoff(OptionType.Put, 5.0))
    opt = BasketOption(payoff, EuropeanExercise(expiry))
    opt.set_pricing_engine(KirkEngine(process1, process2, 0.5))

    tight(opt.npv(), float(refs["kirk_spread"]["put_npv"]))
