"""Tests for AnalyticHolderExtensibleOptionEngine.

Cross-validates against ``migration-harness/references/cluster/w4b.json``.

C++ parity:
ql/pricingengines/exotic/analyticholderextensibleoptionengine.{hpp,cpp}
@ v1.42.1 (099987f0).

Tolerance choice: **CUSTOM** (abs_tol=5e-4) — the formula contains
Newton-Raphson iterations on the critical spots ``I1``/``I2`` to
``eps=1e-3``. That accuracy propagates through the bivariate-normal
expressions, yielding ~1e-4 absolute NPV agreement with C++.
``loose`` (abs_tol=1e-8) is too strict.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.holder_extensible_option import HolderExtensibleOption
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.exotic.analytic_holder_extensible_option_engine import (
    AnalyticHolderExtensibleOptionEngine,
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
from pquantlib.testing.tolerance import custom
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
    t2 = ref + 273

    spot = SimpleQuote(100.0)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.08, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.00, day_counter=dc)
    vol = BlackConstantVol(reference_date=ref, calendar=cal, day_counter=dc, volatility=0.25)
    process = GeneralizedBlackScholesProcess(
        x0=spot, dividend_ts=div, risk_free_ts=rf, black_vol_ts=vol
    )
    return process, t1, t2


_REASON = (
    "Newton-Raphson on critical spots I1/I2 runs to eps=1e-3; that "
    "tolerance propagates through bivariate-normal terms producing ~1e-4 "
    "absolute NPV agreement with C++."
)


def test_call_matches_cpp(refs: dict[str, Any]) -> None:
    process, t1, t2 = _setup()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    opt = HolderExtensibleOption(
        option_type=OptionType.Call,
        premium=1.0,
        second_expiry_date=t2,
        second_strike=105.0,
        payoff=payoff,
        exercise=EuropeanExercise(t1),
    )
    opt.set_pricing_engine(AnalyticHolderExtensibleOptionEngine(process))

    custom(
        opt.npv(),
        float(refs["holder_extensible"]["call_npv"]),
        abs_tol=5e-4,
        rel_tol=1e-4,
        reason=_REASON,
    )


def test_put_matches_cpp(refs: dict[str, Any]) -> None:
    process, t1, t2 = _setup()
    payoff = PlainVanillaPayoff(OptionType.Put, 100.0)
    opt = HolderExtensibleOption(
        option_type=OptionType.Put,
        premium=1.0,
        second_expiry_date=t2,
        second_strike=105.0,
        payoff=payoff,
        exercise=EuropeanExercise(t1),
    )
    opt.set_pricing_engine(AnalyticHolderExtensibleOptionEngine(process))

    custom(
        opt.npv(),
        float(refs["holder_extensible"]["put_npv"]),
        abs_tol=5e-4,
        rel_tol=1e-3,
        reason=_REASON,
    )
