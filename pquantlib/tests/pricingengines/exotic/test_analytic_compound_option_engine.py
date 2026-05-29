"""Tests for AnalyticCompoundOptionEngine.

Cross-validates against ``migration-harness/references/cluster/w4b.json``.

C++ parity:
ql/pricingengines/exotic/analyticcompoundoptionengine.{hpp,cpp}
@ v1.42.1 (099987f0).

Tolerance choice: **CUSTOM** (abs_tol=5e-5) — the formula has a
1e-6 Brent solver inside it (finding the trigger spot ``S*``).
That accuracy gets amplified through the bivariate normal CDFs
and the multi-term linear combination, so absolute NPV agreement
with C++ lands in the 1e-5 range. ``loose`` (abs_tol=1e-8) is
too strict; a custom tolerance with explicit justification is
the discipline-correct choice.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.compound_option import CompoundOption
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.exotic.analytic_compound_option_engine import (
    AnalyticCompoundOptionEngine,
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
    """Same testbed as the C++ probe: spot=100, r=5%, q=2%, sigma=30%."""
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.June, 2026)
    mother_exp = ref + 182
    daughter_exp = ref + 365

    spot = SimpleQuote(100.0)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.02, day_counter=dc)
    vol = BlackConstantVol(reference_date=ref, calendar=cal, day_counter=dc, volatility=0.30)
    process = GeneralizedBlackScholesProcess(
        x0=spot, dividend_ts=div, risk_free_ts=rf, black_vol_ts=vol
    )
    return process, mother_exp, daughter_exp


def _price(
    mother_type: OptionType,
    daughter_type: OptionType,
) -> float:
    process, mother_exp, daughter_exp = _setup()
    mother_strike = 5.0
    daughter_strike = 95.0

    mother_payoff = PlainVanillaPayoff(mother_type, mother_strike)
    daughter_payoff = PlainVanillaPayoff(daughter_type, daughter_strike)

    opt = CompoundOption(
        mother_payoff,
        EuropeanExercise(mother_exp),
        daughter_payoff,
        EuropeanExercise(daughter_exp),
    )
    opt.set_pricing_engine(AnalyticCompoundOptionEngine(process))
    return opt.npv()


_REASON = (
    "Brent solver runs to 1e-6 accuracy on the trigger spot S*; the "
    "1e-6 floor gets amplified through the bivariate-normal terms, "
    "yielding ~1e-5 absolute NPV agreement with C++ (which uses the "
    "same algorithm but solves to a slightly different fixed point)."
)


def test_call_on_call_matches_cpp(refs: dict[str, Any]) -> None:
    custom(
        _price(OptionType.Call, OptionType.Call),
        float(refs["compound"]["call_on_call"]),
        abs_tol=5e-5,
        rel_tol=5e-6,
        reason=_REASON,
    )


def test_put_on_call_matches_cpp(refs: dict[str, Any]) -> None:
    custom(
        _price(OptionType.Put, OptionType.Call),
        float(refs["compound"]["put_on_call"]),
        abs_tol=5e-5,
        rel_tol=5e-6,
        reason=_REASON,
    )


def test_call_on_put_matches_cpp(refs: dict[str, Any]) -> None:
    custom(
        _price(OptionType.Call, OptionType.Put),
        float(refs["compound"]["call_on_put"]),
        abs_tol=5e-5,
        rel_tol=5e-6,
        reason=_REASON,
    )


def test_put_on_put_matches_cpp(refs: dict[str, Any]) -> None:
    custom(
        _price(OptionType.Put, OptionType.Put),
        float(refs["compound"]["put_on_put"]),
        abs_tol=5e-5,
        rel_tol=5e-6,
        reason=_REASON,
    )
