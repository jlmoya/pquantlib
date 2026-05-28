"""Tests for AnalyticContinuousFloatingLookbackEngine.

# C++ parity:
# ql/pricingengines/lookback/analyticcontinuousfloatinglookback.{hpp,cpp}
# @ v1.42.1.

Cross-validates against ``analytic_continuous_floating_lookback`` section
of ``migration-harness/references/cluster/l5e.json``.

Test setup: (S=100, current extremum=100, T=1y, r=5%, q=2%, sigma=30%).
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.lookback_option import (
    ContinuousFloatingLookbackOption,
)
from pquantlib.payoffs import FloatingTypePayoff, OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.lookback.analytic_continuous_floating_lookback_engine import (
    AnalyticContinuousFloatingLookbackEngine,
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


def _build_process() -> tuple[GeneralizedBlackScholesProcess, Date]:
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.June, 2026)
    expiry = ref + 365
    spot_q = SimpleQuote(100.0)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.02, day_counter=dc)
    vol = BlackConstantVol(reference_date=ref, calendar=cal, day_counter=dc, volatility=0.30)
    process = GeneralizedBlackScholesProcess(
        x0=spot_q, dividend_ts=div, risk_free_ts=rf, black_vol_ts=vol
    )
    return process, expiry


def test_floating_call_npv_matches_conze_viswanathan(
    reference_data: dict[str, Any],
) -> None:
    process, expiry = _build_process()
    payoff = FloatingTypePayoff(OptionType.Call)
    exercise = EuropeanExercise(expiry)
    opt = ContinuousFloatingLookbackOption(100.0, payoff, exercise)
    opt.set_pricing_engine(AnalyticContinuousFloatingLookbackEngine(process))
    tight(
        opt.npv(),
        float(reference_data["analytic_continuous_floating_lookback"]["call_npv"]),
    )


def test_floating_put_npv_matches_conze_viswanathan(
    reference_data: dict[str, Any],
) -> None:
    process, expiry = _build_process()
    payoff = FloatingTypePayoff(OptionType.Put)
    exercise = EuropeanExercise(expiry)
    opt = ContinuousFloatingLookbackOption(100.0, payoff, exercise)
    opt.set_pricing_engine(AnalyticContinuousFloatingLookbackEngine(process))
    tight(
        opt.npv(),
        float(reference_data["analytic_continuous_floating_lookback"]["put_npv"]),
    )


def test_engine_rejects_non_floating_payoff() -> None:
    """The engine requires a FloatingTypePayoff; passing a striked
    PlainVanilla should raise."""
    process, expiry = _build_process()
    # Build the option with a FloatingTypePayoff (constructor type
    # constraint), then poke the engine arguments to bypass.
    opt = ContinuousFloatingLookbackOption(
        100.0, FloatingTypePayoff(OptionType.Call), EuropeanExercise(expiry)
    )
    engine = AnalyticContinuousFloatingLookbackEngine(process)
    opt.set_pricing_engine(engine)
    # Manually invalidate the payoff in the engine arguments.
    engine.get_arguments().payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    with pytest.raises(LibraryException, match="Non-floating payoff"):
        engine.calculate()
