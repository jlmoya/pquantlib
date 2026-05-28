"""Tests for AnalyticBarrierEngine (Reiner-Rubinstein).

# C++ parity:
# ql/pricingengines/barrier/analyticbarrierengine.{hpp,cpp} @ v1.42.1.

Cross-validates against ``analytic_barrier`` section of
``migration-harness/references/cluster/l5e.json``.

Test setup: (S=100, K=100, B=95 down / B=105 up, rebate=3, T=1y,
r=5%, q=2%, sigma=30%).
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, EuropeanExercise
from pquantlib.instruments.barrier_option import BarrierOption, BarrierType
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.barrier.analytic_barrier_engine import (
    AnalyticBarrierEngine,
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


def _price(
    process: GeneralizedBlackScholesProcess,
    expiry: Date,
    option_type: OptionType,
    barrier_type: BarrierType,
    barrier: float,
    rebate: float,
    strike: float,
) -> float:
    payoff = PlainVanillaPayoff(option_type, strike)
    exercise = EuropeanExercise(expiry)
    opt = BarrierOption(barrier_type, barrier, rebate, payoff, exercise)
    opt.set_pricing_engine(AnalyticBarrierEngine(process))
    return opt.npv()


# --- 8 reference comparisons (Call/Put x 4 barrier types) ------------------


@pytest.mark.parametrize(
    ("key", "option_type", "bt", "barrier"),
    [
        ("down_in_call", OptionType.Call, BarrierType.DownIn, 95.0),
        ("up_in_call", OptionType.Call, BarrierType.UpIn, 105.0),
        ("down_out_call", OptionType.Call, BarrierType.DownOut, 95.0),
        ("up_out_call", OptionType.Call, BarrierType.UpOut, 105.0),
        ("down_in_put", OptionType.Put, BarrierType.DownIn, 95.0),
        ("up_in_put", OptionType.Put, BarrierType.UpIn, 105.0),
        ("down_out_put", OptionType.Put, BarrierType.DownOut, 95.0),
        ("up_out_put", OptionType.Put, BarrierType.UpOut, 105.0),
    ],
)
def test_npv_matches_reiner_rubinstein(
    reference_data: dict[str, Any],
    key: str,
    option_type: OptionType,
    bt: BarrierType,
    barrier: float,
) -> None:
    process, expiry = _build_process()
    npv = _price(process, expiry, option_type, bt, barrier, 3.0, 100.0)
    tight(npv, float(reference_data["analytic_barrier"][key]))


# --- error paths -----------------------------------------------------------


def test_engine_rejects_american_exercise() -> None:
    process, expiry = _build_process()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    earliest = Date.from_ymd(15, Month.June, 2026)
    exercise = AmericanExercise(earliest, expiry)
    opt = BarrierOption(BarrierType.DownOut, 95.0, 3.0, payoff, exercise)
    opt.set_pricing_engine(AnalyticBarrierEngine(process))
    with pytest.raises(LibraryException, match="only european"):
        opt.npv()


def test_engine_rejects_triggered_barrier_down_out() -> None:
    """Spot=100 with barrier=110 down-out has barrier already touched."""
    process, expiry = _build_process()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    exercise = EuropeanExercise(expiry)
    # Down-out with B=110 > spot=100 means the spot already crossed below.
    opt = BarrierOption(BarrierType.DownOut, 110.0, 3.0, payoff, exercise)
    opt.set_pricing_engine(AnalyticBarrierEngine(process))
    with pytest.raises(LibraryException, match="barrier touched"):
        opt.npv()
