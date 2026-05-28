"""Tests for AnalyticBinaryBarrierEngine.

# C++ parity:
# ql/pricingengines/barrier/analyticbinarybarrierengine.{hpp,cpp}
# @ v1.42.1.

Cross-validates against ``analytic_binary_barrier`` section of
``migration-harness/references/cluster/l5e.json``.

Test setup: cash-or-nothing + asset-or-nothing American binary
barriers, (S=100, K=100, B=95(down)/105(up), cashPayoff=10, T=1y,
r=5%, q=2%, sigma=30%) with payoff-at-expiry American exercise.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, EuropeanExercise
from pquantlib.instruments.barrier_option import BarrierOption, BarrierType
from pquantlib.payoffs import (
    AssetOrNothingPayoff,
    CashOrNothingPayoff,
    OptionType,
)
from pquantlib.pricingengines.barrier.analytic_binary_barrier_engine import (
    AnalyticBinaryBarrierEngine,
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


def _build_process() -> tuple[GeneralizedBlackScholesProcess, Date, Date]:
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
    return process, ref, expiry


def _price_cash(
    process: GeneralizedBlackScholesProcess,
    ref: Date,
    expiry: Date,
    option_type: OptionType,
    bt: BarrierType,
    barrier: float,
    cash: float = 10.0,
) -> float:
    payoff = CashOrNothingPayoff(option_type, 100.0, cash)
    exercise = AmericanExercise(ref, expiry, payoff_at_expiry=True)
    opt = BarrierOption(bt, barrier, 0.0, payoff, exercise)
    opt.set_pricing_engine(AnalyticBinaryBarrierEngine(process))
    return opt.npv()


def _price_asset(
    process: GeneralizedBlackScholesProcess,
    ref: Date,
    expiry: Date,
    option_type: OptionType,
    bt: BarrierType,
    barrier: float,
) -> float:
    payoff = AssetOrNothingPayoff(option_type, 100.0)
    exercise = AmericanExercise(ref, expiry, payoff_at_expiry=True)
    opt = BarrierOption(bt, barrier, 0.0, payoff, exercise)
    opt.set_pricing_engine(AnalyticBinaryBarrierEngine(process))
    return opt.npv()


# --- Cash-or-nothing binary barriers (8 combos) ---------------------------


@pytest.mark.parametrize(
    ("key", "option_type", "bt", "barrier"),
    [
        ("cash_down_in_call", OptionType.Call, BarrierType.DownIn, 95.0),
        ("cash_up_in_call", OptionType.Call, BarrierType.UpIn, 105.0),
        ("cash_down_out_call", OptionType.Call, BarrierType.DownOut, 95.0),
        ("cash_up_out_call", OptionType.Call, BarrierType.UpOut, 105.0),
        ("cash_down_in_put", OptionType.Put, BarrierType.DownIn, 95.0),
        ("cash_up_in_put", OptionType.Put, BarrierType.UpIn, 105.0),
        ("cash_down_out_put", OptionType.Put, BarrierType.DownOut, 95.0),
        ("cash_up_out_put", OptionType.Put, BarrierType.UpOut, 105.0),
    ],
)
def test_cash_or_nothing_npv_matches_reference(
    reference_data: dict[str, Any],
    key: str,
    option_type: OptionType,
    bt: BarrierType,
    barrier: float,
) -> None:
    process, ref, expiry = _build_process()
    npv = _price_cash(process, ref, expiry, option_type, bt, barrier)
    tight(npv, float(reference_data["analytic_binary_barrier"][key]))


# --- Asset-or-nothing binary barriers (2 sampled combos) ------------------


def test_asset_or_nothing_down_in_call(reference_data: dict[str, Any]) -> None:
    process, ref, expiry = _build_process()
    npv = _price_asset(
        process, ref, expiry, OptionType.Call, BarrierType.DownIn, 95.0
    )
    tight(npv, float(reference_data["analytic_binary_barrier"]["asset_down_in_call"]))


def test_asset_or_nothing_up_out_put(reference_data: dict[str, Any]) -> None:
    process, ref, expiry = _build_process()
    npv = _price_asset(
        process, ref, expiry, OptionType.Put, BarrierType.UpOut, 105.0
    )
    tight(npv, float(reference_data["analytic_binary_barrier"]["asset_up_out_put"]))


# --- error paths -----------------------------------------------------------


def test_engine_rejects_european_exercise() -> None:
    process, _ref, expiry = _build_process()
    payoff = CashOrNothingPayoff(OptionType.Call, 100.0, 10.0)
    exercise = EuropeanExercise(expiry)
    opt = BarrierOption(BarrierType.DownIn, 95.0, 0.0, payoff, exercise)
    opt.set_pricing_engine(AnalyticBinaryBarrierEngine(process))
    with pytest.raises(LibraryException, match="non-American exercise"):
        opt.npv()


def test_engine_rejects_non_payoff_at_expiry() -> None:
    process, ref, expiry = _build_process()
    payoff = CashOrNothingPayoff(OptionType.Call, 100.0, 10.0)
    exercise = AmericanExercise(ref, expiry, payoff_at_expiry=False)
    opt = BarrierOption(BarrierType.DownIn, 95.0, 0.0, payoff, exercise)
    opt.set_pricing_engine(AnalyticBinaryBarrierEngine(process))
    with pytest.raises(LibraryException, match="payoff must be at expiry"):
        opt.npv()
