"""Tests for StulzEngine (Stulz 1982 2-asset basket closed form).

# C++ parity: ql/pricingengines/basket/stulzengine.{hpp,cpp} @ v1.42.1.

Cross-validates against ``stulz`` section of
``migration-harness/references/cluster/l5e.json``.

Test setup: S1=S2=100, K=100, q1=q2=0, sigma1=20%, sigma2=30%,
rho=0.5, T=1y, r=5%.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.basket_option import (
    BasketOption,
    MaxBasketPayoff,
    MinBasketPayoff,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.basket.stulz_engine import StulzEngine
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


def _build_two_processes() -> tuple[
    GeneralizedBlackScholesProcess, GeneralizedBlackScholesProcess, Date
]:
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.June, 2026)
    expiry = ref + 365
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.00, day_counter=dc)
    vol1 = BlackConstantVol(reference_date=ref, calendar=cal, day_counter=dc, volatility=0.20)
    vol2 = BlackConstantVol(reference_date=ref, calendar=cal, day_counter=dc, volatility=0.30)
    spot1 = SimpleQuote(100.0)
    spot2 = SimpleQuote(100.0)
    p1 = GeneralizedBlackScholesProcess(
        x0=spot1, dividend_ts=div, risk_free_ts=rf, black_vol_ts=vol1
    )
    p2 = GeneralizedBlackScholesProcess(
        x0=spot2, dividend_ts=div, risk_free_ts=rf, black_vol_ts=vol2
    )
    return p1, p2, expiry


def test_min_basket_call_matches_stulz(reference_data: dict[str, Any]) -> None:
    p1, p2, expiry = _build_two_processes()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    basket = MinBasketPayoff(payoff)
    opt = BasketOption(basket, EuropeanExercise(expiry))
    opt.set_pricing_engine(StulzEngine(p1, p2, 0.5))
    tight(opt.npv(), float(reference_data["stulz"]["min_call_npv"]))


def test_min_basket_put_matches_stulz(reference_data: dict[str, Any]) -> None:
    p1, p2, expiry = _build_two_processes()
    payoff = PlainVanillaPayoff(OptionType.Put, 100.0)
    basket = MinBasketPayoff(payoff)
    opt = BasketOption(basket, EuropeanExercise(expiry))
    opt.set_pricing_engine(StulzEngine(p1, p2, 0.5))
    tight(opt.npv(), float(reference_data["stulz"]["min_put_npv"]))


def test_max_basket_call_matches_stulz(reference_data: dict[str, Any]) -> None:
    p1, p2, expiry = _build_two_processes()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    basket = MaxBasketPayoff(payoff)
    opt = BasketOption(basket, EuropeanExercise(expiry))
    opt.set_pricing_engine(StulzEngine(p1, p2, 0.5))
    tight(opt.npv(), float(reference_data["stulz"]["max_call_npv"]))


def test_max_basket_put_matches_stulz(reference_data: dict[str, Any]) -> None:
    p1, p2, expiry = _build_two_processes()
    payoff = PlainVanillaPayoff(OptionType.Put, 100.0)
    basket = MaxBasketPayoff(payoff)
    opt = BasketOption(basket, EuropeanExercise(expiry))
    opt.set_pricing_engine(StulzEngine(p1, p2, 0.5))
    tight(opt.npv(), float(reference_data["stulz"]["max_put_npv"]))
