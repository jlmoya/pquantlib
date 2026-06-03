"""Tests for the ConvertibleBond subsystem (W-S4).

# C++ parity:
# ql/instruments/bonds/convertiblebonds.{hpp,cpp} +
# ql/instruments/callabilityschedule.hpp (SoftCallability) +
# ql/pricingengines/bond/discretizedconvertible.{hpp,cpp} +
# ql/pricingengines/bond/binomialconvertibleengine.hpp +
# ql/methods/lattices/tflattice.hpp (Tsiveriotis-Fernandes) @ v1.42.1.

PRIMARY cross-validation: the Python ``BinomialConvertibleEngine`` (CRR
Tsiveriotis-Fernandes lattice) reproduces the C++ SAME engine at the SAME
number of time steps (N=801, recorded in the reference JSON) -> TIGHT.
There is no analytic convertible value, so the C++ same-engine value IS the
reference (tree-vs-identical-tree).

Scenario mirrors convertiblebonds.cpp CommonVars, pinned to a fixed
evaluation date:

* spot 50, conversion ratio 2.0, sigma 15%, r 5%, q 2%, credit spread 0.5%.
* 10-year annual 5% fixed-coupon convertible, redemption 100.
* three flavours: European; American; American + a SoftCallability(Call,
  trigger 1.10) at year 5 + a plain Put at year 7 + two FixedDividends.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.cashflows.dividend import FixedDividend
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.exercise import AmericanExercise, EuropeanExercise
from pquantlib.instruments.bond import BondPrice, BondPriceType
from pquantlib.instruments.bonds.convertible_bonds import ConvertibleFixedCouponBond
from pquantlib.instruments.callability import Callability, CallabilityType
from pquantlib.instruments.soft_callability import SoftCallability
from pquantlib.methods.lattices.binomial_tree import CoxRossRubinstein
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.bond.binomial_convertible_engine import (
    BinomialConvertibleEngine,
)
from pquantlib.processes.black_scholes_merton_process import BlackScholesMertonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import tight
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.schedule import MakeSchedule
from pquantlib.time.time_unit import TimeUnit


@pytest.fixture
def ref() -> dict[str, Any]:
    return load_reference("cluster/ws4")


@pytest.fixture
def today() -> Date:
    return Date.from_ymd(2, Month.January, 2020)


@pytest.fixture(autouse=True)
def _eval_date(today: Date) -> Any:  # pyright: ignore[reportUnusedFunction]
    prev = ObservableSettings().evaluation_date
    ObservableSettings().evaluation_date = today
    yield
    ObservableSettings().evaluation_date = prev


def _process(today: Date) -> BlackScholesMertonProcess:
    dc = Actual360()
    cal = TARGET()
    return BlackScholesMertonProcess(
        x0=SimpleQuote(50.0),
        dividend_ts=FlatForward.from_rate(today, 0.02, dc),
        risk_free_ts=FlatForward.from_rate(today, 0.05, dc),
        black_vol_ts=BlackConstantVol(
            reference_date=today, calendar=cal, volatility=0.15, day_counter=dc
        ),
    )


def _schedule(issue: Date, maturity: Date) -> Any:
    return (
        MakeSchedule()
        .from_date(issue)
        .to(maturity)
        .with_frequency(Frequency.Annual)
        .with_calendar(TARGET())
        .backwards()
        .build()
    )


def _dates(today: Date) -> tuple[Date, Date]:
    cal = TARGET()
    issue = cal.advance(today, 2, TimeUnit.Days)
    maturity = cal.advance(issue, 10, TimeUnit.Years)
    issue = cal.advance(maturity, -10, TimeUnit.Years)
    return issue, maturity


def test_european_no_callability(today: Date, ref: dict[str, Any]) -> None:
    issue, maturity = _dates(today)
    bond = ConvertibleFixedCouponBond(
        exercise=EuropeanExercise(maturity),
        conversion_ratio=ref["conversion_ratio"],
        callability=[],
        issue_date=issue,
        settlement_days=3,
        coupons=[0.05],
        day_counter=Actual360(),
        schedule=_schedule(issue, maturity),
        redemption=100.0,
    )
    bond.set_pricing_engine(
        BinomialConvertibleEngine(
            CoxRossRubinstein, _process(today), ref["time_steps"], SimpleQuote(0.005)
        )
    )
    tight(bond.npv(), ref["conv_fixed_eu"])


def test_american_no_callability(today: Date, ref: dict[str, Any]) -> None:
    issue, maturity = _dates(today)
    bond = ConvertibleFixedCouponBond(
        exercise=AmericanExercise(issue, maturity),
        conversion_ratio=ref["conversion_ratio"],
        callability=[],
        issue_date=issue,
        settlement_days=3,
        coupons=[0.05],
        day_counter=Actual360(),
        schedule=_schedule(issue, maturity),
        redemption=100.0,
    )
    bond.set_pricing_engine(
        BinomialConvertibleEngine(
            CoxRossRubinstein, _process(today), ref["time_steps"], SimpleQuote(0.005)
        )
    )
    tight(bond.npv(), ref["conv_fixed_am"])


def test_american_callput_dividends(today: Date, ref: dict[str, Any]) -> None:
    issue, maturity = _dates(today)
    cal = TARGET()
    call_date = cal.advance(issue, 5, TimeUnit.Years)
    put_date = cal.advance(issue, 7, TimeUnit.Years)
    div1 = cal.advance(issue, 1, TimeUnit.Years)
    div2 = cal.advance(issue, 3, TimeUnit.Years)

    callability = [
        SoftCallability(BondPrice(108.0, BondPriceType.Clean), call_date, 1.10),
        Callability(BondPrice(101.0, BondPriceType.Clean), CallabilityType.Put, put_date),
    ]
    dividends = [FixedDividend(1.0, div1), FixedDividend(1.5, div2)]

    bond = ConvertibleFixedCouponBond(
        exercise=AmericanExercise(issue, maturity),
        conversion_ratio=ref["conversion_ratio"],
        callability=callability,
        issue_date=issue,
        settlement_days=3,
        coupons=[0.05],
        day_counter=Actual360(),
        schedule=_schedule(issue, maturity),
        redemption=100.0,
    )
    bond.set_pricing_engine(
        BinomialConvertibleEngine(
            CoxRossRubinstein,
            _process(today),
            ref["time_steps"],
            SimpleQuote(0.005),
            dividends=dividends,
        )
    )
    tight(bond.npv(), ref["conv_fixed_am_callput"])
