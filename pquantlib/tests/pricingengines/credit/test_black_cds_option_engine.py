"""Cross-validate BlackCDSOptionEngine against the C++ reference.

Probe source: migration-harness/cpp/probes/cluster_w3d/probe.cpp
Reference:    migration-harness/references/cluster/w3d.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.cds_option import CDSOption
from pquantlib.instruments.credit_default_swap import (
    CreditDefaultSwap,
    ProtectionSide,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.credit.black_cds_option_engine import (
    BlackCDSOptionEngine,
)
from pquantlib.pricingengines.credit.midpoint_cds_engine import MidPointCdsEngine
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.credit.flat_hazard_rate import FlatHazardRate
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import MakeSchedule
from pquantlib.time.time_unit import TimeUnit


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w3d")


@pytest.fixture(autouse=True)
def _set_eval_date() -> None:  # pyright: ignore[reportUnusedFunction]
    """Pin the evaluation date to the curve reference date."""
    ObservableSettings().evaluation_date = Date.from_ymd(15, Month.January, 2024)


def _build_cds() -> tuple[CreditDefaultSwap, FlatForward, FlatHazardRate]:
    today = Date.from_ymd(15, Month.January, 2024)
    cal = TARGET()
    dc = Actual365Fixed()
    yts = FlatForward(today, SimpleQuote(0.03), dc,
                      Compounding.Continuous, Frequency.Annual)
    dts = FlatHazardRate(today, SimpleQuote(0.02), dc)
    exercise = cal.advance(today, 1, TimeUnit.Years)
    end = cal.advance(exercise, 5, TimeUnit.Years)
    sched = (
        MakeSchedule()
        .from_date(exercise)
        .to(end)
        .with_calendar(cal)
        .with_tenor(Period(3, TimeUnit.Months))
        .with_convention(BusinessDayConvention.Following)
        .with_rule(DateGeneration.CDS2015)
        .build()
    )
    cds = CreditDefaultSwap(
        side=ProtectionSide.Buyer,
        notional=1.0e7,
        spread=0.0100,
        schedule=sched,
        payment_convention=BusinessDayConvention.Following,
        day_counter=dc,
        settles_accrual=True,
        pays_at_default_time=True,
        protection_start=exercise,
    )
    cds.set_pricing_engine(MidPointCdsEngine(dts, 0.4, yts))
    return cds, yts, dts


def test_black_cds_option_knockout_npv_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    """Knock-out payer CDS option NPV matches the C++ Black-76 value.

    LOOSE tier: the schedule generation routes through the
    CDS2015 date-generation rule whose Python port has been verified
    to <= 1e-8 difference vs the C++ run, but the cumulative
    discount + survival product can pick up small floating-point
    rounding through the (still numerically equivalent) Black-76 ops.
    """
    cds, yts, dts = _build_cds()
    today = Date.from_ymd(15, Month.January, 2024)
    cal = TARGET()
    exercise_date = cal.advance(today, 1, TimeUnit.Years)
    opt = CDSOption(
        underlying=cds,
        exercise=EuropeanExercise(exercise_date),
        knocks_out=True,
    )
    vol = SimpleQuote(0.30)
    opt.set_pricing_engine(BlackCDSOptionEngine(dts, 0.4, yts, vol))

    ref = cpp_ref["cds_option"]
    tolerance.loose(opt.npv(), ref["npv_knockout"])
    tolerance.loose(opt.risky_annuity(), ref["risky_annuity"])
    tolerance.loose(opt.atm_rate(), ref["atm_rate"])


def test_black_cds_option_non_knockout_npv_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    """Non-knock-out payer CDS option = knock-out NPV +
    front-end-protection contribution. Matches C++ to LOOSE tier.
    """
    cds, yts, dts = _build_cds()
    today = Date.from_ymd(15, Month.January, 2024)
    cal = TARGET()
    exercise_date = cal.advance(today, 1, TimeUnit.Years)
    opt = CDSOption(
        underlying=cds,
        exercise=EuropeanExercise(exercise_date),
        knocks_out=False,
    )
    vol = SimpleQuote(0.30)
    opt.set_pricing_engine(BlackCDSOptionEngine(dts, 0.4, yts, vol))

    ref = cpp_ref["cds_option"]
    tolerance.loose(opt.npv(), ref["npv_non_knockout"])


def test_cds_option_receiver_must_knock_out() -> None:
    """Constructing a receiver (Seller-side underlying) without knock-out
    must raise.

    # C++ parity: cdsoption.cpp:74-75.
    """
    today = Date.from_ymd(15, Month.January, 2024)
    cal = TARGET()
    dc = Actual365Fixed()
    exercise = cal.advance(today, 1, TimeUnit.Years)
    end = cal.advance(exercise, 5, TimeUnit.Years)
    sched = (
        MakeSchedule()
        .from_date(exercise)
        .to(end)
        .with_calendar(cal)
        .with_tenor(Period(3, TimeUnit.Months))
        .with_convention(BusinessDayConvention.Following)
        .with_rule(DateGeneration.CDS2015)
        .build()
    )
    cds_seller = CreditDefaultSwap(
        side=ProtectionSide.Seller,
        notional=1.0e7,
        spread=0.0100,
        schedule=sched,
        payment_convention=BusinessDayConvention.Following,
        day_counter=dc,
        protection_start=exercise,
    )

    with pytest.raises(LibraryException):
        CDSOption(
            underlying=cds_seller,
            exercise=EuropeanExercise(exercise),
            knocks_out=False,
        )


def test_cds_option_upfront_underlying_rejected() -> None:
    """Underlying with upfront raises ``LibraryException``.

    # C++ parity: cdsoption.cpp:76 — ``QL_REQUIRE(!swap->upfront(), ...)``.
    """
    today = Date.from_ymd(15, Month.January, 2024)
    cal = TARGET()
    dc = Actual365Fixed()
    exercise = cal.advance(today, 1, TimeUnit.Years)
    end = cal.advance(exercise, 5, TimeUnit.Years)
    sched = (
        MakeSchedule()
        .from_date(exercise)
        .to(end)
        .with_calendar(cal)
        .with_tenor(Period(3, TimeUnit.Months))
        .with_convention(BusinessDayConvention.Following)
        .with_rule(DateGeneration.CDS2015)
        .build()
    )
    cds_upf = CreditDefaultSwap.with_upfront(
        side=ProtectionSide.Buyer,
        notional=1.0e7,
        upfront=0.001,
        spread=0.0100,
        schedule=sched,
        payment_convention=BusinessDayConvention.Following,
        day_counter=dc,
        protection_start=exercise,
    )

    with pytest.raises(LibraryException):
        CDSOption(
            underlying=cds_upf,
            exercise=EuropeanExercise(exercise),
        )
