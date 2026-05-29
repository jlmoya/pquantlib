"""Tests for MakeCDS fluent factory.

# C++ parity: ql/instruments/makecds.{hpp,cpp} ``MakeCreditDefaultSwap``.

Smoke tests:
- MakeCDS builds a CDS with sensible defaults.
- Chained setters propagate to the built instrument.
- Engine wiring via ``with_pricing_engine`` works end-to-end.
- A MakeCDS-built CDS reproduces the NPV of a manually-built CDS with
  the same schedule + leg setup (EXACT after equal engine assignment).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.instruments.credit_default_swap import (
    CreditDefaultSwap,
    ProtectionSide,
)
from pquantlib.instruments.make_cds import MakeCDS, make_cds
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.credit.midpoint_cds_engine import MidPointCdsEngine
from pquantlib.termstructures.credit.flat_hazard_rate import FlatHazardRate
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.tolerance import exact
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.weekends_only import WeekendsOnly
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


@pytest.fixture
def eval_date_fixture() -> Iterator[Date]:
    d = Date.from_ymd(15, Month.June, 2026)
    ObservableSettings().evaluation_date = d
    yield d
    ObservableSettings().evaluation_date = None


# --- construction guards ---------------------------------------------------


def test_make_cds_requires_exactly_one_of_termination_tenor_schedule() -> None:
    """C++ parity: three overloads cover (tenor), (termDate), (schedule)."""
    with pytest.raises(Exception, match="provide exactly one"):
        MakeCDS(running_spread=0.02)
    with pytest.raises(Exception, match="provide exactly one"):
        MakeCDS(tenor=Period(5, TimeUnit.Years),
                termination_date=Date.from_ymd(1, Month.January, 2030),
                running_spread=0.02)


def test_make_cds_via_tenor_builds(eval_date_fixture: Date) -> None:
    """Single-mode: tenor → builds default 5y CDS."""
    eval_date = eval_date_fixture
    cds = (
        MakeCDS(tenor=Period(5, TimeUnit.Years), running_spread=0.02)
        .with_calendar(WeekendsOnly())
        .with_rule(DateGeneration.TwentiethIMM)
        .build()
    )
    assert cds.side() == ProtectionSide.Buyer
    assert cds.notional() == 1.0
    assert cds.running_spread() == 0.02
    # Trade date defaults to current eval date.
    assert cds.trade_date() == eval_date


def test_make_cds_via_termination_date_builds(eval_date_fixture: Date) -> None:
    eval_date = eval_date_fixture
    term = eval_date + Period(5, TimeUnit.Years)
    cds = (
        MakeCDS(termination_date=term, running_spread=0.025)
        .with_calendar(WeekendsOnly())
        .with_rule(DateGeneration.Backward)  # non-IMM keeps termination exact
        .build()
    )
    assert cds.running_spread() == 0.025
    # Maturity should equal the termination date (no IMM-rule adjustment).
    assert cds.protection_end_date() == term


def test_make_cds_via_schedule_passes_through(eval_date_fixture: Date) -> None:
    from pquantlib.time.schedule import Schedule  # noqa: PLC0415

    eval_date = eval_date_fixture
    cal = WeekendsOnly()
    schedule = Schedule.from_rule(
        effective_date=eval_date,
        termination_date=eval_date + Period(3, TimeUnit.Years),
        tenor=Period.from_frequency(Frequency.Quarterly),
        calendar=cal,
        convention=BusinessDayConvention.Following,
        termination_date_convention=BusinessDayConvention.Unadjusted,
        rule=DateGeneration.TwentiethIMM,
        end_of_month=False,
    )
    cds = (
        MakeCDS(schedule=schedule, running_spread=0.03)
        .with_calendar(cal)
        .build()
    )
    # Coupons match the schedule.
    assert len(cds.coupons()) == len(schedule.dates) - 1


# --- chainable setter coverage --------------------------------------------


def test_make_cds_setters_propagate(eval_date_fixture: Date) -> None:
    eval_date = eval_date_fixture
    cds = (
        MakeCDS(tenor=Period(2, TimeUnit.Years), running_spread=0.02)
        .with_side(ProtectionSide.Seller)
        .with_notional(10_000_000.0)
        .with_coupon_tenor(Period(6, TimeUnit.Months))
        .with_calendar(WeekendsOnly())
        .with_convention(BusinessDayConvention.Following)
        .with_rule(DateGeneration.TwentiethIMM)
        .with_day_counter(Actual360())
        .settles_accrual(True)
        .pays_at_default_time(True)
        .with_trade_date(eval_date)
        .with_cash_settlement_days(3)
        .build()
    )
    assert cds.side() == ProtectionSide.Seller
    assert cds.notional() == 10_000_000.0
    assert cds.settles_accrual() is True
    assert cds.pays_at_default_time() is True


def test_make_cds_upfront_rate_uses_with_upfront_path(
    eval_date_fixture: Date,
) -> None:
    """Non-zero upfront_rate routes through CreditDefaultSwap.with_upfront."""
    cds = (
        MakeCDS(tenor=Period(5, TimeUnit.Years), running_spread=0.01)
        .with_upfront_rate(0.05)
        .with_calendar(WeekendsOnly())
        .with_rule(DateGeneration.TwentiethIMM)
        .build()
    )
    assert cds.upfront() == 0.05


def test_make_cds_engine_wires_correctly(eval_date_fixture: Date) -> None:
    """``with_pricing_engine`` produces a CDS that prices on the engine."""
    eval_date = eval_date_fixture
    cal = WeekendsOnly()
    dc365 = Actual365Fixed()
    discount = FlatForward.from_rate(
        eval_date, 0.03, dc365, Compounding.Continuous, Frequency.Annual,
    )
    probability = FlatHazardRate.from_rate(eval_date, 0.02, dc365)
    engine = MidPointCdsEngine(probability, 0.4, discount)

    cds = (
        MakeCDS(tenor=Period(5, TimeUnit.Years), running_spread=0.02)
        .with_calendar(cal)
        .with_notional(10_000_000.0)
        .with_rule(DateGeneration.TwentiethIMM)
        .with_pricing_engine(engine)
        .build()
    )
    # Should be priceable.
    _ = cds.npv()


# --- equivalence with manual CDS ------------------------------------------


def test_make_cds_npv_matches_manual_cds(eval_date_fixture: Date) -> None:
    """A MakeCDS-built CDS prices identically to a manually-built CDS.

    EXACT after assigning the same engine + the same schedule.
    """
    from pquantlib.time.schedule import Schedule  # noqa: PLC0415

    eval_date = eval_date_fixture
    cal = WeekendsOnly()
    bdc = BusinessDayConvention.Following
    dc360 = Actual360()
    dc365 = Actual365Fixed()
    notional = 10_000_000.0
    spread = 0.02
    schedule = Schedule.from_rule(
        effective_date=eval_date,
        termination_date=eval_date + Period(5, TimeUnit.Years),
        tenor=Period.from_frequency(Frequency.Quarterly),
        calendar=cal,
        convention=bdc,
        termination_date_convention=bdc,
        rule=DateGeneration.TwentiethIMM,
        end_of_month=False,
    )

    # Engine.
    discount = FlatForward.from_rate(
        eval_date, 0.03, dc365, Compounding.Continuous, Frequency.Annual,
    )
    probability = FlatHazardRate.from_rate(eval_date, 0.02, dc365)

    # Manual.
    cds_manual = CreditDefaultSwap(
        ProtectionSide.Buyer, notional, spread, schedule, bdc, dc360,
        settles_accrual=True,
        pays_at_default_time=True,
        protection_start=eval_date,
        claim=None,
        last_period_day_counter=None,
        rebates_accrual=True,
        trade_date=eval_date,
    )
    cds_manual.set_pricing_engine(MidPointCdsEngine(probability, 0.4, discount))

    # Factory — same schedule (forced via the schedule constructor mode).
    # The default coupon_tenor / convention from MakeCDS happen to match
    # the manual schedule under Quarterly + Following.
    cds_factory = (
        MakeCDS(schedule=schedule, running_spread=spread)
        .with_side(ProtectionSide.Buyer)
        .with_notional(notional)
        .with_calendar(cal)
        .with_convention(bdc)
        .with_day_counter(dc360)
        .with_protection_start(eval_date)
        .with_trade_date(eval_date)
        .with_pricing_engine(MidPointCdsEngine(probability, 0.4, discount))
        .build()
    )

    exact(cds_factory.npv(), cds_manual.npv())


# --- free-function alias --------------------------------------------------


def test_make_cds_free_function_constructs(eval_date_fixture: Date) -> None:
    """``make_cds(...)`` is a convenience alias for ``MakeCDS(...)``."""
    _ = eval_date_fixture
    builder = make_cds(tenor=Period(5, TimeUnit.Years), running_spread=0.02)
    assert isinstance(builder, MakeCDS)
    cds = (
        builder
        .with_calendar(WeekendsOnly())
        .with_rule(DateGeneration.TwentiethIMM)
        .build()
    )
    assert cds.running_spread() == 0.02
