"""Shared market + assertion helpers for the v1.43 coupon-driven bond ports.

Reproduces, exactly, the market built by
``migration-harness/cpp/probes/v143_inst_bondsamort/probe.cpp``:

* evaluation date 15-Jan-2024, TARGET calendar;
* flat 3% forecast / flat 4% discount curves (Actual/365 Fixed);
* Euribor6M plus a 10Y ``EuriborSwapIsdaFixA``-style swap index on it;
* historic fixings seeded over every TARGET business day of
  [1-Jan-2020, today] from the serial-number formula
  ``0.0250 + 1e-5 * (serial % 50)`` (IBOR) /
  ``0.0300 + 1e-5 * (serial % 50)`` (swap). The formula is deliberately
  date-dependent: a coupon that reads the *wrong* fixing date then reads a
  different rate, so a mis-forwarded ``fixing_days`` / ``in_arrears`` cannot
  hide behind a constant history;
* a constant 16% lognormal swaption vol driving an ``AnalyticHaganPricer``
  (the pricer the C++ test suite attaches to ``CmsRateBond``), and constant
  20% / 0% optionlet vols for the capped/floored and in-arrears IBOR legs.

This is a plain helper module rather than a ``conftest.py`` so that the generic
fixture names (``cpp``, ``market``) stay local to the modules that want them —
``test_btp.py`` in this same directory already binds both names to a different
probe.

The ``compare_bond`` helper asserts a bond against one probe entry: the
headline prices *and* the complete cashflow listing (payment date, amount,
ex-coupon date, and for coupons the nominal, accrual dates, accrual period,
fixing date and rate). An NPV alone can match while two errors cancel.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from typing import Any, cast

from pquantlib.cashflows.coupon import Coupon
from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
from pquantlib.cashflows.inflation_coupon import InflationCoupon
from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.indexes.swap_index import SwapIndex
from pquantlib.instruments.bond import Bond
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.bond.discounting_bond_engine import DiscountingBondEngine
from pquantlib.pricingengines.conundrum_pricer import (
    AnalyticHaganPricer,
    YieldCurveModel,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.volatility.optionlet.constant_optionlet_vol import (
    ConstantOptionletVolatility,
)
from pquantlib.termstructures.volatility.swaption.swaption_constant_vol import (
    SwaptionConstantVolatility,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

#: A tolerance-tier assertion — ``tolerance.tight`` or ``tolerance.loose``.
Check = Callable[[float, float], None]

TODAY: Date = Date.from_ymd(15, Month.January, 2024)
CMS_START: Date = Date.from_ymd(22, Month.August, 2020)
CMS_END: Date = Date.from_ymd(22, Month.August, 2030)
FRN_END: Date = Date.from_ymd(22, Month.August, 2025)

DC_A365: Actual365Fixed = Actual365Fixed()
DC_360: Actual360 = Actual360()
DC_30360: Thirty360 = Thirty360(Thirty360Convention.BondBasis)

#: Probe ``kEarlySettleProbe`` — earlier than every issue date used here, so
#: ``settlement_date(EARLY)`` exercises Bond's issue-date clamp.
EARLY: Date = Date.from_ymd(3, Month.January, 2006)

#: Probe ``gNotionalSamples`` for the nominal-denominated bonds.
NOTIONAL_SAMPLES: list[Date] = [
    Date.from_ymd(1, Month.January, 2021),
    Date.from_ymd(22, Month.August, 2022),
    Date.from_ymd(22, Month.August, 2025),
    Date.from_ymd(22, Month.August, 2028),
    Date.from_ymd(1, Month.January, 2031),
]

#: Probe ``kCustomNotionals10`` / ``kCustomNotionals10Frn``.
CUSTOM_NOTIONALS: list[float] = [
    100.0, 95.0, 88.0, 80.0, 71.0, 61.0, 50.0, 38.0, 25.0, 11.0,
]
CUSTOM_NOTIONALS_FRN: list[float] = [
    100.0, 96.0, 91.0, 85.0, 78.0, 70.0, 61.0, 51.0, 40.0, 28.0,
]


class CmsMarket:
    """The probe's nominal market: curves, indexes, engine and coupon pricers."""

    def __init__(self) -> None:
        forecast = FlatForward.from_rate(TODAY, 0.03, DC_A365)
        self.discount_curve: FlatForward = FlatForward.from_rate(TODAY, 0.04, DC_A365)
        self.ibor: Euribor = Euribor.six_months(
            cast(YieldTermStructureProtocol, forecast)
        )
        self.swap_index: SwapIndex = SwapIndex(
            "EuriborSwapIsdaFixA",
            Period(10, TimeUnit.Years),
            self.ibor.fixing_days(),
            EURCurrency(),
            self.ibor.fixing_calendar(),
            Period(1, TimeUnit.Years),
            BusinessDayConvention.Unadjusted,
            self.ibor.day_counter(),
            self.ibor,
        )
        calendar = TARGET()
        # The fixing history is process-wide state; start from a clean slate so
        # the module is independent of whatever other test modules seeded.
        self.ibor.clear_fixings()
        self.swap_index.clear_fixings()
        day = Date.from_ymd(1, Month.January, 2020)
        while day <= TODAY:
            if calendar.is_business_day(day):
                offset = 0.00001 * float(day.serial_number() % 50)
                self.ibor.add_fixing(day, 0.0250 + offset, True)
                self.swap_index.add_fixing(day, 0.0300 + offset, True)
            day = day + 1

        self.engine: DiscountingBondEngine = DiscountingBondEngine(self.discount_curve)
        self.cms_pricer: AnalyticHaganPricer = AnalyticHaganPricer(
            SwaptionConstantVolatility(
                reference_date=TODAY,
                calendar=calendar,
                business_day_convention=BusinessDayConvention.Following,
                volatility=0.16,
                day_counter=DC_A365,
                volatility_type=VolatilityType.ShiftedLognormal,
            ),
            YieldCurveModel.Standard,
            SimpleQuote(0.0),
        )
        self.optionlet_vol_20: ConstantOptionletVolatility = ConstantOptionletVolatility(
            reference_date=TODAY,
            calendar=calendar,
            business_day_convention=BusinessDayConvention.Following,
            volatility=0.20,
            day_counter=DC_A365,
        )
        self.optionlet_vol_0: ConstantOptionletVolatility = ConstantOptionletVolatility(
            reference_date=TODAY,
            calendar=calendar,
            business_day_convention=BusinessDayConvention.Following,
            volatility=0.0,
            day_counter=DC_A365,
        )


def load_reference() -> dict[str, Any]:
    """The whole ``v143/inst/bondsamort`` probe output."""
    return reference_reader.load("v143/inst/bondsamort")


@contextmanager
def pinned_evaluation_date(date: Date) -> Generator[None]:
    """Pin ``Settings::evaluationDate`` for the duration of the block."""
    settings = ObservableSettings()
    saved = settings.evaluation_date
    settings.evaluation_date = date
    try:
        yield
    finally:
        settings.evaluation_date = saved


def build_market() -> CmsMarket:
    """Build the probe's market. Seeds the process-wide fixing histories."""
    with pinned_evaluation_date(TODAY):
        return CmsMarket()


def cms_schedule() -> Schedule:
    """Probe ``cmsSchedule()`` — 10 annual periods, 22-Aug-2020 → 22-Aug-2030."""
    return Schedule.from_rule(
        effective_date=CMS_START,
        termination_date=CMS_END,
        tenor=Period(1, TimeUnit.Years),
        calendar=TARGET(),
        convention=BusinessDayConvention.Unadjusted,
        termination_date_convention=BusinessDayConvention.Unadjusted,
        rule=DateGeneration.Backward,
        end_of_month=False,
    )


def frn_schedule() -> Schedule:
    """Probe ``frnSchedule()`` — 10 semiannual periods, 22-Aug-2020 → 22-Aug-2025."""
    return Schedule.from_rule(
        effective_date=CMS_START,
        termination_date=FRN_END,
        tenor=Period(6, TimeUnit.Months),
        calendar=TARGET(),
        convention=BusinessDayConvention.Unadjusted,
        termination_date_convention=BusinessDayConvention.Unadjusted,
        rule=DateGeneration.Backward,
        end_of_month=False,
    )


def _fixing_serial(cf: object) -> int:
    if isinstance(cf, FloatingRateCoupon | InflationCoupon):
        return cf.fixing_date().serial_number()
    return 0


def compare_cashflows(
    bond: Bond, ref: dict[str, Any], check: Check
) -> None:
    """Assert the full cashflow listing against the probe entry."""
    flows = bond.cashflows()
    expected = ref["cashflows"]
    assert len(flows) == len(expected)
    for i, (cf, rc) in enumerate(zip(flows, expected, strict=True)):
        assert cf.date().serial_number() == rc["date_serial"], f"cf[{i}] payment date"
        check(cf.amount(), rc["amount"])
        ex = cf.ex_coupon_date()
        ex_serial = 0 if ex == Date() else ex.serial_number()
        assert ex_serial == rc["ex_coupon_serial"], f"cf[{i}] ex-coupon date"
        assert isinstance(cf, Coupon) is rc["is_coupon"], f"cf[{i}] coupon-ness"
        if not rc["is_coupon"]:
            continue
        coupon = cast(Coupon, cf)
        check(coupon.nominal(), rc["nominal"])
        assert coupon.accrual_start_date().serial_number() == rc["accrual_start_serial"]
        assert coupon.accrual_end_date().serial_number() == rc["accrual_end_serial"]
        check(coupon.accrual_period(), rc["accrual_period"])
        assert _fixing_serial(cf) == rc["fixing_serial"], f"cf[{i}] fixing date"
        check(coupon.rate(), rc["rate"])


def compare_bond(
    bond: Bond,
    ref: dict[str, Any],
    check: Check,
    samples: Sequence[Date] = tuple(NOTIONAL_SAMPLES),
) -> None:
    """Assert every pinned property of ``bond`` against one probe entry."""
    assert bond.settlement_date().serial_number() == ref["settlement_serial"]
    assert (
        bond.settlement_date(EARLY).serial_number()
        == ref["settlement_before_issue_serial"]
    )
    assert bond.start_date().serial_number() == ref["start_serial"]
    assert bond.maturity_date().serial_number() == ref["maturity_serial"]
    issue = bond.issue_date()
    assert (0 if issue == Date() else issue.serial_number()) == ref["issue_serial"]
    assert len(bond.cashflows()) == ref["n_cashflows"]
    assert len(bond.redemptions()) == ref["n_redemptions"]
    assert bond.is_tradable() is ref["is_tradable"]

    check(bond.npv(), ref["npv"])
    check(bond.clean_price(), ref["clean_price"])
    check(bond.dirty_price(), ref["dirty_price"])
    check(bond.accrued_amount(), ref["accrued"])
    check(bond.settlement_value(), ref["settlement_value"])

    notionals = bond.notionals()
    assert len(notionals) == len(ref["notionals"])
    for actual, expected in zip(notionals, ref["notionals"], strict=True):
        check(actual, expected)
    for date, entry in zip(samples, ref["notional_at"], strict=True):
        assert date.serial_number() == entry["date_serial"]
        check(bond.notional(date), entry["value"])

    compare_cashflows(bond, ref, check)
