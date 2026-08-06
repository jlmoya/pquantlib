"""Cross-validate the eight v1.43 chained leg builders against C++.

Probe: ``v143/cf/legs`` — ``FixedRateLeg``, ``IborLeg``, ``OvernightLeg``,
``CmsLeg``, ``CPILeg``, ``DigitalIborLeg``, ``DigitalCmsLeg``, ``AverageBMALeg``.

A leg builder is a pile of optional setters, and the failure this file exists
to catch is a setter that is *accepted and dropped*. So the reference pins the
whole per-coupon listing — runtime type, payment date, nominal, accrual
start/end, reference period, accrual period, rate, amount, plus fixing date,
gearing, spread, ex-coupon date, cap/floor and digital strikes where the coupon
has them — and :func:`_assert_leg` checks every field of every coupon. Each
builder contributes one all-defaults leg plus one leg per setter (or setter
group) moved off its default, so a dropped setter shows up as a diff against
the C++ listing rather than as a still-plausible number.

Where C++ itself cannot produce a rate — a capped/floored leg that C++
deliberately leaves without a pricer, a CMS leg with no Hagan pricer — the
reference records ``null`` and the assertion checks that Python is *equally*
unable to produce one, so an accidental value would fail too.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest

from pquantlib.cashflows.average_bma_coupon import AverageBMALeg
from pquantlib.cashflows.capped_floored_coupon import (
    CappedFlooredCoupon,
    CappedFlooredOvernightIndexedCoupon,
)
from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.cms_coupon import CmsLeg
from pquantlib.cashflows.coupon import Coupon
from pquantlib.cashflows.coupon_pricer import BlackIborCouponPricer, set_coupon_pricer
from pquantlib.cashflows.cpi_coupon import CPICashFlow, CPICoupon, CPILeg
from pquantlib.cashflows.digital_cms_coupon import DigitalCmsLeg
from pquantlib.cashflows.digital_coupon import DigitalCoupon
from pquantlib.cashflows.digital_ibor_coupon import DigitalIborLeg
from pquantlib.cashflows.fixed_rate_coupon import FixedRateLeg
from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
from pquantlib.cashflows.ibor_coupon import IborLeg
from pquantlib.cashflows.overnight_indexed_coupon import (
    OvernightIndexedCoupon,
    OvernightLeg,
)
from pquantlib.cashflows.overnight_indexed_coupon_pricer import (
    ArithmeticAveragedOvernightIndexedCouponPricer,
)
from pquantlib.cashflows.rate_averaging import RateAveraging
from pquantlib.cashflows.replication import DigitalReplication, Replication
from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.bma_index import BMAIndex
from pquantlib.indexes.ibor.eonia import Eonia
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.indexes.inflation.cpi import InterpolationType
from pquantlib.indexes.inflation.eu_hicp import EUHICP
from pquantlib.indexes.swap_index import SwapIndex
from pquantlib.interest_rate import InterestRate
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.position import PositionType
from pquantlib.termstructures.volatility.optionlet.constant_optionlet_vol import (
    ConstantOptionletVolatility,
)
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
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

# --- probe constants (must mirror v143_cf_legs/probe.cpp exactly) -----------

_TODAY = Date.from_ymd(15, Month.January, 2024)
_FLAT_RATE = 0.03
_NOMINAL = 1_000_000.0
_CAPLET_VOL = 0.15

_BDC = BusinessDayConvention
_MF = _BDC.ModifiedFollowing
_FOLLOWING = _BDC.Following
_UNADJUSTED = _BDC.Unadjusted
_PRECEDING = _BDC.Preceding


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/cf/legs")


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    ObservableSettings().evaluation_date = _TODAY
    yield
    ObservableSettings().evaluation_date = None


# --- market data mirroring the probe ---------------------------------------


def _curve() -> FlatForward:
    return FlatForward.from_rate(_TODAY, _FLAT_RATE, Actual365Fixed())


def _euribor6m() -> Euribor:
    return Euribor.six_months(_curve())


def _eonia() -> Eonia:
    return Eonia(_curve())


def _cms10y() -> SwapIndex:
    ibor = _euribor6m()
    return SwapIndex(
        "EuriborSwapIsdaFixA",
        Period(10, TimeUnit.Years),
        ibor.fixing_days(),
        EURCurrency(),
        ibor.fixing_calendar(),
        Period(1, TimeUnit.Years),
        _UNADJUSTED,
        ibor.day_counter(),
        ibor,
    )


def _bma() -> BMAIndex:
    return BMAIndex(_curve())


def _black_pricer() -> BlackIborCouponPricer:
    vol = ConstantOptionletVolatility(
        reference_date=_TODAY,
        calendar=TARGET(),
        business_day_convention=_FOLLOWING,
        volatility=_CAPLET_VOL,
        day_counter=Actual365Fixed(),
    )
    return BlackIborCouponPricer(vol)


def _sched(
    d1: Date,
    d2: Date,
    tenor: Period,
    convention: BusinessDayConvention = _MF,
    rule: DateGeneration = DateGeneration.Forward,
) -> Schedule:
    return Schedule.from_rule(
        effective_date=d1,
        termination_date=d2,
        tenor=tenor,
        calendar=TARGET(),
        convention=convention,
        termination_date_convention=convention,
        rule=rule,
        end_of_month=False,
    )


def _main_schedule() -> Schedule:
    return _sched(
        Date.from_ymd(15, Month.February, 2024),
        Date.from_ymd(15, Month.February, 2026),
        Period(6, TimeUnit.Months),
    )


def _quarterly_schedule() -> Schedule:
    return _sched(
        Date.from_ymd(15, Month.February, 2024),
        Date.from_ymd(15, Month.February, 2025),
        Period(3, TimeUnit.Months),
    )


def _stub_schedule() -> Schedule:
    return _sched(
        Date.from_ymd(20, Month.March, 2024),
        Date.from_ymd(15, Month.February, 2026),
        Period(6, TimeUnit.Months),
        rule=DateGeneration.Backward,
    )


def _unadjusted_schedule() -> Schedule:
    return _sched(
        Date.from_ymd(15, Month.May, 2024),
        Date.from_ymd(15, Month.November, 2024),
        Period(1, TimeUnit.Months),
        convention=_UNADJUSTED,
    )


def _unadjusted_past_schedule() -> Schedule:
    return _sched(
        Date.from_ymd(15, Month.May, 2021),
        Date.from_ymd(15, Month.November, 2021),
        Period(1, TimeUnit.Months),
        convention=_UNADJUSTED,
    )


def _bma_schedule() -> Schedule:
    return _sched(
        Date.from_ymd(15, Month.February, 2024),
        Date.from_ymd(15, Month.February, 2025),
        Period(3, TimeUnit.Months),
    )


def _bma_stub_schedule() -> Schedule:
    return _sched(
        Date.from_ymd(20, Month.March, 2024),
        Date.from_ymd(15, Month.February, 2025),
        Period(3, TimeUnit.Months),
        rule=DateGeneration.Backward,
    )


def _ramp_fixing(year: int, month: int) -> float:
    """Mirror the probe's ``rampFixing``."""
    return 100.0 + 0.5 * (12 * (year - 2018) + (month - 1))


@pytest.fixture
def seeded_eu_hicp() -> Iterator[EUHICP]:
    """EUHICP with the probe's deterministic 2018-2023 ramp history."""
    idx = EUHICP()
    idx.clear_fixings()
    for y in range(2018, 2024):
        for m in range(1, 13):
            idx.add_fixing(Date.from_ymd(1, Month(m), y), _ramp_fixing(y, m), True)
    yield idx
    idx.clear_fixings()


_CPI_SCHEDULE_START = Date.from_ymd(15, Month.January, 2021)
_CPI_SCHEDULE_END = Date.from_ymd(15, Month.January, 2023)
_CPI_LAG = Period(3, TimeUnit.Months)
_CPI_BASE = _ramp_fixing(2020, 10)
_CPI_BASE_DATE = Date.from_ymd(15, Month.October, 2020)


def _cpi_schedule() -> Schedule:
    return _sched(_CPI_SCHEDULE_START, _CPI_SCHEDULE_END, Period(1, TimeUnit.Years))


# --- assertion helper -------------------------------------------------------


def _assert_maybe(actual: float | None, expected: float | None, what: str) -> None:
    """TIGHT-compare, treating C++ ``Null<Real>`` / a throwing accessor as None."""
    if expected is None:
        assert actual is None, f"{what}: expected C++ null, got {actual!r}"
        return
    assert actual is not None, f"{what}: expected {expected!r}, got None"
    tolerance.tight(actual, expected, reason=what)


def _safe_rate(c: Coupon) -> float | None:
    try:
        return c.rate()
    except LibraryException:
        return None


def _safe_amount(c: CashFlow) -> float | None:
    try:
        return c.amount()
    except LibraryException:
        return None


def _safe_fixing_date(c: FloatingRateCoupon) -> int | None:
    try:
        return c.fixing_date().serial_number()
    except LibraryException:
        return None


def _assert_coupon(cf: Coupon, r: dict[str, Any], where: str, compare_rates: bool) -> None:
    tolerance.exact(cf.nominal(), r["nominal"], reason=f"{where}.nominal")
    assert cf.accrual_start_date().serial_number() == r["accrual_start"]
    assert cf.accrual_end_date().serial_number() == r["accrual_end"]
    assert cf.reference_period_start().serial_number() == r["ref_period_start"]
    assert cf.reference_period_end().serial_number() == r["ref_period_end"]
    tolerance.tight(
        cf.accrual_period(), r["accrual_period"], reason=f"{where}.accrual_period"
    )
    ex = cf.ex_coupon_date()
    assert (None if ex == Date() else ex.serial_number()) == r["ex_coupon_date"], (
        f"{where}: ex-coupon date"
    )
    if compare_rates:
        _assert_maybe(_safe_rate(cf), r["rate"], f"{where}.rate")


def _assert_floating(cf: FloatingRateCoupon, r: dict[str, Any], where: str) -> None:
    tolerance.exact(cf.gearing(), r["gearing"], reason=f"{where}.gearing")
    tolerance.exact(cf.spread(), r["spread"], reason=f"{where}.spread")
    assert cf.is_in_arrears() == r["is_in_arrears"], f"{where}: in-arrears"
    assert cf.fixing_days() == int(r["fixing_days"]), f"{where}: fixing days"
    assert _safe_fixing_date(cf) == r["fixing_date"], f"{where}: fixing date"


def _assert_capped(cf: CappedFlooredCoupon, r: dict[str, Any], where: str) -> None:
    assert cf.is_capped() == r["is_capped"], f"{where}: is_capped"
    assert cf.is_floored() == r["is_floored"], f"{where}: is_floored"
    _assert_maybe(cf.cap(), r["cap"], f"{where}.cap")
    _assert_maybe(cf.floor(), r["floor"], f"{where}.floor")


def _assert_overnight(cf: OvernightIndexedCoupon, r: dict[str, Any], where: str) -> None:
    assert (
        cf.rate_computation_start_date().serial_number() == r["rate_computation_start"]
    ), f"{where}: rate computation start"
    assert cf.rate_computation_end_date().serial_number() == r["rate_computation_end"], (
        f"{where}: rate computation end"
    )
    assert len(cf.fixing_dates()) == int(r["n_fixings"]), f"{where}: n fixings"
    assert int(cf.averaging_method()) == r["averaging_method"], (
        f"{where}: averaging method"
    )


def _assert_digital(cf: DigitalCoupon, r: dict[str, Any], where: str) -> None:
    assert cf.has_call() == r["has_call"], f"{where}: has_call"
    assert cf.has_put() == r["has_put"], f"{where}: has_put"
    assert cf.is_long_call() == r["is_long_call"], f"{where}: is_long_call"
    assert cf.is_long_put() == r["is_long_put"], f"{where}: is_long_put"
    _assert_maybe(cf.call_strike(), r["call_strike"], f"{where}.call_strike")
    _assert_maybe(cf.put_strike(), r["put_strike"], f"{where}.put_strike")
    _assert_maybe(
        cf.call_digital_payoff(), r["call_digital_payoff"], f"{where}.call_payoff"
    )
    _assert_maybe(cf.put_digital_payoff(), r["put_digital_payoff"], f"{where}.put_payoff")


def _assert_cpi(cf: CPICoupon, r: dict[str, Any], where: str) -> None:
    tolerance.exact(cf.fixed_rate(), r["fixed_rate"], reason=f"{where}.fixed_rate")
    _assert_maybe(cf.base_cpi(), r["base_cpi"], f"{where}.base_cpi")
    bd = cf.base_date()
    assert (None if bd == Date() else bd.serial_number()) == r["base_date"]
    tolerance.tight(cf.index_fixing(), r["index_fixing"], reason=f"{where}.index_fixing")
    assert int(cf.observation_interpolation()) == r["observation_interpolation"]


def _assert_cpi_cashflow(cf: CPICashFlow, r: dict[str, Any], where: str) -> None:
    tolerance.exact(cf.notional(), r["notional"], reason=f"{where}.notional")
    assert cf.observation_date().serial_number() == r["observation_date"]
    assert cf.growth_only() == r["growth_only"], f"{where}: growth_only"
    tolerance.tight(cf.index_fixing(), r["index_fixing"], reason=f"{where}.index_fixing")


def _assert_leg(
    leg: list[CashFlow], ref: list[dict[str, Any]], *, compare_rates: bool = True
) -> None:
    """Assert every emitted field of every coupon against the C++ listing.

    ``compare_rates=False`` restricts the comparison to structure (dates,
    nominals, accruals, gearings, strikes, ...) for the one leg whose *rate*
    depends on a documented carve-out; see
    ``test_digital_ibor_in_arrears_rate_needs_convexity_adjustment``.
    """
    assert len(leg) == len(ref), f"leg size {len(leg)} != C++ {len(ref)}"
    for i, (cf, r) in enumerate(zip(leg, ref, strict=True)):
        where = f"coupon[{i}]"
        assert type(cf).__name__ == r["type"], (
            f"{where}: type {type(cf).__name__} != C++ {r['type']}"
        )
        assert cf.date().serial_number() == r["payment_date"], f"{where}: payment date"
        if compare_rates:
            _assert_maybe(_safe_amount(cf), r["amount"], f"{where}.amount")
        if isinstance(cf, Coupon):
            _assert_coupon(cf, r, where, compare_rates)
        if isinstance(cf, FloatingRateCoupon):
            _assert_floating(cf, r, where)
        if isinstance(cf, CappedFlooredCoupon):
            _assert_capped(cf, r, where)
        if isinstance(cf, CappedFlooredOvernightIndexedCoupon):
            assert cf.naked_option() == r["naked_option"], f"{where}: naked_option"
        if isinstance(cf, OvernightIndexedCoupon):
            _assert_overnight(cf, r, where)
        if isinstance(cf, DigitalCoupon):
            _assert_digital(cf, r, where)
        if isinstance(cf, CPICoupon):
            _assert_cpi(cf, r, where)
        if isinstance(cf, CPICashFlow):
            _assert_cpi_cashflow(cf, r, where)


# ===========================================================================
# FixedRateLeg — with_notionals, with_coupon_rates (4 overloads),
# with_payment_adjustment, with_first_period_day_counter,
# with_last_period_day_counter, with_payment_calendar, with_payment_lag,
# with_ex_coupon_period
# ===========================================================================


def _fixed_default() -> list[CashFlow]:
    return (
        FixedRateLeg(_main_schedule())
        .with_notionals(_NOMINAL)
        .with_coupon_rates(0.035, Actual360())
        .build()
    )


def _fixed_full() -> list[CashFlow]:
    return (
        FixedRateLeg(_main_schedule())
        .with_notionals([_NOMINAL, 2.0 * _NOMINAL])
        .with_coupon_rates(
            [0.03, 0.04], Actual360(), Compounding.Compounded, Frequency.Semiannual
        )
        .with_payment_adjustment(_MF)
        .with_first_period_day_counter(Thirty360(Thirty360Convention.BondBasis))
        .with_last_period_day_counter(Actual365Fixed())
        .with_payment_calendar(TARGET())
        .with_payment_lag(3)
        .with_ex_coupon_period(Period(2, TimeUnit.Days), TARGET(), _PRECEDING, False)
        .build()
    )


def _fixed_interest_rate() -> list[CashFlow]:
    return (
        FixedRateLeg(_main_schedule())
        .with_notionals(_NOMINAL)
        .with_coupon_rates(
            InterestRate(0.037, Actual365Fixed(), Compounding.Compounded, Frequency.Annual)
        )
        .build()
    )


def _fixed_interest_rate_vector() -> list[CashFlow]:
    return (
        FixedRateLeg(_main_schedule())
        .with_notionals(_NOMINAL)
        .with_coupon_rates(
            [
                InterestRate(0.031, Actual360(), Compounding.Simple, Frequency.Annual),
                InterestRate(
                    0.041,
                    Thirty360(Thirty360Convention.BondBasis),
                    Compounding.Compounded,
                    Frequency.Semiannual,
                ),
            ]
        )
        .build()
    )


def _fixed_stub() -> list[CashFlow]:
    return (
        FixedRateLeg(_stub_schedule())
        .with_notionals(_NOMINAL)
        .with_coupon_rates(0.035, Actual360())
        .with_first_period_day_counter(Thirty360(Thirty360Convention.BondBasis))
        .with_last_period_day_counter(Actual365Fixed())
        .build()
    )


def _fixed_adj(convention: BusinessDayConvention) -> list[CashFlow]:
    return (
        FixedRateLeg(_unadjusted_schedule())
        .with_notionals(_NOMINAL)
        .with_coupon_rates(0.035, Actual360())
        .with_payment_adjustment(convention)
        .build()
    )


# ===========================================================================
# IborLeg — with_notionals, with_payment_day_counter, with_payment_adjustment,
# with_payment_lag, with_payment_calendar, with_fixing_days, with_gearings,
# with_spreads, with_caps, with_floors, in_arrears, with_zero_payments,
# with_ex_coupon_period, with_fixing_convention, with_indexed_coupons,
# with_at_par_coupons
# ===========================================================================


def _ibor_default() -> list[CashFlow]:
    return IborLeg(_main_schedule(), _euribor6m()).with_notionals(_NOMINAL).build()


def _ibor_full() -> list[CashFlow]:
    return (
        IborLeg(_main_schedule(), _euribor6m())
        .with_notionals([_NOMINAL, 2.0 * _NOMINAL])
        .with_payment_day_counter(Actual360())
        .with_payment_adjustment(_MF)
        .with_payment_lag(2)
        .with_payment_calendar(TARGET())
        .with_fixing_days(1)
        .with_gearings(1.5)
        .with_spreads(0.002)
        .with_fixing_convention(_FOLLOWING)
        .with_ex_coupon_period(Period(3, TimeUnit.Days), TARGET(), _PRECEDING, False)
        .build()
    )


def _ibor_zero_payments() -> list[CashFlow]:
    return (
        IborLeg(_main_schedule(), _euribor6m())
        .with_notionals(_NOMINAL)
        .with_payment_lag(2)
        .with_zero_payments(True)
        .build()
    )


def _ibor_capped() -> list[CashFlow]:
    return (
        IborLeg(_main_schedule(), _euribor6m())
        .with_notionals(_NOMINAL)
        .with_caps(0.035)
        .with_floors(0.01)
        .build()
    )


def _ibor_in_arrears() -> list[CashFlow]:
    return (
        IborLeg(_main_schedule(), _euribor6m())
        .with_notionals(_NOMINAL)
        .in_arrears(True)
        .build()
    )


def _ibor_zero_gearing() -> list[CashFlow]:
    return (
        IborLeg(_main_schedule(), _euribor6m())
        .with_notionals(_NOMINAL)
        .with_payment_day_counter(Actual360())
        .with_gearings(0.0)
        .with_spreads(0.025)
        .with_floors(0.03)
        .build()
    )


def _ibor_quarterly_default() -> list[CashFlow]:
    return IborLeg(_quarterly_schedule(), _euribor6m()).with_notionals(_NOMINAL).build()


def _ibor_indexed_coupons() -> list[CashFlow]:
    return (
        IborLeg(_quarterly_schedule(), _euribor6m())
        .with_notionals(_NOMINAL)
        .with_indexed_coupons(True)
        .build()
    )


def _ibor_at_par_coupons() -> list[CashFlow]:
    return (
        IborLeg(_quarterly_schedule(), _euribor6m())
        .with_notionals(_NOMINAL)
        .with_at_par_coupons(True)
        .build()
    )


def _ibor_adj(convention: BusinessDayConvention) -> list[CashFlow]:
    return (
        IborLeg(_unadjusted_schedule(), _euribor6m())
        .with_notionals(_NOMINAL)
        .with_payment_adjustment(convention)
        .build()
    )


def _ibor_stub() -> list[CashFlow]:
    return IborLeg(_stub_schedule(), _euribor6m()).with_notionals(_NOMINAL).build()


# ===========================================================================
# OvernightLeg — with_notionals, with_payment_day_counter,
# with_payment_adjustment, with_payment_calendar, with_payment_lag,
# with_gearings, with_spreads, with_averaging_method, with_caps, with_floors,
# with_naked_option, in_arrears, with_last_recent_period,
# with_last_recent_period_calendar, with_payment_dates, with_coupon_pricer
# ===========================================================================


def _on_default() -> list[CashFlow]:
    return OvernightLeg(_main_schedule(), _eonia()).with_notionals(_NOMINAL).build()


def _on_full() -> list[CashFlow]:
    return (
        OvernightLeg(_main_schedule(), _eonia())
        .with_notionals([_NOMINAL, 2.0 * _NOMINAL])
        .with_payment_day_counter(Actual360())
        .with_payment_adjustment(_MF)
        .with_payment_calendar(TARGET())
        .with_payment_lag(2)
        .with_gearings(1.5)
        .with_spreads(0.001)
        .build()
    )


def _on_simple_averaging() -> list[CashFlow]:
    return (
        OvernightLeg(_main_schedule(), _eonia())
        .with_notionals(_NOMINAL)
        .with_averaging_method(RateAveraging.Simple)
        .build()
    )


def _on_in_advance() -> list[CashFlow]:
    return (
        OvernightLeg(_main_schedule(), _eonia())
        .with_notionals(_NOMINAL)
        .in_arrears(False)
        .build()
    )


def _on_last_recent() -> list[CashFlow]:
    return (
        OvernightLeg(_main_schedule(), _eonia())
        .with_notionals(_NOMINAL)
        .with_last_recent_period(Period(1, TimeUnit.Months))
        .with_last_recent_period_calendar(TARGET())
        .build()
    )


def _on_payment_dates() -> list[CashFlow]:
    schedule = _main_schedule()
    dates = [
        TARGET().advance(schedule.date(i), 10, TimeUnit.Days, _FOLLOWING)
        for i in range(1, len(schedule))
    ]
    return (
        OvernightLeg(schedule, _eonia())
        .with_notionals(_NOMINAL)
        .with_payment_dates(dates)
        .build()
    )


def _on_capped() -> list[CashFlow]:
    return (
        OvernightLeg(_main_schedule(), _eonia())
        .with_notionals(_NOMINAL)
        .with_caps(0.032)
        .with_floors(0.01)
        .with_naked_option(True)
        .build()
    )


def _on_explicit_pricer() -> list[CashFlow]:
    return (
        OvernightLeg(_main_schedule(), _eonia())
        .with_notionals(_NOMINAL)
        .with_averaging_method(RateAveraging.Simple)
        .with_coupon_pricer(ArithmeticAveragedOvernightIndexedCouponPricer())
        .build()
    )


def _on_zero_gearing() -> list[CashFlow]:
    return (
        OvernightLeg(_main_schedule(), _eonia())
        .with_notionals(_NOMINAL)
        .with_payment_day_counter(Actual360())
        .with_gearings(0.0)
        .with_spreads(0.022)
        .with_caps(0.02)
        .build()
    )


def _on_adj(convention: BusinessDayConvention) -> list[CashFlow]:
    return (
        OvernightLeg(_unadjusted_schedule(), _eonia())
        .with_notionals(_NOMINAL)
        .with_payment_adjustment(convention)
        .build()
    )


def _on_stub() -> list[CashFlow]:
    return OvernightLeg(_stub_schedule(), _eonia()).with_notionals(_NOMINAL).build()


# ===========================================================================
# CmsLeg — with_notionals, with_payment_day_counter, with_payment_adjustment,
# with_fixing_days, with_gearings, with_spreads, with_caps, with_floors,
# in_arrears, with_zero_payments, with_fixing_convention, with_ex_coupon_period
# ===========================================================================


def _cms_default() -> list[CashFlow]:
    return CmsLeg(_main_schedule(), _cms10y()).with_notionals(_NOMINAL).build()


def _cms_full() -> list[CashFlow]:
    return (
        CmsLeg(_main_schedule(), _cms10y())
        .with_notionals([_NOMINAL, 2.0 * _NOMINAL])
        .with_payment_day_counter(Thirty360(Thirty360Convention.BondBasis))
        .with_payment_adjustment(_MF)
        .with_fixing_days(1)
        .with_gearings(1.5)
        .with_spreads(0.002)
        .with_fixing_convention(_FOLLOWING)
        .with_ex_coupon_period(Period(4, TimeUnit.Days), TARGET(), _PRECEDING, False)
        .build()
    )


def _cms_in_arrears() -> list[CashFlow]:
    return (
        CmsLeg(_main_schedule(), _cms10y()).with_notionals(_NOMINAL).in_arrears(True).build()
    )


def _cms_zero_payments() -> list[CashFlow]:
    return (
        CmsLeg(_main_schedule(), _cms10y())
        .with_notionals(_NOMINAL)
        .with_zero_payments(True)
        .build()
    )


def _cms_capped() -> list[CashFlow]:
    return (
        CmsLeg(_main_schedule(), _cms10y())
        .with_notionals(_NOMINAL)
        .with_caps(0.05)
        .with_floors(0.01)
        .build()
    )


def _cms_zero_gearing() -> list[CashFlow]:
    return (
        CmsLeg(_main_schedule(), _cms10y())
        .with_notionals(_NOMINAL)
        .with_payment_day_counter(Actual360())
        .with_gearings(0.0)
        .with_spreads(0.028)
        .with_caps(0.02)
        .build()
    )


def _cms_adj(convention: BusinessDayConvention) -> list[CashFlow]:
    return (
        CmsLeg(_unadjusted_schedule(), _cms10y())
        .with_notionals(_NOMINAL)
        .with_payment_adjustment(convention)
        .build()
    )


# ===========================================================================
# CPILeg — with_notionals, with_fixed_rates, with_payment_day_counter,
# with_payment_adjustment, with_payment_calendar,
# with_observation_interpolation, with_subtract_inflation_nominal, with_caps,
# with_floors, with_ex_coupon_period, with_base_date
# ===========================================================================


def _cpi_default(idx: EUHICP) -> list[CashFlow]:
    return (
        CPILeg(_cpi_schedule(), idx, _CPI_BASE, _CPI_LAG)
        .with_notionals(_NOMINAL)
        .with_fixed_rates(0.02)
        .build()
    )


def _cpi_full(idx: EUHICP) -> list[CashFlow]:
    return (
        CPILeg(_cpi_schedule(), idx, _CPI_BASE, _CPI_LAG)
        .with_notionals([_NOMINAL, 2.0 * _NOMINAL])
        .with_fixed_rates([0.02, 0.025])
        .with_payment_day_counter(Actual360())
        .with_payment_adjustment(_FOLLOWING)
        .with_payment_calendar(TARGET())
        .with_observation_interpolation(InterpolationType.Linear)
        .with_subtract_inflation_nominal(False)
        .with_ex_coupon_period(Period(5, TimeUnit.Days), TARGET(), _PRECEDING, False)
        .with_base_date(_CPI_BASE_DATE)
        .build()
    )


def _cpi_zero_rate_floored(idx: EUHICP) -> list[CashFlow]:
    return (
        CPILeg(_cpi_schedule(), idx, _CPI_BASE, _CPI_LAG)
        .with_notionals(_NOMINAL)
        .with_fixed_rates(0.0)
        .with_payment_day_counter(Actual360())
        .with_floors(0.015)
        .build()
    )


def _cpi_zero_rate_capped(idx: EUHICP) -> list[CashFlow]:
    return (
        CPILeg(_cpi_schedule(), idx, _CPI_BASE, _CPI_LAG)
        .with_notionals(_NOMINAL)
        .with_fixed_rates(0.0)
        .with_payment_day_counter(Actual360())
        .with_floors(0.015)
        .with_caps(0.005)
        .build()
    )


def _cpi_base_date_only(idx: EUHICP) -> list[CashFlow]:
    return (
        CPILeg(_cpi_schedule(), idx, None, _CPI_LAG)
        .with_notionals(_NOMINAL)
        .with_fixed_rates(0.02)
        .with_base_date(_CPI_BASE_DATE)
        .build()
    )


def _cpi_adj(idx: EUHICP, convention: BusinessDayConvention) -> list[CashFlow]:
    return (
        CPILeg(_unadjusted_past_schedule(), idx, _CPI_BASE, _CPI_LAG)
        .with_notionals(_NOMINAL)
        .with_fixed_rates(0.02)
        .with_payment_adjustment(convention)
        .build()
    )


# ===========================================================================
# DigitalIborLeg / DigitalCmsLeg
# ===========================================================================


def _priced(leg: list[CashFlow]) -> list[CashFlow]:
    set_coupon_pricer(leg, _black_pricer())
    return leg


def _digital_ibor_default() -> list[CashFlow]:
    return _priced(
        DigitalIborLeg(_main_schedule(), _euribor6m())
        .with_notionals(_NOMINAL)
        .with_call_strikes(0.03)
        .build()
    )


def _digital_ibor_full() -> list[CashFlow]:
    return _priced(
        DigitalIborLeg(_main_schedule(), _euribor6m())
        .with_notionals([_NOMINAL, 2.0 * _NOMINAL])
        .with_payment_day_counter(Actual360())
        .with_payment_adjustment(_MF)
        .with_fixing_days(1)
        .with_gearings(1.5)
        .with_spreads(0.002)
        .with_call_strikes(0.03)
        .with_long_call_option(PositionType.Short)
        .with_call_atm(True)
        .with_call_payoffs(0.04)
        .with_put_strikes(0.02)
        .with_long_put_option(PositionType.Short)
        .with_put_atm(True)
        .with_put_payoffs(0.015)
        .with_replication(DigitalReplication(Replication.Central, 1e-3))
        .with_naked_option(True)
        .build()
    )


def _digital_ibor_in_arrears() -> list[CashFlow]:
    return _priced(
        DigitalIborLeg(_main_schedule(), _euribor6m())
        .with_notionals(_NOMINAL)
        .with_call_strikes(0.03)
        .in_arrears(True)
        .build()
    )


def _digital_ibor_zero_gearing() -> list[CashFlow]:
    return _priced(
        DigitalIborLeg(_main_schedule(), _euribor6m())
        .with_notionals(_NOMINAL)
        .with_payment_day_counter(Actual360())
        .with_gearings(0.0)
        .with_spreads(0.027)
        .with_call_strikes(0.03)
        .build()
    )


def _digital_cms_default() -> list[CashFlow]:
    return (
        DigitalCmsLeg(_main_schedule(), _cms10y())
        .with_notionals(_NOMINAL)
        .with_call_strikes(0.03)
        .build()
    )


def _digital_cms_full() -> list[CashFlow]:
    return (
        DigitalCmsLeg(_main_schedule(), _cms10y())
        .with_notionals([_NOMINAL, 2.0 * _NOMINAL])
        .with_payment_day_counter(Actual360())
        .with_payment_adjustment(_MF)
        .with_fixing_days(1)
        .with_gearings(1.5)
        .with_spreads(0.002)
        .with_call_strikes(0.03)
        .with_long_call_option(PositionType.Short)
        .with_call_atm(True)
        .with_call_payoffs(0.04)
        .with_put_strikes(0.02)
        .with_long_put_option(PositionType.Short)
        .with_put_atm(True)
        .with_put_payoffs(0.015)
        .with_replication(DigitalReplication(Replication.Central, 1e-3))
        .with_naked_option(True)
        .build()
    )


def _digital_cms_in_arrears() -> list[CashFlow]:
    return (
        DigitalCmsLeg(_main_schedule(), _cms10y())
        .with_notionals(_NOMINAL)
        .with_call_strikes(0.03)
        .in_arrears(True)
        .build()
    )


def _digital_cms_zero_gearing() -> list[CashFlow]:
    return (
        DigitalCmsLeg(_main_schedule(), _cms10y())
        .with_notionals(_NOMINAL)
        .with_payment_day_counter(Actual360())
        .with_gearings(0.0)
        .with_spreads(0.027)
        .with_call_strikes(0.03)
        .build()
    )


# ===========================================================================
# AverageBMALeg — with_notionals, with_payment_day_counter,
# with_payment_adjustment, with_gearings, with_spreads
# ===========================================================================


def _bma_default() -> list[CashFlow]:
    return AverageBMALeg(_bma_schedule(), _bma()).with_notionals(_NOMINAL).build()


def _bma_full() -> list[CashFlow]:
    return (
        AverageBMALeg(_bma_schedule(), _bma())
        .with_notionals([_NOMINAL, 2.0 * _NOMINAL])
        .with_payment_day_counter(Actual360())
        .with_payment_adjustment(_PRECEDING)
        .with_gearings(1.5)
        .with_spreads(0.002)
        .build()
    )


def _bma_adj(convention: BusinessDayConvention) -> list[CashFlow]:
    return (
        AverageBMALeg(_unadjusted_schedule(), _bma())
        .with_notionals(_NOMINAL)
        .with_payment_adjustment(convention)
        .build()
    )


def _bma_stub() -> list[CashFlow]:
    return AverageBMALeg(_bma_stub_schedule(), _bma()).with_notionals(_NOMINAL).build()


# ===========================================================================
# Cross-validation
# ===========================================================================

_BUILDERS: dict[str, Callable[[], list[CashFlow]]] = {
    "fixed_default": _fixed_default,
    "fixed_full": _fixed_full,
    "fixed_interest_rate": _fixed_interest_rate,
    "fixed_interest_rate_vector": _fixed_interest_rate_vector,
    "fixed_stub": _fixed_stub,
    "fixed_adj_following": lambda: _fixed_adj(_FOLLOWING),
    "fixed_adj_unadjusted": lambda: _fixed_adj(_UNADJUSTED),
    "ibor_default": _ibor_default,
    "ibor_full": _ibor_full,
    "ibor_zero_payments": _ibor_zero_payments,
    "ibor_capped": _ibor_capped,
    "ibor_in_arrears": _ibor_in_arrears,
    "ibor_zero_gearing": _ibor_zero_gearing,
    "ibor_quarterly_default": _ibor_quarterly_default,
    "ibor_indexed_coupons": _ibor_indexed_coupons,
    "ibor_at_par_coupons": _ibor_at_par_coupons,
    "ibor_adj_following": lambda: _ibor_adj(_FOLLOWING),
    "ibor_adj_unadjusted": lambda: _ibor_adj(_UNADJUSTED),
    "ibor_stub": _ibor_stub,
    "on_default": _on_default,
    "on_full": _on_full,
    "on_simple_averaging": _on_simple_averaging,
    "on_in_advance": _on_in_advance,
    "on_last_recent": _on_last_recent,
    "on_payment_dates": _on_payment_dates,
    "on_capped": _on_capped,
    "on_explicit_pricer": _on_explicit_pricer,
    "on_zero_gearing": _on_zero_gearing,
    "on_adj_following": lambda: _on_adj(_FOLLOWING),
    "on_adj_unadjusted": lambda: _on_adj(_UNADJUSTED),
    "on_stub": _on_stub,
    "cms_default": _cms_default,
    "cms_full": _cms_full,
    "cms_in_arrears": _cms_in_arrears,
    "cms_zero_payments": _cms_zero_payments,
    "cms_capped": _cms_capped,
    "cms_zero_gearing": _cms_zero_gearing,
    "cms_adj_following": lambda: _cms_adj(_FOLLOWING),
    "cms_adj_unadjusted": lambda: _cms_adj(_UNADJUSTED),
    "digital_ibor_default": _digital_ibor_default,
    "digital_ibor_full": _digital_ibor_full,
    "digital_ibor_in_arrears": _digital_ibor_in_arrears,
    "digital_ibor_zero_gearing": _digital_ibor_zero_gearing,
    "digital_cms_default": _digital_cms_default,
    "digital_cms_full": _digital_cms_full,
    "digital_cms_in_arrears": _digital_cms_in_arrears,
    "digital_cms_zero_gearing": _digital_cms_zero_gearing,
    "bma_default": _bma_default,
    "bma_full": _bma_full,
    "bma_adj_following": lambda: _bma_adj(_FOLLOWING),
    "bma_adj_unadjusted": lambda: _bma_adj(_UNADJUSTED),
    "bma_stub": _bma_stub,
}

_CPI_BUILDERS: dict[str, Callable[[EUHICP], list[CashFlow]]] = {
    "cpi_default": _cpi_default,
    "cpi_full": _cpi_full,
    "cpi_zero_rate_floored": _cpi_zero_rate_floored,
    "cpi_zero_rate_capped": _cpi_zero_rate_capped,
    "cpi_base_date_only": _cpi_base_date_only,
    "cpi_adj_following": lambda idx: _cpi_adj(idx, _FOLLOWING),
    "cpi_adj_unadjusted": lambda idx: _cpi_adj(idx, _UNADJUSTED),
}


# Legs whose *rate* depends on a carve-out this port has not landed. The
# structure is still cross-validated; the rate is covered by an explicitly
# skipped test that names the missing piece.
_STRUCTURE_ONLY = {"digital_ibor_in_arrears"}

# Reference keys with no Python counterpart at all, each with its reason.
_UNREACHABLE: dict[str, str] = {
    "on_explicit_pricer_convexity": (
        "ArithmeticAveragedOvernightIndexedCouponPricer's Hull-White convexity "
        "correction (mean reversion / volatility / Takada approximation) is a "
        "documented carve-out in overnight_indexed_coupon_pricer.py, so a "
        "non-zero volatility changes nothing in Python"
    ),
}


@pytest.mark.parametrize("key", sorted(_BUILDERS))
def test_leg_matches_cpp(key: str, cpp: dict[str, Any]) -> None:
    _assert_leg(_BUILDERS[key](), cpp[key], compare_rates=key not in _STRUCTURE_ONLY)


@pytest.mark.parametrize("key", sorted(_CPI_BUILDERS))
def test_cpi_leg_matches_cpp(key: str, cpp: dict[str, Any], seeded_eu_hicp: EUHICP) -> None:
    _assert_leg(_CPI_BUILDERS[key](seeded_eu_hicp), cpp[key])


def test_reference_covers_every_builder(cpp: dict[str, Any]) -> None:
    """No reference key may go unasserted, and no builder unreferenced."""
    assert set(cpp) == set(_BUILDERS) | set(_CPI_BUILDERS) | set(_UNREACHABLE)


@pytest.mark.skip(
    reason="BlackIborCouponPricer's in-arrears timing adjustment (C++ "
    "timingAdjustment_ / adjustedFixing, couponpricer.cpp) is a documented "
    "carve-out; the C++ rate is pinned in v143/cf/legs "
    "['digital_ibor_in_arrears'] so the gap is measurable once it lands. "
    "Everything else about this leg IS cross-validated."
)
def test_digital_ibor_in_arrears_rate_needs_convexity_adjustment(
    cpp: dict[str, Any],
) -> None:
    _assert_leg(_digital_ibor_in_arrears(), cpp["digital_ibor_in_arrears"])


@pytest.mark.skip(reason=_UNREACHABLE["on_explicit_pricer_convexity"])
def test_on_explicit_pricer_with_convexity_adjustment(cpp: dict[str, Any]) -> None:
    leg = (
        OvernightLeg(_main_schedule(), _eonia())
        .with_notionals(_NOMINAL)
        .with_averaging_method(RateAveraging.Simple)
        .with_coupon_pricer(
            ArithmeticAveragedOvernightIndexedCouponPricer(0.05, 0.20, True)
        )
        .build()
    )
    _assert_leg(leg, cpp["on_explicit_pricer_convexity"])


# ===========================================================================
# Setter-bites-independently checks
#
# The cross-validation above would still pass if two setters happened to
# produce the same listing, so the setters whose effect could coincide with
# the default are additionally asserted to *differ* from it.
# ===========================================================================


def _pay_dates(leg: list[CashFlow]) -> list[int]:
    return [cf.date().serial_number() for cf in leg]


def _rates(leg: list[CashFlow]) -> list[float | None]:
    return [_safe_rate(cf) for cf in leg if isinstance(cf, Coupon)]


def test_payment_lag_moves_payment_dates() -> None:
    """FixedRateLeg / IborLeg / OvernightLeg with_payment_lag."""
    assert _pay_dates(_fixed_full()) != _pay_dates(_fixed_default())
    assert _pay_dates(_ibor_full()) != _pay_dates(_ibor_default())
    assert _pay_dates(_on_full()) != _pay_dates(_on_default())


def test_payment_adjustment_moves_payment_dates() -> None:
    """with_payment_adjustment on all six builders that have it.

    Only visible on an unadjusted schedule: on a schedule whose dates are
    already business days the payment roll is a no-op, and once a payment lag
    is non-zero ``Calendar.advance`` over Days ignores the convention entirely.
    """
    assert _pay_dates(_fixed_adj(_FOLLOWING)) != _pay_dates(_fixed_adj(_UNADJUSTED))
    assert _pay_dates(_ibor_adj(_FOLLOWING)) != _pay_dates(_ibor_adj(_UNADJUSTED))
    assert _pay_dates(_on_adj(_FOLLOWING)) != _pay_dates(_on_adj(_UNADJUSTED))
    assert _pay_dates(_cms_adj(_FOLLOWING)) != _pay_dates(_cms_adj(_UNADJUSTED))
    assert _pay_dates(_bma_adj(_FOLLOWING)) != _pay_dates(_bma_adj(_UNADJUSTED))


def test_cpi_payment_adjustment_moves_payment_dates(seeded_eu_hicp: EUHICP) -> None:
    """CPILeg.with_payment_adjustment."""
    assert _pay_dates(_cpi_adj(seeded_eu_hicp, _FOLLOWING)) != _pay_dates(
        _cpi_adj(seeded_eu_hicp, _UNADJUSTED)
    )


def test_zero_payments_collapses_to_one_date() -> None:
    """IborLeg / CmsLeg with_zero_payments."""
    for leg in (_ibor_zero_payments(), _cms_zero_payments()):
        dates = _pay_dates(leg)
        assert len(set(dates)) == 1
        assert len(dates) > 1


def test_explicit_payment_dates_override_the_roll() -> None:
    """OvernightLeg.with_payment_dates."""
    assert _pay_dates(_on_payment_dates()) != _pay_dates(_on_default())


def test_wrong_number_of_payment_dates_raises() -> None:
    """OvernightLeg.with_payment_dates length check (C++ QL_REQUIRE)."""
    with pytest.raises(LibraryException, match="explicit payment dates"):
        (
            OvernightLeg(_main_schedule(), _eonia())
            .with_notionals(_NOMINAL)
            .with_payment_dates([_TODAY])
            .build()
        )


def test_indexed_coupons_flag_changes_the_forecast() -> None:
    """IborLeg.with_indexed_coupons / with_at_par_coupons.

    Only observable when the accrual period differs from the index tenor —
    hence a quarterly schedule against the 6M index.
    """
    indexed = _rates(_ibor_indexed_coupons())
    at_par = _rates(_ibor_at_par_coupons())
    assert indexed != at_par
    assert at_par == _rates(_ibor_quarterly_default())


def test_averaging_method_changes_the_rate() -> None:
    """OvernightLeg.with_averaging_method."""
    assert _rates(_on_simple_averaging()) != _rates(_on_default())
    assert all(
        c.averaging_method() == RateAveraging.Simple
        for c in _on_simple_averaging()
        if isinstance(c, OvernightIndexedCoupon)
    )


def test_explicit_coupon_pricer_is_the_one_attached() -> None:
    """OvernightLeg.with_coupon_pricer.

    The observable consequence is instance identity: every coupon must hold
    the supplied pricer rather than the one its own constructor attached. A
    *numeric* consequence is not available — the only arithmetic-pricer
    parameters that move the rate drive the Hull-White convexity correction,
    which is a documented carve-out (see the skipped
    ``test_on_explicit_pricer_with_convexity_adjustment``).
    """
    pricer = ArithmeticAveragedOvernightIndexedCouponPricer()
    leg = (
        OvernightLeg(_main_schedule(), _eonia())
        .with_notionals(_NOMINAL)
        .with_averaging_method(RateAveraging.Simple)
        .with_coupon_pricer(pricer)
        .build()
    )
    assert all(c.pricer() is pricer for c in leg if isinstance(c, OvernightIndexedCoupon))
    # ... and it is *not* the one the coupon would have built for itself.
    assert all(
        c.pricer() is not pricer
        for c in _on_simple_averaging()
        if isinstance(c, OvernightIndexedCoupon)
    )


def test_mismatched_coupon_pricer_raises() -> None:
    """OvernightLeg.with_coupon_pricer type check (C++ QL_REQUIRE)."""
    with pytest.raises(LibraryException, match="Wrong coupon pricer"):
        (
            OvernightLeg(_main_schedule(), _eonia())
            .with_notionals(_NOMINAL)
            .with_coupon_pricer(
                ArithmeticAveragedOvernightIndexedCouponPricer()
            )  # averaging is Compound
            .build()
        )


def test_in_advance_moves_the_observation_window() -> None:
    """OvernightLeg.in_arrears(False)."""
    advance = [
        (c.rate_computation_start_date(), c.rate_computation_end_date())
        for c in _on_in_advance()
        if isinstance(c, OvernightIndexedCoupon)
    ]
    arrears = [
        (c.rate_computation_start_date(), c.rate_computation_end_date())
        for c in _on_default()
        if isinstance(c, OvernightIndexedCoupon)
    ]
    assert advance != arrears
    # each in-advance window is the *previous* period
    assert advance[1:] == arrears[:-1]


def test_last_recent_period_shortens_the_observation_window() -> None:
    """OvernightLeg.with_last_recent_period / with_last_recent_period_calendar."""
    for c in _on_last_recent():
        if isinstance(c, OvernightIndexedCoupon):
            assert c.rate_computation_start_date() > c.accrual_start_date()
            assert c.rate_computation_end_date() == c.accrual_end_date()
    assert _rates(_on_last_recent()) != _rates(_on_default())


def test_naked_option_flag_reaches_the_coupon() -> None:
    """OvernightLeg.with_naked_option."""
    assert all(
        c.naked_option()
        for c in _on_capped()
        if isinstance(c, CappedFlooredOvernightIndexedCoupon)
    )


def _day_counter_names(leg: list[CashFlow]) -> list[str]:
    names: list[str] = []
    for cf in leg:
        assert isinstance(cf, Coupon)
        names.append(cf.day_counter().name())
    return names


def test_first_and_last_period_day_counters_bite() -> None:
    """FixedRateLeg.with_first_period_day_counter / with_last_period_day_counter."""
    plain = _day_counter_names(_fixed_default())
    stubbed = _day_counter_names(_fixed_stub())
    # 30/360 on the first coupon, Act/365F on the last, Act/360 in between.
    assert stubbed[0] != plain[0]
    assert stubbed[-1] != plain[-1]
    assert stubbed[1] == plain[1]


def test_ex_coupon_period_sets_ex_coupon_dates() -> None:
    """FixedRateLeg / IborLeg / CmsLeg with_ex_coupon_period."""
    for full, default in (
        (_fixed_full(), _fixed_default()),
        (_ibor_full(), _ibor_default()),
        (_cms_full(), _cms_default()),
    ):
        assert all(c.ex_coupon_date() != Date() for c in full if isinstance(c, Coupon))
        assert all(c.ex_coupon_date() == Date() for c in default if isinstance(c, Coupon))


def test_subtract_inflation_nominal_flips_growth_only(seeded_eu_hicp: EUHICP) -> None:
    """CPILeg.with_subtract_inflation_nominal."""
    default_flow = _cpi_default(seeded_eu_hicp)[-1]
    full_flow = _cpi_full(seeded_eu_hicp)[-1]
    assert isinstance(default_flow, CPICashFlow)
    assert isinstance(full_flow, CPICashFlow)
    assert default_flow.growth_only() is True
    assert full_flow.growth_only() is False


def test_cpi_caps_and_floors_clamp_the_zero_rate(seeded_eu_hicp: EUHICP) -> None:
    """CPILeg.with_caps / with_floors — only reachable on a zero fixed rate."""
    floored = _cpi_zero_rate_floored(seeded_eu_hicp)
    capped = _cpi_zero_rate_capped(seeded_eu_hicp)
    assert all(c.rate() == 0.015 for c in floored if isinstance(c, Coupon))
    assert all(c.rate() == 0.005 for c in capped if isinstance(c, Coupon))


def test_cpi_caps_on_a_non_zero_rate_raise(seeded_eu_hicp: EUHICP) -> None:
    """C++ QL_FAIL("caps/floors on CPI coupons not implemented.")."""
    with pytest.raises(LibraryException, match="caps/floors on CPI coupons"):
        (
            CPILeg(_cpi_schedule(), seeded_eu_hicp, _CPI_BASE, _CPI_LAG)
            .with_notionals(_NOMINAL)
            .with_fixed_rates(0.02)
            .with_caps(0.03)
            .build()
        )


def test_digital_positions_and_payoffs_reach_the_coupons() -> None:
    """DigitalIborLeg / DigitalCmsLeg with_long_call_option / with_long_put_option /
    with_call_payoffs / with_put_payoffs / with_call_strikes / with_put_strikes."""
    for leg in (_digital_ibor_full(), _digital_cms_full()):
        for c in leg:
            assert isinstance(c, DigitalCoupon)
            assert c.is_long_call() is False
            assert c.is_long_put() is False
            assert c.call_strike() == 0.03
            assert c.put_strike() == 0.02
            assert c.call_digital_payoff() == 0.04
            assert c.put_digital_payoff() == 0.015


def _digital_ibor_variant(**kwargs: Any) -> list[CashFlow]:
    builder = (
        DigitalIborLeg(_main_schedule(), _euribor6m())
        .with_notionals(_NOMINAL)
        .with_call_strikes(kwargs.get("call_strike", 0.03))
    )
    if "replication" in kwargs:
        builder = builder.with_replication(kwargs["replication"])
    if kwargs.get("call_atm"):
        builder = builder.with_call_atm(True)
    if kwargs.get("naked"):
        builder = builder.with_naked_option(True)
    return _priced(builder.build())


def test_digital_replication_and_naked_change_the_rate() -> None:
    """DigitalIborLeg.with_replication / with_naked_option.

    Neither has a direct accessor, so each is asserted through the rate the
    call-spread replication produces.
    """
    plain = _rates(_digital_ibor_variant())
    assert (
        _rates(_digital_ibor_variant(replication=DigitalReplication(Replication.Central, 1e-2)))
        != plain
    )
    assert _rates(_digital_ibor_variant(naked=True)) != plain


def test_digital_atm_inclusion_changes_the_at_the_money_payoff() -> None:
    """DigitalIborLeg.with_call_atm / with_put_atm.

    # C++ parity: ``isCallATMIncluded_`` is consulted only by
    # ``DigitalCoupon::callPayoff`` / ``putPayoff`` (digitalcoupon.cpp:297-330),
    # in the ``|strike - rate| <= 1e-16`` branch. It therefore has *no* effect
    # on the replicated rate away from the boundary — the observable
    # consequence is the payoff at a strike set exactly to the underlying's
    # own rate.
    """
    reference = _digital_ibor_variant()[0]
    assert isinstance(reference, DigitalCoupon)
    at_the_money_strike = reference.underlying().rate()

    without = _digital_ibor_variant(call_strike=at_the_money_strike)[0]
    with_atm = _digital_ibor_variant(call_strike=at_the_money_strike, call_atm=True)[0]
    assert isinstance(without, DigitalCoupon)
    assert isinstance(with_atm, DigitalCoupon)
    # Not cash-or-nothing here, so the ATM payoff is the underlying rate itself.
    assert without._call_payoff() == 0.0  # pyright: ignore[reportPrivateUsage]
    tolerance.tight(
        with_atm._call_payoff(),  # pyright: ignore[reportPrivateUsage]
        at_the_money_strike,
    )


def test_bma_gearing_and_spread_reach_the_coupons() -> None:
    """AverageBMALeg.with_gearings / with_spreads / with_notionals /
    with_payment_day_counter."""
    full = _bma_full()
    default = _bma_default()
    assert all(
        c.gearing() == 1.5 and c.spread() == 0.002
        for c in full
        if isinstance(c, FloatingRateCoupon)
    )
    first, second = full[0], full[1]
    assert isinstance(first, Coupon)
    assert isinstance(second, Coupon)
    assert first.nominal() == _NOMINAL
    assert second.nominal() == 2.0 * _NOMINAL
    assert _day_counter_names(full)[0] != _day_counter_names(default)[0]


def test_too_many_nominals_raises() -> None:
    """FloatingLeg's vector-size guards (C++ QL_REQUIRE)."""
    with pytest.raises(LibraryException, match="too many nominals"):
        IborLeg(_main_schedule(), _euribor6m()).with_notionals(
            [1.0, 2.0, 3.0, 4.0, 5.0]
        ).build()


def test_zero_and_in_arrears_are_incompatible() -> None:
    """C++ QL_REQUIRE(!isZero || !isInArrears)."""
    with pytest.raises(LibraryException, match="not compatible"):
        (
            IborLeg(_main_schedule(), _euribor6m())
            .with_notionals(_NOMINAL)
            .in_arrears(True)
            .with_zero_payments(True)
            .build()
        )
