"""ActualActual — Act/Act day counter with 7 convention aliases.

# C++ parity: ql/time/daycounters/actualactual.hpp + actualactual.cpp (v1.42.1).

C++ defines 4 distinct Impl classes mapped from 7 ``Convention`` values:

- ISMA   } share ISMA_Impl when a Schedule is provided, otherwise
- Bond   } Old_ISMA_Impl (a recursive fallback).
- ISDA   ┐
- Historical ── share ISDA_Impl.
- Actual365  ┘
- AFB    } share AFB_Impl.
- Euro   }

The Python port collapses these to a single ``ActualActual`` class
dispatching on the ``Convention`` IntEnum and an optional ``schedule``.
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

_MONTHS_PER_YEAR: int = 12
_YEAR_DIVISOR_365: float = 365.0
_DAYS_LEAP: float = 366.0


class Convention(IntEnum):
    ISMA = 0
    Bond = 1
    ISDA = 2
    Historical = 3
    Actual365 = 4
    AFB = 5
    Euro = 6


class ActualActual(DayCounter):
    def __init__(self, convention: Convention, schedule: Schedule | None = None) -> None:
        super().__init__()
        self._convention: Convention = convention
        self._schedule: Schedule | None = schedule

    def name(self) -> str:
        if self._convention in (Convention.ISMA, Convention.Bond):
            return "Actual/Actual (ISMA)"
        if self._convention in (Convention.ISDA, Convention.Historical, Convention.Actual365):
            return "Actual/Actual (ISDA)"
        return "Actual/Actual (AFB)"

    def year_fraction(
        self,
        d1: Date,
        d2: Date,
        ref_period_start: Date | None = None,
        ref_period_end: Date | None = None,
    ) -> float:
        c = self._convention
        if c in (Convention.ISMA, Convention.Bond):
            if self._schedule is not None and len(self._schedule) > 0:
                return _yf_isma_with_schedule(d1, d2, self._schedule)
            return _yf_old_isma(d1, d2, ref_period_start, ref_period_end)
        if c in (Convention.ISDA, Convention.Historical, Convention.Actual365):
            return _yf_isda(d1, d2)
        # AFB / Euro
        return _yf_afb(d1, d2)


# --- ISDA / Historical / Actual365 -----------------------------------------


def _yf_isda(d1: Date, d2: Date) -> float:
    """Mirrors C++ ``ActualActual::ISDA_Impl::yearFraction``."""
    if d1 == d2:
        return 0.0
    if d1 > d2:
        return -_yf_isda(d2, d1)
    y1, y2 = d1.year(), d2.year()
    dib1 = _DAYS_LEAP if Date.is_leap(y1) else _YEAR_DIVISOR_365
    dib2 = _DAYS_LEAP if Date.is_leap(y2) else _YEAR_DIVISOR_365
    sum_ = float(y2 - y1 - 1)
    sum_ += (Date.from_ymd(1, Month.January, y1 + 1) - d1) / dib1
    sum_ += (d2 - Date.from_ymd(1, Month.January, y2)) / dib2
    return sum_


# --- AFB / Euro ------------------------------------------------------------


def _yf_afb(d1: Date, d2: Date) -> float:
    """Mirrors C++ ``ActualActual::AFB_Impl::yearFraction``."""
    if d1 == d2:
        return 0.0
    if d1 > d2:
        return -_yf_afb(d2, d1)

    new_d2 = d2
    temp = d2
    sum_ = 0.0
    while temp > d1:
        temp = new_d2 - Period(1, TimeUnit.Years)
        if temp.day_of_month() == 28 and temp.month() == Month.February and Date.is_leap(temp.year()):
            temp = temp + 1
        if temp >= d1:
            sum_ += 1.0
            new_d2 = temp

    den = _YEAR_DIVISOR_365
    if Date.is_leap(new_d2.year()):
        leap_day = Date.from_ymd(29, Month.February, new_d2.year())
        if new_d2 > leap_day and d1 <= leap_day:
            den += 1.0
    elif Date.is_leap(d1.year()):
        leap_day = Date.from_ymd(29, Month.February, d1.year())
        if new_d2 > leap_day and d1 <= leap_day:
            den += 1.0
    return sum_ + (new_d2 - d1) / den


# --- ISMA / Bond — Old_ISMA (no schedule) ----------------------------------


def _yf_old_isma(
    d1: Date,
    d2: Date,
    ref_period_start: Date | None,
    ref_period_end: Date | None,
) -> float:
    """Mirrors C++ ``ActualActual::Old_ISMA_Impl::yearFraction`` (recursive)."""
    if d1 == d2:
        return 0.0
    if d1 > d2:
        return -_yf_old_isma(d2, d1, ref_period_start, ref_period_end)

    ref_start = ref_period_start if ref_period_start is not None and ref_period_start != Date() else d1
    ref_end = ref_period_end if ref_period_end is not None and ref_period_end != Date() else d2

    qassert.require(
        ref_end > ref_start and ref_end > d1,
        f"invalid reference period: date 1: {d1}, date 2: {d2}, "
        f"reference period start: {ref_start}, reference period end: {ref_end}",
    )

    months = round(12 * (ref_end - ref_start) / 365)
    if months == 0:
        # Short period — take ref as 1 year from d1.
        ref_start = d1
        ref_end = d1 + Period(1, TimeUnit.Years)
        months = _MONTHS_PER_YEAR

    period = months / 12.0

    if d2 <= ref_end:
        if d1 >= ref_start:
            return period * (d2 - d1) / (ref_end - ref_start)
        # Long first coupon: d1 < refStart < refEnd, d2 <= refEnd.
        previous_ref = ref_start - Period(months, TimeUnit.Months)
        if d2 > ref_start:
            return _yf_old_isma(d1, ref_start, previous_ref, ref_start) + _yf_old_isma(
                ref_start, d2, ref_start, ref_end
            )
        return _yf_old_isma(d1, d2, previous_ref, ref_start)

    # refEnd < d2 — split into [d1, refEnd] + whole periods + tail.
    qassert.require(
        ref_start <= d1,
        f"invalid dates: d1 < refPeriodStart < refPeriodEnd < d2 "
        f"(d1={d1}, refStart={ref_start}, refEnd={ref_end}, d2={d2})",
    )
    sum_ = _yf_old_isma(d1, ref_end, ref_start, ref_end)
    i = 0
    new_ref_start = ref_end
    new_ref_end = ref_end + Period(months, TimeUnit.Months)
    while d2 >= new_ref_end:
        sum_ += period
        i += 1
        new_ref_start = ref_end + Period(months * i, TimeUnit.Months)
        new_ref_end = ref_end + Period(months * (i + 1), TimeUnit.Months)
    sum_ += _yf_old_isma(new_ref_start, d2, new_ref_start, new_ref_end)
    return sum_


# --- ISMA / Bond — with-Schedule (full algorithm) -------------------------


def _yf_isma_with_schedule(d1: Date, d2: Date, schedule: Schedule) -> float:
    """Mirrors C++ ``ActualActual::ISMA_Impl::yearFraction``."""
    if d1 == d2:
        return 0.0
    if d2 < d1:
        return -_yf_isma_with_schedule(d2, d1, schedule)

    coupon_dates = _coupon_dates_with_quasi_payments(schedule)
    first_date = min(coupon_dates)
    last_date = max(coupon_dates)
    qassert.require(
        d1 >= first_date and d2 <= last_date,
        f"Dates out of range of schedule: date 1: {d1}, date 2: {d2}, "
        f"first date: {first_date}, last date: {last_date}",
    )

    yf_sum = 0.0
    for i in range(len(coupon_dates) - 1):
        start = coupon_dates[i]
        end = coupon_dates[i + 1]
        if d1 < end and d2 > start:
            yf_sum += _yf_with_ref_dates(max(d1, start), min(d2, end), start, end)
    return yf_sum


def _coupon_dates_with_quasi_payments(schedule: Schedule) -> list[Date]:
    """Mirrors C++ ``getListOfPeriodDatesIncludingQuasiPayments``."""
    issue_date = schedule.date(0)
    dates = list(schedule.dates)

    if not schedule.has_is_regular() or not schedule.is_regular_at(1):
        first_coupon = schedule.date(1)
        notional_first_coupon = schedule.calendar.advance_period(
            first_coupon,
            -schedule.tenor,
            schedule.business_day_convention,
            schedule.end_of_month,
        )
        dates[0] = notional_first_coupon
        if notional_first_coupon > issue_date:
            prior_notional = schedule.calendar.advance_period(
                notional_first_coupon,
                -schedule.tenor,
                schedule.business_day_convention,
                schedule.end_of_month,
            )
            dates.insert(0, prior_notional)

    if not schedule.has_is_regular() or not schedule.is_regular_at(len(schedule) - 1):
        notional_last_coupon = schedule.calendar.advance_period(
            schedule.date(len(schedule) - 2),
            schedule.tenor,
            schedule.business_day_convention,
            schedule.end_of_month,
        )
        dates[len(schedule) - 1] = notional_last_coupon
        if notional_last_coupon < schedule.end_date:
            next_notional = schedule.calendar.advance_period(
                notional_last_coupon,
                schedule.tenor,
                schedule.business_day_convention,
                schedule.end_of_month,
            )
            dates.append(next_notional)

    return dates


def _coupons_per_year(ref_start: Date, ref_end: Date) -> int:
    """Mirrors C++ ``findCouponsPerYear``. Only correct for periods > 15 days.

    Guarded against division-by-zero: callers should already filter via the
    ``reference_day_count >= 16`` branch in ``_yf_with_ref_dates``, but the
    guard here is defence-in-depth in case the helper is invoked directly.
    """
    months = round(12 * (ref_end - ref_start) / 365.0)
    qassert.require(
        months != 0,
        f"_coupons_per_year: reference period too short ({ref_end - ref_start} days, "
        f"rounded to 0 months) — caller must guard via the 15-day-floor in C++.",
    )
    return round(12.0 / months)


def _yf_with_ref_dates(d1: Date, d2: Date, d3: Date, d4: Date) -> float:
    """Mirrors C++ ``yearFractionWithReferenceDates``."""
    qassert.require(d1 <= d2, f"This function is only correct if d1 <= d2 (d1={d1}, d2={d2})")
    reference_day_count = float(d4 - d3)
    if reference_day_count < 16:
        coupons_per_year = 1
        reference_day_count = float((d1 + Period(1, TimeUnit.Years)) - d1)
    else:
        coupons_per_year = _coupons_per_year(d3, d4)
    return (d2 - d1) / (reference_day_count * coupons_per_year)


__all__ = [
    "ActualActual",
    "Convention",
]
