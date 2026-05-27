"""AmortizingFixedRateBond — fixed-rate bond with variable notional.

# C++ parity: ql/instruments/bonds/amortizingfixedratebond.{hpp,cpp}
   (v1.42.1).

The notional schedule is supplied directly (one entry per coupon period).
Two free helpers from the C++ source — ``sinkingSchedule`` and
``sinkingNotionals`` (French amortisation) — are also ported.

Carve-outs:
- ``payment_lag`` parameter is accepted for parity but ignored (the L2-D
  ``fixed_rate_leg`` builder doesn't yet thread it).
- ``ex_coupon_period`` and friends are accepted but ignored (same).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pquantlib import qassert
from pquantlib.cashflows.fixed_rate_leg import fixed_rate_leg
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.instruments.bond import Bond
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit


class AmortizingFixedRateBond(Bond):
    """Fixed-rate bond with a per-period notional vector.

    # C++ parity: amortizingfixedratebond.cpp:27-60.
    """

    def __init__(
        self,
        settlement_days: int,
        notionals: Sequence[float],
        schedule: Schedule,
        coupons: Sequence[float],
        accrual_day_counter: DayCounter,
        payment_convention: BusinessDayConvention = BusinessDayConvention.Following,
        issue_date: Date | None = None,
        redemptions: Sequence[float] = (100.0,),
        payment_lag: int = 0,
    ) -> None:
        del payment_lag  # accepted for parity, ignored — see module docstring.
        Bond.__init__(self, settlement_days, schedule.calendar, issue_date)
        self._frequency: Frequency = schedule.tenor.frequency()
        self._day_counter: DayCounter = accrual_day_counter
        self._maturity_date = schedule.end_date

        self._cashflows = list(
            fixed_rate_leg(
                schedule,
                nominals=list(notionals),
                rates=list(coupons),
                day_counter=accrual_day_counter,
                compounding=Compounding.Simple,
                frequency=Frequency.Annual,
                payment_adjustment=payment_convention,
            )
        )

        self._add_redemptions_to_cashflows(list(redemptions))

        qassert.require(len(self._cashflows) > 0, "bond with no cashflows!")

        for cf in self._cashflows:
            cf.register_with(self)

    def frequency(self) -> Frequency:
        return self._frequency

    def day_counter(self) -> DayCounter:
        return self._day_counter


# --- free helpers -----------------------------------------------------------


def sinking_schedule(
    start_date: Date,
    bond_length: Period,
    frequency: Frequency,
    payment_calendar: Calendar,
) -> Schedule:
    """Build a schedule for French-style amortisation.

    # C++ parity: amortizingfixedratebond.cpp:63-72.
    """
    maturity = start_date + bond_length
    return Schedule.from_rule(
        start_date,
        maturity,
        _period_from_frequency(frequency),
        payment_calendar,
        BusinessDayConvention.Unadjusted,
        BusinessDayConvention.Unadjusted,
        DateGeneration.Backward,
        False,
    )


def _period_from_frequency(f: Frequency) -> Period:
    """Mirror C++ ``Period(Frequency)`` ctor used by sinkingSchedule."""
    # period.hpp Period(Frequency) -> 12/f months for monthly-divisible
    # frequencies, etc.
    val = int(f)
    if val == 0:
        # NoFrequency or NoFrequency-like
        return Period(0, TimeUnit.Days)
    if val == 1:
        return Period(1, TimeUnit.Years)
    if val in (2, 3, 4, 6, 12):
        return Period(12 // val, TimeUnit.Months)
    if val == 52:
        return Period(1, TimeUnit.Weeks)
    if val in (365, -1):
        return Period(1, TimeUnit.Days)
    qassert.fail(f"unknown frequency: {f}")


def _days_min_max(p: Period) -> tuple[int, int]:
    """C++ parity: amortizingfixedratebond.cpp anonymous-ns ``daysMinMax``."""
    if p.units == TimeUnit.Days:
        return (p.length, p.length)
    if p.units == TimeUnit.Weeks:
        return (7 * p.length, 7 * p.length)
    if p.units == TimeUnit.Months:
        return (28 * p.length, 31 * p.length)
    if p.units == TimeUnit.Years:
        return (365 * p.length, 366 * p.length)
    qassert.fail(f"unknown time unit ({p.units})")


def _is_sub_period(sub: Period, super_: Period) -> tuple[bool, int]:
    """C++ parity: amortizingfixedratebond.cpp anonymous-ns ``isSubPeriod``.

    Returns (matches, n_periods).
    """
    super_min, super_max = _days_min_max(super_)
    sub_min, sub_max = _days_min_max(sub)
    min_ratio = super_min / sub_max
    max_ratio = super_max / sub_min
    low = math.floor(min_ratio)
    high = math.ceil(max_ratio)
    for i in range(low, high + 1):
        test = sub * i
        if test == super_:
            return True, i
    return False, 0


def sinking_notionals(
    bond_length: Period,
    sinking_frequency: Frequency,
    coupon_rate: float,
    initial_notional: float,
) -> list[float]:
    """Build a French-amortisation notional schedule.

    # C++ parity: amortizingfixedratebond.cpp:123-150.
    """
    matched, n_periods = _is_sub_period(_period_from_frequency(sinking_frequency), bond_length)
    qassert.require(matched, "Bond frequency is incompatible with the maturity tenor")
    notionals: list[float] = [0.0] * (n_periods + 1)
    notionals[0] = initial_notional
    coupon = coupon_rate / float(int(sinking_frequency))
    compounded_interest = 1.0
    total_value = (1.0 + coupon) ** n_periods
    for i in range(n_periods - 1):
        compounded_interest *= 1.0 + coupon
        if coupon < 1.0e-12:
            current = initial_notional * (1.0 - (i + 1.0) / n_periods)
        else:
            current = initial_notional * (
                compounded_interest - (compounded_interest - 1.0) / (1.0 - 1.0 / total_value)
            )
        notionals[i + 1] = current
    notionals[-1] = 0.0
    return notionals


__all__ = [
    "AmortizingFixedRateBond",
    "sinking_notionals",
    "sinking_schedule",
]
