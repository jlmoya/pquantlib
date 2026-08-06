"""fixed_rate_leg — keyword-argument façade over :class:`FixedRateLeg`.

# C++ parity: ql/cashflows/fixedratecoupon.hpp class ``FixedRateLeg`` (v1.43).

The chained builder itself lives next to the coupon it builds, in
:mod:`pquantlib.cashflows.fixed_rate_coupon`. This module keeps the
older keyword-argument entry point that the instrument layer already calls;
it constructs a :class:`~pquantlib.cashflows.fixed_rate_coupon.FixedRateLeg`
and builds it, so there is exactly one implementation of the leg logic.

The façade covers the common subset of the C++ setter surface. For
``with_first_period_day_counter``, ``with_last_period_day_counter`` and
``with_ex_coupon_period``, use the builder class directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pquantlib.cashflows.fixed_rate_coupon import FixedRateLeg
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency

if TYPE_CHECKING:
    from pquantlib.cashflows.cash_flow import CashFlow
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.interest_rate import InterestRate
    from pquantlib.time.calendar import Calendar
    from pquantlib.time.schedule import Schedule


def fixed_rate_leg(
    schedule: Schedule,
    nominals: Sequence[float],
    rates: Sequence[float] | Sequence[InterestRate],
    day_counter: DayCounter | None = None,
    compounding: Compounding = Compounding.Simple,
    frequency: Frequency = Frequency.Annual,
    payment_adjustment: BusinessDayConvention = BusinessDayConvention.Following,
    payment_calendar: Calendar | None = None,
    payment_lag: int = 0,
) -> list[CashFlow]:
    """Build a leg of FixedRateCoupons from a schedule.

    C++ parity: ql/cashflows/fixedratecoupon.cpp:180-275 ``FixedRateLeg::operator Leg()``.

    Either ``rates`` is a sequence of plain floats — in which case
    ``day_counter`` (required) + ``compounding`` + ``frequency`` define
    the InterestRate convention — or ``rates`` is a sequence of
    fully-built ``InterestRate`` instances (in which case ``day_counter``
    is ignored).

    Sequence semantics mirror C++: if the rates / nominals list is
    shorter than the number of coupon periods, the last value is
    repeated. Empty rates or empty nominals raises.

    ``payment_lag`` mirrors C++ ``withPaymentLag``: the payment date is
    ``payment_calendar.advance(end, payment_lag, Days, payment_adjustment)``.
    A lag of 0 collapses to ``adjust(end, payment_adjustment)``.
    """
    builder = (
        FixedRateLeg(schedule)
        .with_notionals(nominals)
        .with_coupon_rates(rates, day_counter, compounding, frequency)
        .with_payment_adjustment(payment_adjustment)
        .with_payment_lag(payment_lag)
    )
    if payment_calendar is not None:
        builder = builder.with_payment_calendar(payment_calendar)
    return builder.build()


__all__ = ["fixed_rate_leg"]
