"""ibor_leg — keyword-argument façade over :class:`IborLeg`.

# C++ parity: ql/cashflows/iborcoupon.hpp class ``IborLeg`` (v1.43).

The chained builder itself lives next to the coupon it builds, in
:mod:`pquantlib.cashflows.ibor_coupon`. This module keeps the older
keyword-argument entry point that the instrument layer already calls; it
constructs an :class:`~pquantlib.cashflows.ibor_coupon.IborLeg` and builds
it, so there is exactly one implementation of the leg logic.

The façade covers the common subset of the C++ setter surface. For
``with_caps`` / ``with_floors`` / ``with_zero_payments`` /
``with_ex_coupon_period`` / ``with_indexed_coupons`` / ``with_at_par_coupons``
and per-period fixing-day vectors, use the builder class directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pquantlib.cashflows.ibor_coupon import IborLeg
from pquantlib.time.business_day_convention import BusinessDayConvention

if TYPE_CHECKING:
    from pquantlib.cashflows.cash_flow import CashFlow
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.termstructures.protocols import IborIndexProtocol
    from pquantlib.time.calendar import Calendar
    from pquantlib.time.schedule import Schedule


def ibor_leg(
    schedule: Schedule,
    index: IborIndexProtocol,
    nominals: Sequence[float],
    *,
    payment_day_counter: DayCounter | None = None,
    payment_adjustment: BusinessDayConvention = BusinessDayConvention.Following,
    payment_calendar: Calendar | None = None,
    payment_lag: int = 0,
    fixing_days: int | None = None,
    gearings: float | Sequence[float] = 1.0,
    spreads: float | Sequence[float] = 0.0,
    in_arrears: bool = False,
    fixing_convention: BusinessDayConvention = BusinessDayConvention.Preceding,
) -> list[CashFlow]:
    """Build a leg of IborCoupons from a schedule + IBOR index.

    C++ parity: ql/cashflows/iborcoupon.cpp ``IborLeg::operator Leg()``.

    ``fixing_days`` defaults to ``index.fixing_days()``. ``payment_day_counter``
    defaults to ``index.day_counter()``.

    ``payment_lag`` mirrors C++ ``withPaymentLag``: the payment date is
    ``payment_calendar.advance(end, payment_lag, Days, payment_adjustment)``.
    A lag of 0 collapses to ``adjust(end, payment_adjustment)``.
    """
    builder = (
        IborLeg(schedule, index)
        .with_notionals(nominals)
        .with_payment_adjustment(payment_adjustment)
        .with_payment_lag(payment_lag)
        .with_gearings(gearings)
        .with_spreads(spreads)
        .in_arrears(in_arrears)
        .with_fixing_convention(fixing_convention)
    )
    if payment_day_counter is not None:
        builder = builder.with_payment_day_counter(payment_day_counter)
    if payment_calendar is not None:
        builder = builder.with_payment_calendar(payment_calendar)
    if fixing_days is not None:
        builder = builder.with_fixing_days(fixing_days)
    return builder.build()


__all__ = ["ibor_leg"]
