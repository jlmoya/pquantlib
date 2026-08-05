"""overnight_leg — keyword-argument façade over :class:`OvernightLeg`.

# C++ parity: ql/cashflows/overnightindexedcoupon.hpp class ``OvernightLeg``
  (v1.43).

The chained builder itself lives next to the coupon it builds, in
:mod:`pquantlib.cashflows.overnight_indexed_coupon`. This module keeps the
older keyword-argument entry point that the instrument layer already calls;
it constructs an
:class:`~pquantlib.cashflows.overnight_indexed_coupon.OvernightLeg` and
builds it, so there is exactly one implementation of the leg logic.

The façade covers the common subset of the C++ setter surface. For caps,
floors, naked options, averaging method, in-advance fixing, last-recent
period, explicit payment dates and a custom coupon pricer, use the builder
class directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pquantlib.cashflows.overnight_indexed_coupon import OvernightLeg
from pquantlib.time.business_day_convention import BusinessDayConvention

if TYPE_CHECKING:
    from pquantlib.cashflows.cash_flow import CashFlow
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.termstructures.protocols import OvernightIndexProtocol
    from pquantlib.time.calendar import Calendar
    from pquantlib.time.schedule import Schedule


def overnight_leg(
    schedule: Schedule,
    index: OvernightIndexProtocol,
    nominals: Sequence[float],
    *,
    payment_day_counter: DayCounter | None = None,
    payment_adjustment: BusinessDayConvention = BusinessDayConvention.Following,
    payment_calendar: Calendar | None = None,
    payment_lag: int = 0,
    gearings: float | Sequence[float] = 1.0,
    spreads: float | Sequence[float] = 0.0,
) -> list[CashFlow]:
    """Build a leg of OvernightIndexedCoupons from a schedule.

    C++ parity: ql/cashflows/overnightindexedcoupon.cpp
    ``OvernightLeg::operator Leg()``.

    ``payment_lag`` mirrors C++ ``withPaymentLag``: the payment date is
    ``payment_calendar.advance(end, payment_lag, Days, payment_adjustment)``.
    A lag of 0 collapses to ``adjust(end, payment_adjustment)``. Overnight
    legs are the main user of the lag — a compounded overnight coupon only
    fixes on its accrual end date, so it is normally paid a day or two later.
    """
    builder = (
        OvernightLeg(schedule, index)
        .with_notionals(nominals)
        .with_payment_adjustment(payment_adjustment)
        .with_payment_lag(payment_lag)
        .with_gearings(gearings)
        .with_spreads(spreads)
    )
    if payment_day_counter is not None:
        builder = builder.with_payment_day_counter(payment_day_counter)
    if payment_calendar is not None:
        builder = builder.with_payment_calendar(payment_calendar)
    return builder.build()


__all__ = ["overnight_leg"]
