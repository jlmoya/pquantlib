"""overnight_leg — free-function OvernightLeg builder.

# C++ parity: ql/cashflows/overnightindexedcoupon.hpp class ``OvernightLeg``.

Same Python-idiomatic divergence as ``fixed_rate_leg``: the C++
chained-builder ``with*`` setters become keyword arguments on a free
function. Carve-outs (deferred): lookback_days, lockout_days,
observation_shift, compound_spread_daily, caps / floors, naked_option,
last_recent_period, custom payment_dates, telescopic_value_dates.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.overnight_indexed_coupon import OvernightIndexedCoupon
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.schedule import Schedule

if TYPE_CHECKING:
    from pquantlib.termstructures.protocols import OvernightIndexProtocol


def _scalar_or_seq(value: float | Sequence[float], i: int, default: float) -> float:
    if isinstance(value, int | float):
        return float(value)
    return float(value[i] if i < len(value) else value[-1] if len(value) > 0 else default)


def overnight_leg(
    schedule: Schedule,
    index: OvernightIndexProtocol,
    nominals: Sequence[float],
    *,
    payment_day_counter: DayCounter | None = None,
    payment_adjustment: BusinessDayConvention = BusinessDayConvention.Following,
    payment_calendar: Calendar | None = None,
    gearings: float | Sequence[float] = 1.0,
    spreads: float | Sequence[float] = 0.0,
) -> list[CashFlow]:
    """Build a leg of OvernightIndexedCoupons from a schedule.

    C++ parity: ql/cashflows/overnightindexedcoupon.cpp
    ``OvernightLeg::operator Leg()``.
    """
    qassert.require(len(nominals) > 0, "no notional given")
    qassert.require(len(schedule) >= 2, "schedule has fewer than 2 dates")

    cal = payment_calendar if payment_calendar is not None else schedule.calendar
    dc = payment_day_counter if payment_day_counter is not None else index.day_counter()

    leg: list[CashFlow] = []
    n_periods = len(schedule) - 1
    for i in range(n_periods):
        start = schedule.date(i)
        end = schedule.date(i + 1)
        payment_date = cal.adjust(end, payment_adjustment)
        nominal_val = float(nominals[i] if i < len(nominals) else nominals[-1])
        g = _scalar_or_seq(gearings, i, 1.0)
        s = _scalar_or_seq(spreads, i, 0.0)
        leg.append(
            OvernightIndexedCoupon(
                payment_date,
                nominal_val,
                start,
                end,
                index,
                g,
                s,
                None,
                None,
                dc,
            )
        )
    return leg
