"""ibor_leg — free-function IborLeg builder.

# C++ parity: ql/cashflows/iborcoupon.hpp class ``IborLeg`` (operator Leg()).

Same Python-idiomatic divergence as ``fixed_rate_leg``: the C++
chained-builder ``with*`` setters become keyword arguments on a free
function. Carve-outs (deferred):

- ``with_caps`` / ``with_floors`` (cap/floor coupons require
  OptionletVolatilityStructure — deferred).
- ``with_zero_payments`` / ``with_payment_lag`` / ``with_ex_coupon_period``.
- ``with_indexed_coupons`` / ``with_at_par_coupons`` (Settings toggle).
- Per-period fixing_days / gearings / spreads vectors collapse to
  scalar-or-uniform-list for L2-D coverage.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.ibor_coupon import IborCoupon
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.schedule import Schedule

if TYPE_CHECKING:
    from pquantlib.termstructures.protocols import IborIndexProtocol


def _scalar_or_seq(value: float | Sequence[float], i: int, default: float) -> float:
    if isinstance(value, int | float):
        return float(value)
    return float(value[i] if i < len(value) else value[-1] if len(value) > 0 else default)


def ibor_leg(
    schedule: Schedule,
    index: IborIndexProtocol,
    nominals: Sequence[float],
    *,
    payment_day_counter: DayCounter | None = None,
    payment_adjustment: BusinessDayConvention = BusinessDayConvention.Following,
    payment_calendar: Calendar | None = None,
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
    """
    qassert.require(len(nominals) > 0, "no notional given")
    qassert.require(len(schedule) >= 2, "schedule has fewer than 2 dates")

    cal = payment_calendar if payment_calendar is not None else schedule.calendar
    dc = payment_day_counter if payment_day_counter is not None else index.day_counter()
    fix_days = fixing_days if fixing_days is not None else index.fixing_days()

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
            IborCoupon(
                payment_date,
                nominal_val,
                start,
                end,
                fix_days,
                index,
                g,
                s,
                None,
                None,
                dc,
                in_arrears,
                None,
                fixing_convention,
            )
        )
    return leg
