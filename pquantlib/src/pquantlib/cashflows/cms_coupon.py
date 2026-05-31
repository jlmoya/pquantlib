"""CmsCoupon — a coupon paying a CMS (constant-maturity-swap) rate.

# C++ parity: ql/cashflows/cmscoupon.hpp + .cpp (v1.42.1, 099987f0).

A :class:`~pquantlib.cashflows.floating_rate_coupon.FloatingRateCoupon` whose
underlying index is a :class:`~pquantlib.indexes.swap_index.SwapIndex`. The
coupon's rate is the (convexity-adjusted) par swap rate, set by a
:class:`~pquantlib.cashflows.cms_coupon_pricer.CmsCouponPricer` (Hagan /
conundrum replication) attached via ``set_pricer``.

# C++ parity divergence (Visitability): the C++ ``accept`` / ``AcyclicVisitor``
# dispatch is not ported (PQuantLib does not port the coupon visitor pattern —
# consistent with FloatingRateCoupon, which has no ``accept``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.swap_index import SwapIndex
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date
from pquantlib.time.schedule import Schedule

if TYPE_CHECKING:
    from pquantlib.time.calendar import Calendar


class CmsCoupon(FloatingRateCoupon):
    """A coupon paying a (gearing/spread-adjusted) CMS rate.

    .. warning:: As in C++, no date adjustment is performed: the start and end
       dates passed should already be rolled to business days.
    """

    def __init__(
        self,
        payment_date: Date,
        nominal: float,
        start_date: Date,
        end_date: Date,
        fixing_days: int,
        swap_index: SwapIndex,
        gearing: float = 1.0,
        spread: float = 0.0,
        ref_period_start: Date | None = None,
        ref_period_end: Date | None = None,
        day_counter: DayCounter | None = None,
        is_in_arrears: bool = False,
        ex_coupon_date: Date | None = None,
        fixing_convention: BusinessDayConvention = BusinessDayConvention.Preceding,
    ) -> None:
        super().__init__(
            payment_date,
            nominal,
            start_date,
            end_date,
            fixing_days,
            # C++ parity: FloatingRateCoupon takes an InterestRateIndex;
            # SwapIndex *is* one. PQuantLib's FloatingRateCoupon narrows the
            # static type to the Ibor/Overnight protocols, so we suppress the
            # mismatch here — the coupon only touches index methods
            # (fixing/day_counter/fixing_calendar) that SwapIndex provides.
            swap_index,  # type: ignore[arg-type]
            gearing,
            spread,
            ref_period_start,
            ref_period_end,
            day_counter,
            is_in_arrears,
            ex_coupon_date,
            fixing_convention,
        )
        self._swap_index: SwapIndex = swap_index

    def swap_index(self) -> SwapIndex:
        return self._swap_index


def _scalar_or_seq(value: float | Sequence[float], i: int, default: float) -> float:
    if isinstance(value, int | float):
        return float(value)
    return float(value[i] if i < len(value) else value[-1] if len(value) > 0 else default)


def cms_leg(
    schedule: Schedule,
    swap_index: SwapIndex,
    nominals: float | Sequence[float],
    *,
    payment_day_counter: DayCounter | None = None,
    payment_adjustment: BusinessDayConvention = BusinessDayConvention.Following,
    fixing_days: int | None = None,
    gearings: float | Sequence[float] = 1.0,
    spreads: float | Sequence[float] = 0.0,
    caps: float | Sequence[float] | None = None,
    floors: float | Sequence[float] | None = None,
    in_arrears: bool = False,
    fixing_convention: BusinessDayConvention = BusinessDayConvention.Preceding,
) -> list[CashFlow]:
    """Build a leg of CmsCoupons from a schedule + swap index.

    # C++ parity: ql/cashflows/cmscoupon.cpp ``CmsLeg::operator Leg()`` — the
    # chained-builder ``with*`` setters become keyword arguments on this free
    # function (same Python-idiomatic divergence as ``ibor_leg``).

    ``fixing_days`` defaults to ``swap_index.fixing_days()``.
    ``payment_day_counter`` defaults to ``swap_index.day_counter()``.

    .. note:: Capped / floored CMS coupons (``caps`` / ``floors``) require
       ``CappedFlooredCmsCoupon``, which lands in W12-B — passing non-``None``
       caps/floors here raises. Plain (uncapped) CMS legs are fully supported.
    """
    if caps is not None or floors is not None:
        qassert.fail(
            "capped/floored CMS legs require CappedFlooredCmsCoupon "
            "(deferred to W12-B); pass caps=floors=None for a plain CMS leg"
        )

    eff_fixing_days = (
        fixing_days if fixing_days is not None else swap_index.fixing_days()
    )
    dc = payment_day_counter if payment_day_counter is not None else swap_index.day_counter()
    cal: Calendar = schedule.calendar

    leg: list[CashFlow] = []
    n = len(schedule) - 1
    for i in range(n):
        start = schedule[i]
        end = schedule[i + 1]
        payment_date = cal.adjust(end, payment_adjustment)
        nominal = _scalar_or_seq(nominals, i, 0.0)
        gearing = _scalar_or_seq(gearings, i, 1.0)
        spread = _scalar_or_seq(spreads, i, 0.0)
        leg.append(
            CmsCoupon(
                payment_date,
                nominal,
                start,
                end,
                eff_fixing_days,
                swap_index,
                gearing,
                spread,
                start,
                end,
                dc,
                in_arrears,
                None,
                fixing_convention,
            )
        )
    return leg


__all__ = ["CmsCoupon", "cms_leg"]
