"""DigitalCmsCoupon + digital_cms_leg — digital option on a CMS coupon.

# C++ parity: ql/cashflows/digitalcmscoupon.{hpp,cpp} (v1.42.1, 099987f0).

:class:`DigitalCmsCoupon` is a :class:`~pquantlib.cashflows.digital_coupon.DigitalCoupon`
whose underlying is a :class:`~pquantlib.cashflows.cms_coupon.CmsCoupon` (W12-A).
``digital_cms_leg`` is the free-function port of the C++ ``DigitalCmsLeg``
chained builder.

.. note:: Replication of a CMS digital requires a CMS optionlet pricer (Hagan
   caplet/floorlet via static replication) — see the note on
   :class:`~pquantlib.cashflows.capped_floored_coupon.CappedFlooredCmsCoupon`.
   The structure is fully ported; the rate evaluates once such a pricer is
   wired (the base ``CmsCouponPricer`` does not price cap/floor).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.cms_coupon import CmsCoupon
from pquantlib.cashflows.digital_coupon import DigitalCoupon
from pquantlib.position import PositionType
from pquantlib.time.business_day_convention import BusinessDayConvention

if TYPE_CHECKING:
    from pquantlib.cashflows.replication import DigitalReplication
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.indexes.swap_index import SwapIndex
    from pquantlib.time.schedule import Schedule


class DigitalCmsCoupon(DigitalCoupon):
    """Digital call/put option on a CMS coupon.

    # C++ parity: ql/cashflows/digitalcmscoupon.hpp:36-55 + .cpp.
    """

    def __init__(
        self,
        underlying: CmsCoupon,
        call_strike: float | None = None,
        call_position: PositionType = PositionType.Long,
        is_call_atm_included: bool = False,
        call_digital_payoff: float | None = None,
        put_strike: float | None = None,
        put_position: PositionType = PositionType.Long,
        is_put_atm_included: bool = False,
        put_digital_payoff: float | None = None,
        replication: DigitalReplication | None = None,
        naked_option: bool = False,
    ) -> None:
        super().__init__(
            underlying,
            call_strike,
            call_position,
            is_call_atm_included,
            call_digital_payoff,
            put_strike,
            put_position,
            is_put_atm_included,
            put_digital_payoff,
            replication,
            naked_option,
        )


def _seq(value: float | Sequence[float] | None, i: int, default: float) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if len(value) == 0:
        return default
    return float(value[i] if i < len(value) else value[-1])


def digital_cms_leg(
    schedule: Schedule,
    swap_index: SwapIndex,
    nominals: float | Sequence[float],
    *,
    payment_day_counter: DayCounter | None = None,
    payment_adjustment: BusinessDayConvention = BusinessDayConvention.Following,
    fixing_days: int | None = None,
    gearings: float | Sequence[float] = 1.0,
    spreads: float | Sequence[float] = 0.0,
    in_arrears: bool = False,
    call_strikes: float | Sequence[float] | None = None,
    long_call_option: PositionType = PositionType.Long,
    call_atm: bool = False,
    call_payoffs: float | Sequence[float] | None = None,
    put_strikes: float | Sequence[float] | None = None,
    long_put_option: PositionType = PositionType.Long,
    put_atm: bool = False,
    put_payoffs: float | Sequence[float] | None = None,
    replication: DigitalReplication | None = None,
    naked_option: bool = False,
) -> list[CashFlow]:
    """Build a leg of :class:`DigitalCmsCoupon` from a schedule + swap index.

    # C++ parity: ql/cashflows/digitalcmscoupon.cpp ``DigitalCmsLeg::operator
    # Leg()`` (via ``FloatingDigitalLeg``). Chained ``with*`` setters become
    # keyword arguments; per-period vectors collapse to scalar-or-uniform-list.

    ``fixing_days`` defaults to ``swap_index.fixing_days()``;
    ``payment_day_counter`` defaults to ``swap_index.day_counter()``.
    """
    eff_fixing_days = fixing_days if fixing_days is not None else swap_index.fixing_days()
    dc = payment_day_counter if payment_day_counter is not None else swap_index.day_counter()
    cal = schedule.calendar

    leg: list[CashFlow] = []
    n = len(schedule) - 1
    for i in range(n):
        start = schedule[i]
        end = schedule[i + 1]
        payment_date = cal.adjust(end, payment_adjustment)
        nominal = _seq(nominals, i, 0.0)
        assert nominal is not None
        underlying = CmsCoupon(
            payment_date,
            nominal,
            start,
            end,
            eff_fixing_days,
            swap_index,
            _seq(gearings, i, 1.0) or 1.0,
            _seq(spreads, i, 0.0) or 0.0,
            start,
            end,
            dc,
            in_arrears,
        )
        leg.append(
            DigitalCmsCoupon(
                underlying,
                _seq(call_strikes, i, 0.0),
                long_call_option,
                call_atm,
                _seq(call_payoffs, i, 0.0),
                _seq(put_strikes, i, 0.0),
                long_put_option,
                put_atm,
                _seq(put_payoffs, i, 0.0),
                replication,
                naked_option,
            )
        )
    return leg


__all__ = ["DigitalCmsCoupon", "digital_cms_leg"]
