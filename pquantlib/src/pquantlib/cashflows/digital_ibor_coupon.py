"""DigitalIborCoupon + digital_ibor_leg — digital option on an Ibor coupon.

# C++ parity: ql/cashflows/digitaliborcoupon.{hpp,cpp} (v1.42.1, 099987f0).

:class:`DigitalIborCoupon` is a :class:`~pquantlib.cashflows.digital_coupon.DigitalCoupon`
whose underlying is an :class:`~pquantlib.cashflows.ibor_coupon.IborCoupon`.
``digital_ibor_leg`` is the free-function port of the C++ ``DigitalIborLeg``
chained builder (the ``with*`` setters become keyword arguments — same idiom
as :func:`~pquantlib.cashflows.ibor_leg.ibor_leg`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.digital_coupon import DigitalCoupon
from pquantlib.cashflows.ibor_coupon import IborCoupon
from pquantlib.position import PositionType
from pquantlib.time.business_day_convention import BusinessDayConvention

if TYPE_CHECKING:
    from pquantlib.cashflows.replication import DigitalReplication
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.termstructures.protocols import IborIndexProtocol
    from pquantlib.time.schedule import Schedule


class DigitalIborCoupon(DigitalCoupon):
    """Digital call/put option on an Ibor coupon.

    # C++ parity: ql/cashflows/digitaliborcoupon.hpp:36-55 + .cpp:28-42.
    """

    def __init__(
        self,
        underlying: IborCoupon,
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


def digital_ibor_leg(
    schedule: Schedule,
    index: IborIndexProtocol,
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
    """Build a leg of :class:`DigitalIborCoupon` from a schedule + Ibor index.

    # C++ parity: ql/cashflows/digitaliborcoupon.cpp ``DigitalIborLeg::operator
    # Leg()`` (via ``FloatingDigitalLeg``). The chained ``with*`` setters become
    # keyword arguments. Per-period vectors collapse to scalar-or-uniform-list.

    ``fixing_days`` defaults to ``index.fixing_days()``; ``payment_day_counter``
    defaults to ``index.day_counter()``.
    """
    eff_fixing_days = fixing_days if fixing_days is not None else index.fixing_days()
    dc = payment_day_counter if payment_day_counter is not None else index.day_counter()
    cal = schedule.calendar

    leg: list[CashFlow] = []
    n = len(schedule) - 1
    for i in range(n):
        start = schedule[i]
        end = schedule[i + 1]
        payment_date = cal.adjust(end, payment_adjustment)
        nominal = _seq(nominals, i, 0.0)
        assert nominal is not None
        underlying = IborCoupon(
            payment_date,
            nominal,
            start,
            end,
            eff_fixing_days,
            index,
            _seq(gearings, i, 1.0) or 1.0,
            _seq(spreads, i, 0.0) or 0.0,
            start,
            end,
            dc,
            in_arrears,
        )
        leg.append(
            DigitalIborCoupon(
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


__all__ = ["DigitalIborCoupon", "digital_ibor_leg"]
