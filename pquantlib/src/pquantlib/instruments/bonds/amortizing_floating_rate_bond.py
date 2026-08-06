"""AmortizingFloatingRateBond — IBOR floater with a per-period notional vector.

# C++ parity: ql/instruments/bonds/amortizingfloatingratebond.{hpp,cpp} (v1.43).

Unlike :class:`~pquantlib.instruments.bonds.floating_rate_bond.FloatingRateBond`
this class drives the full :class:`~pquantlib.cashflows.ibor_coupon.IborLeg`
surface: the notional vector, caps/floors, the payment lag and all four
ex-coupon settings are forwarded, so nothing the constructor accepts is
dropped.

``IborLeg`` attaches a default :class:`BlackIborCouponPricer
<pquantlib.cashflows.coupon_pricer.BlackIborCouponPricer>` only when the leg
carries no caps, no floors and is not in arrears — exactly as C++ does
(iborcoupon.cpp ``IborLeg::operator Leg()``). For the capped / floored /
in-arrears cases the caller must attach a pricer carrying an optionlet
volatility with
:func:`~pquantlib.cashflows.coupon_pricer.set_coupon_pricer` before pricing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.ibor_coupon import IborLeg
from pquantlib.instruments.bond import Bond
from pquantlib.time.business_day_convention import BusinessDayConvention

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.termstructures.protocols import IborIndexProtocol
    from pquantlib.time.calendar import Calendar
    from pquantlib.time.date import Date
    from pquantlib.time.period import Period
    from pquantlib.time.schedule import Schedule


class AmortizingFloatingRateBond(Bond):
    """Amortizing floating-rate bond (possibly capped and/or floored).

    # C++ parity: amortizingfloatingratebond.cpp:29-75.
    """

    def __init__(
        self,
        settlement_days: int,
        notionals: Sequence[float],
        schedule: Schedule,
        ibor_index: IborIndexProtocol,
        payment_day_counter: DayCounter,
        payment_convention: BusinessDayConvention = BusinessDayConvention.Following,
        fixing_days: int | None = None,
        gearings: float | Sequence[float] = (1.0,),
        spreads: float | Sequence[float] = (0.0,),
        caps: float | Sequence[float] | None = None,
        floors: float | Sequence[float] | None = None,
        in_arrears: bool = False,
        issue_date: Date | None = None,
        ex_coupon_period: Period | None = None,
        ex_coupon_calendar: Calendar | None = None,
        ex_coupon_convention: BusinessDayConvention = BusinessDayConvention.Unadjusted,
        ex_coupon_end_of_month: bool = False,
        redemptions: Sequence[float] = (100.0,),
        payment_lag: int = 0,
    ) -> None:
        Bond.__init__(self, settlement_days, schedule.calendar, issue_date)
        self._maturity_date = schedule.end_date

        builder = (
            IborLeg(schedule, ibor_index)
            .with_notionals(notionals)
            .with_payment_day_counter(payment_day_counter)
            .with_payment_adjustment(payment_convention)
            .with_payment_lag(payment_lag)
            .with_gearings(gearings)
            .with_spreads(spreads)
            .in_arrears(in_arrears)
        )
        if fixing_days is not None:
            builder = builder.with_fixing_days(fixing_days)
        if caps is not None:
            builder = builder.with_caps(caps)
        if floors is not None:
            builder = builder.with_floors(floors)
        # C++ spells "no ex-coupon" as a default-constructed ``Period()`` and
        # always calls the setter; FloatingLeg then skips the ex-coupon date
        # when the period is null. Python spells it ``None``.
        if ex_coupon_period is not None:
            qassert.require(
                ex_coupon_calendar is not None,
                "an ex-coupon calendar is required when an ex-coupon period is given",
            )
            assert ex_coupon_calendar is not None
            builder = builder.with_ex_coupon_period(
                ex_coupon_period,
                ex_coupon_calendar,
                ex_coupon_convention,
                ex_coupon_end_of_month,
            )
        self._cashflows = builder.build()

        self._add_redemptions_to_cashflows(list(redemptions))

        qassert.require(len(self._cashflows) > 0, "bond with no cashflows!")

        # C++ ``registerWith(index)``. ``IborIndexProtocol`` does not declare
        # the Observable surface, so probe for it as FloatingRateBond does.
        register = getattr(ibor_index, "register_with", None)
        if callable(register):
            register(self)
        for cf in self._cashflows:
            cf.register_with(self)


__all__ = ["AmortizingFloatingRateBond"]
