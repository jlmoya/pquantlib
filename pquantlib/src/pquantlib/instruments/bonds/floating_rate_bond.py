"""FloatingRateBond — IBOR-indexed floating-rate bond.

# C++ parity: ql/instruments/bonds/floatingratebond.{hpp,cpp} (v1.42.1).

Carve-outs:
- ``caps`` / ``floors`` are accepted as inputs but ignored by the
  underlying L2-D ``ibor_leg`` builder (cap/floor coupons need
  OptionletVolatilityStructure; deferred to a later cluster).
- ``ex_coupon_period`` / ``ex_coupon_calendar`` likewise deferred at
  the ibor_leg builder level.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.ibor_leg import ibor_leg
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.instruments.bond import Bond
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date
from pquantlib.time.schedule import Schedule

if TYPE_CHECKING:
    from pquantlib.termstructures.protocols import IborIndexProtocol


class FloatingRateBond(Bond):
    """Floating-rate (IBOR-indexed) bond.

    # C++ parity: floatingratebond.cpp:30-73.
    """

    def __init__(
        self,
        settlement_days: int,
        face_amount: float,
        schedule: Schedule,
        ibor_index: IborIndexProtocol,
        accrual_day_counter: DayCounter,
        payment_convention: BusinessDayConvention = BusinessDayConvention.Following,
        fixing_days: int | None = None,
        gearings: float | Sequence[float] = 1.0,
        spreads: float | Sequence[float] = 0.0,
        caps: Sequence[float] | None = None,
        floors: Sequence[float] | None = None,
        in_arrears: bool = False,
        redemption: float = 100.0,
        issue_date: Date | None = None,
        fixing_convention: BusinessDayConvention = BusinessDayConvention.Preceding,
    ) -> None:
        del caps, floors  # accepted for parity, ignored — see module docstring.
        Bond.__init__(self, settlement_days, schedule.calendar, issue_date)
        self._maturity_date = schedule.end_date

        self._cashflows = list(
            ibor_leg(
                schedule,
                ibor_index,
                nominals=[face_amount],
                payment_day_counter=accrual_day_counter,
                payment_adjustment=payment_convention,
                fixing_days=fixing_days,
                gearings=gearings,
                spreads=spreads,
                in_arrears=in_arrears,
                fixing_convention=fixing_convention,
            )
        )

        self._add_redemptions_to_cashflows([redemption])

        qassert.require(len(self._cashflows) > 0, "bond with no cashflows!")
        qassert.require(len(self._redemptions) == 1, "multiple redemptions created")

        # Register on the index so an updated forecast curve propagates.
        reg = getattr(ibor_index, "register_with", None)
        if callable(reg):
            reg(self)
        for cf in self._cashflows:
            cf.register_with(self)


__all__ = ["FloatingRateBond"]
