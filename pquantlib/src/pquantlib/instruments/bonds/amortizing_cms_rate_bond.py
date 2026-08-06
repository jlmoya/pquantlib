"""AmortizingCmsRateBond — CMS-rate bond with a per-period notional vector.

# C++ parity: ql/instruments/bonds/amortizingcmsratebond.{hpp,cpp} (v1.43).

Same leg machinery as :class:`~pquantlib.instruments.bonds.cms_rate_bond.CmsRateBond`,
but the notional is a vector (one entry per coupon period, the last entry
repeating for any remaining periods) and the redemption is a vector too — so
each amortisation step can be redeemed at its own price.

The notional vector drives ``AmortizingPayment`` cashflows via
:meth:`Bond._add_redemptions_to_cashflows
<pquantlib.instruments.bond.Bond._add_redemptions_to_cashflows>`; a sinking-fund
schedule can be produced with
:func:`~pquantlib.instruments.bonds.amortizing_fixed_rate_bond.sinking_notionals`.

As for ``CmsRateBond``, the coupons carry no pricer until the caller attaches
one (see that class's docstring).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.cms_coupon import CmsLeg
from pquantlib.instruments.bond import Bond
from pquantlib.time.business_day_convention import BusinessDayConvention

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.indexes.swap_index import SwapIndex
    from pquantlib.time.date import Date
    from pquantlib.time.schedule import Schedule


class AmortizingCmsRateBond(Bond):
    """Amortizing CMS-rate bond.

    # C++ parity: amortizingcmsratebond.cpp:29-64.
    """

    def __init__(
        self,
        settlement_days: int,
        notionals: Sequence[float],
        schedule: Schedule,
        swap_index: SwapIndex,
        payment_day_counter: DayCounter,
        payment_convention: BusinessDayConvention = BusinessDayConvention.Following,
        fixing_days: int | None = None,
        gearings: float | Sequence[float] = (1.0,),
        spreads: float | Sequence[float] = (0.0,),
        caps: float | Sequence[float] | None = None,
        floors: float | Sequence[float] | None = None,
        in_arrears: bool = False,
        issue_date: Date | None = None,
        redemptions: Sequence[float] = (100.0,),
    ) -> None:
        Bond.__init__(self, settlement_days, schedule.calendar, issue_date)
        self._maturity_date = schedule.end_date

        builder = (
            CmsLeg(schedule, swap_index)
            .with_notionals(notionals)
            .with_payment_day_counter(payment_day_counter)
            .with_payment_adjustment(payment_convention)
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
        self._cashflows = builder.build()

        self._add_redemptions_to_cashflows(list(redemptions))

        # C++ checks only the non-emptiness here: an amortising bond has one
        # redemption per notional step, so ``redemptions_.size() == 1`` does
        # not hold (amortizingcmsratebond.cpp:62).
        qassert.require(len(self._cashflows) > 0, "bond with no cashflows!")

        swap_index.register_with(self)
        for cf in self._cashflows:
            cf.register_with(self)


__all__ = ["AmortizingCmsRateBond"]
