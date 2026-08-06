"""CmsRateBond — bond whose coupons pay a CMS (swap-rate) fixing.

# C++ parity: ql/instruments/bonds/cmsratebond.{hpp,cpp} (v1.43).

The class is a pure pass-through: it forwards every constructor argument to
:class:`~pquantlib.cashflows.cms_coupon.CmsLeg` and then builds the single
redemption on top of the resulting coupons.

Pricing a CMS coupon needs a :class:`CmsCouponPricer
<pquantlib.cashflows.cms_coupon_pricer.CmsCouponPricer>` — ``CmsLeg`` attaches
none (neither does C++), so callers must
:func:`~pquantlib.cashflows.coupon_pricer.set_coupon_pricer` a Hagan
replication pricer (e.g.
:class:`~pquantlib.pricingengines.conundrum_pricer.AnalyticHaganPricer`) on
``bond.cashflows()`` before asking for an NPV, exactly as the C++ test suite
does in ``test-suite/assetswap.cpp``.
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


class CmsRateBond(Bond):
    """CMS-rate bond.

    # C++ parity: cmsratebond.cpp:31-70.
    """

    def __init__(
        self,
        settlement_days: int,
        face_amount: float,
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
        redemption: float = 100.0,
        issue_date: Date | None = None,
    ) -> None:
        Bond.__init__(self, settlement_days, schedule.calendar, issue_date)
        self._maturity_date = schedule.end_date

        # C++ passes ``Null<Natural>()`` down to the coupon, which resolves it
        # to ``index->fixingDays()``; leaving the setter unset reaches the same
        # coupon through ``detail::get(fixingDays, i, index->fixingDays())``.
        builder = (
            CmsLeg(schedule, swap_index)
            .with_notionals(face_amount)
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

        self._add_redemptions_to_cashflows([redemption])

        qassert.require(len(self._cashflows) > 0, "bond with no cashflows!")
        qassert.require(len(self._redemptions) == 1, "multiple redemptions created")

        swap_index.register_with(self)
        for cf in self._cashflows:
            cf.register_with(self)


__all__ = ["CmsRateBond"]
