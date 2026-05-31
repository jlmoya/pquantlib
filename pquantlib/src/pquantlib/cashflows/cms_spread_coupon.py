"""CmsSpreadCoupon — a coupon paying a CMS spread.

# C++ parity: ql/experimental/coupons/cmsspreadcoupon.hpp + .cpp (v1.42.1,
# 099987f0).

A :class:`~pquantlib.cashflows.floating_rate_coupon.FloatingRateCoupon` whose
underlying index is a
:class:`~pquantlib.indexes.swap_spread_index.SwapSpreadIndex`. The coupon is
inert without a pricer (``rate()`` requires one); the bivariate-lognormal
``LognormalCmsSpreadPricer`` that prices it is **deferred** in PQuantLib (it
depends on the not-ported ``CmsCoupon`` / ``CmsCouponPricer`` family — see
``docs/carve-outs.md``). This class ports the coupon shell so a CMS-spread leg
can be *built* and inspected; pricing lands when the CMS-coupon foundation is
ported.

# C++ parity divergence (Visitability): the C++ ``accept`` /
# ``AcyclicVisitor`` dispatch is not ported (PQuantLib does not port the
# coupon visitor pattern — consistent with FloatingRateCoupon, which has no
# ``accept``).
"""

from __future__ import annotations

from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.swap_spread_index import SwapSpreadIndex
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date


class CmsSpreadCoupon(FloatingRateCoupon):
    """A coupon paying a (gearing/spread-adjusted) CMS spread.

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
        index: SwapSpreadIndex,
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
            # SwapSpreadIndex *is* one. PQuantLib's FloatingRateCoupon narrows
            # the static type to the Ibor/Overnight protocols, so we suppress
            # the mismatch here — the only index methods the coupon touches
            # (fixing/day_counter/fixing_calendar) are provided by
            # SwapSpreadIndex.
            index,  # type: ignore[arg-type]
            gearing,
            spread,
            ref_period_start,
            ref_period_end,
            day_counter,
            is_in_arrears,
            ex_coupon_date,
            fixing_convention,
        )
        self._swap_spread_index: SwapSpreadIndex = index

    def swap_spread_index(self) -> SwapSpreadIndex:
        return self._swap_spread_index
