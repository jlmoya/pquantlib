"""CmsSpreadCouponPricer — base pricer for CMS-spread coupons.

# C++ parity: ql/experimental/coupons/cmsspreadcoupon.hpp (v1.42.1, 099987f0)
# ``CmsSpreadCouponPricer``.

Abstract base holding the correlation quote between the two underlying swap
rates (the spread's bivariate dynamics input).

# C++ parity divergence (Handle vs object): C++ stores a ``Handle<Quote>``;
# PQuantLib threads the ``Quote`` object directly (no relinkable-Handle layer).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib.cashflows.coupon_pricer import CouponPricer

if TYPE_CHECKING:
    from pquantlib.quotes.quote import Quote


class CmsSpreadCouponPricer(CouponPricer):
    """Base pricer for CMS-spread coupons (holds the correlation quote).

    C++ parity: cmsspreadcoupon.hpp:141-161.
    """

    def __init__(self, correlation: Quote | None = None) -> None:
        super().__init__()
        self._correlation: Quote | None = correlation

    def correlation(self) -> Quote | None:
        return self._correlation

    def set_correlation(self, correlation: Quote | None = None) -> None:
        """C++ parity: ``CmsSpreadCouponPricer::setCorrelation``."""
        self._correlation = correlation
        self.update()


__all__ = ["CmsSpreadCouponPricer"]
