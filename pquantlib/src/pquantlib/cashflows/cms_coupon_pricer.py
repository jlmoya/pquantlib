"""CmsCouponPricer — base pricer for CMS coupons.

# C++ parity: ql/cashflows/couponpricer.hpp (v1.42.1, 099987f0) —
# ``CmsCouponPricer`` + the ``MeanRevertingPricer`` mix-in.

``CmsCouponPricer`` is the abstract base for CMS-coupon pricers; it holds a
:class:`~pquantlib.termstructures.volatility.swaption.swaption_volatility_structure.SwaptionVolatilityStructure`
(the vol input the Hagan / conundrum replication integrates against) and
exposes ``swaption_volatility`` / ``set_swaption_volatility``.

``MeanRevertingPricer`` is the (abstract) mix-in for CMS pricers carrying a
mean-reversion quote (calibratable to CMS market quotes).

# C++ parity divergence (Handle vs object): C++ stores a
# ``Handle<SwaptionVolatilityStructure>``; PQuantLib threads the vol structure
# directly (the project does not port the relinkable-Handle indirection — the
# vol object can be swapped via ``set_swaption_volatility``).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from pquantlib.cashflows.coupon_pricer import CouponPricer

if TYPE_CHECKING:
    from pquantlib.quotes.quote import Quote
    from pquantlib.termstructures.volatility.swaption.swaption_volatility_structure import (
        SwaptionVolatilityStructure,
    )


class CmsCouponPricer(CouponPricer):
    """Base pricer for vanilla CMS coupons.

    C++ parity: ql/cashflows/couponpricer.hpp:149-170.
    """

    def __init__(self, swaption_vol: SwaptionVolatilityStructure | None = None) -> None:
        super().__init__()
        self._swaption_vol: SwaptionVolatilityStructure | None = swaption_vol

    def swaption_volatility(self) -> SwaptionVolatilityStructure | None:
        return self._swaption_vol

    def set_swaption_volatility(
        self, swaption_vol: SwaptionVolatilityStructure | None = None
    ) -> None:
        """C++ parity: ``CmsCouponPricer::setSwaptionVolatility`` — re-link the
        vol structure and notify observers."""
        self._swaption_vol = swaption_vol
        self.update()


class MeanRevertingPricer:
    """Mix-in for CMS pricers carrying a mean-reversion parameter.

    C++ parity: ql/cashflows/couponpricer.hpp:172-179 (``MeanRevertingPricer``).
    """

    @abstractmethod
    def mean_reversion(self) -> float: ...

    @abstractmethod
    def set_mean_reversion(self, mean_reversion: Quote) -> None: ...


__all__ = ["CmsCouponPricer", "MeanRevertingPricer"]
