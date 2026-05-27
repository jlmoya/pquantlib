"""CouponPricer hierarchy — pricers for floating-rate coupons.

# C++ parity: ql/cashflows/couponpricer.hpp + .cpp (v1.42.1).

The C++ hierarchy:

    FloatingRateCouponPricer (abstract, Observer+Observable)
      -> IborCouponPricer (abstract — capletVol, useIndexedCoupon)
            -> BlackIborCouponPricer (concrete — Black-vol with cap/floor)
      -> CmsCouponPricer (abstract)
            -> ...

Python port carve-outs (deferred to L4-style modelling work):
- ``OptionletVolatilityStructure`` (CapFloorVolTermStructure) is NOT
  ported — the cap / floor / optionletPrice machinery in
  BlackIborCouponPricer is consequently stubbed out (raises a
  LibraryException if called). Plain swaplet pricing (no cap, no floor)
  works.
- ``CmsCouponPricer`` and the MeanRevertingPricer mix-in are NOT ported.
- ``setCouponPricer`` / ``setCouponPricers`` free functions: simple
  variant ``set_coupon_pricer`` is provided for the most common case
  (one pricer applied to every floating coupon in the leg).

This keeps the public surface aligned with C++ shape while skipping the
cap/floor pricing machinery — sufficient for L2-D's plain
IborCoupon / OvernightIndexedCoupon use cases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.patterns.observer import Observable

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pquantlib.cashflows.cash_flow import CashFlow
    from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon


# -----------------------------------------------------------------------
# Abstract bases
# -----------------------------------------------------------------------


class CouponPricer(ABC, Observable):
    """Top-level pricer abstract base.

    Not present in the C++ hierarchy as a distinct class — C++ uses
    FloatingRateCouponPricer at the top. We expose the same name to
    match the L2-D design spec, and alias FloatingRateCouponPricer to it.
    """

    @abstractmethod
    def swaplet_price(self) -> float: ...

    @abstractmethod
    def swaplet_rate(self) -> float: ...

    @abstractmethod
    def caplet_price(self, effective_cap: float) -> float: ...

    @abstractmethod
    def caplet_rate(self, effective_cap: float) -> float: ...

    @abstractmethod
    def floorlet_price(self, effective_floor: float) -> float: ...

    @abstractmethod
    def floorlet_rate(self, effective_floor: float) -> float: ...

    @abstractmethod
    def initialize(self, coupon: FloatingRateCoupon) -> None: ...

    # Observer.update() — propagate to observers.
    def update(self) -> None:
        self.notify_observers()


# C++-name alias for downstream code that imports the more specific name.
FloatingRateCouponPricer = CouponPricer


# -----------------------------------------------------------------------
# IborCouponPricer — base for IBOR-style coupons
# -----------------------------------------------------------------------


class IborCouponPricer(CouponPricer):
    """Base pricer for plain (non-capped / non-floored) IBOR coupons.

    The C++ class is abstract (capletPrice / capletRate / ... are pure
    virtual). Here, we make it instantiable and provide a trivial
    implementation: swaplet_rate = gearing * indexFixing + spread (no
    convexity adjustment, no cap/floor handling).

    Cap / floor methods raise LibraryException — they require an
    OptionletVolatilityStructure (deferred carve-out for L2-D).
    """

    def __init__(self) -> None:
        super().__init__()
        self._coupon: FloatingRateCoupon | None = None
        self._gearing: float = 1.0
        self._spread: float = 0.0
        self._accrual_period: float = 0.0
        self._discount: float = 1.0

    # --- pricer wiring -------------------------------------------------

    def initialize(self, coupon: FloatingRateCoupon) -> None:
        """Capture per-coupon state needed by the rate / price methods.

        C++ parity: ql/cashflows/couponpricer.cpp ``IborCouponPricer::initialize``.
        """
        self._coupon = coupon
        self._gearing = coupon.gearing()
        self._spread = coupon.spread()
        self._accrual_period = coupon.accrual_period()

    def _adjusted_fixing(self, fixing: float | None = None) -> float:
        """No convexity adjustment in the plain pricer.

        C++ parity: ql/cashflows/couponpricer.cpp ``BlackIborCouponPricer::adjustedFixing``
        — here simplified to just return the raw fixing.
        """
        if fixing is None:
            qassert.require(self._coupon is not None, "coupon not set")
            assert self._coupon is not None
            fixing = self._coupon.index_fixing()
        return fixing

    # --- CouponPricer interface ----------------------------------------

    def swaplet_rate(self) -> float:
        """gearing * (adjusted) fixing + spread.

        C++ parity: ql/cashflows/couponpricer.hpp:215-217 (BlackIborCouponPricer
        inline; simplified — no Black-vol adjustment).
        """
        return self._gearing * self._adjusted_fixing() + self._spread

    def swaplet_price(self) -> float:
        return self.swaplet_rate() * self._accrual_period * self._discount

    def caplet_price(self, effective_cap: float) -> float:
        del effective_cap
        msg = "caplet pricing requires OptionletVolatilityStructure (deferred carve-out)"
        raise LibraryException(msg)

    def caplet_rate(self, effective_cap: float) -> float:
        del effective_cap
        msg = "caplet pricing requires OptionletVolatilityStructure (deferred carve-out)"
        raise LibraryException(msg)

    def floorlet_price(self, effective_floor: float) -> float:
        del effective_floor
        msg = "floorlet pricing requires OptionletVolatilityStructure (deferred carve-out)"
        raise LibraryException(msg)

    def floorlet_rate(self, effective_floor: float) -> float:
        del effective_floor
        msg = "floorlet pricing requires OptionletVolatilityStructure (deferred carve-out)"
        raise LibraryException(msg)


# -----------------------------------------------------------------------
# BlackIborCouponPricer — Black-vol pricer (without cap/floor handling)
# -----------------------------------------------------------------------


class BlackIborCouponPricer(IborCouponPricer):
    """Black-formula IBOR coupon pricer (cap/floor handling deferred).

    C++ parity: ql/cashflows/couponpricer.hpp:111-146.

    Without an OptionletVolatilityStructure (deferred carve-out), the
    BlackIborCouponPricer behaves identically to IborCouponPricer for
    plain swaplets — it differs from the base only by the (would-be)
    cap/floor pricing surface. We expose the class so callers can pass
    BlackIborCouponPricer() through the pricer slot, matching the C++
    pattern, but cap/floor methods raise as in the base class.
    """


# -----------------------------------------------------------------------
# Leg-wide pricer attach helper
# -----------------------------------------------------------------------


def set_coupon_pricer(leg: Iterable[CashFlow], pricer: CouponPricer) -> None:
    """Attach a single pricer to every floating coupon in ``leg``.

    C++ parity: ql/cashflows/couponpricer.cpp ``setCouponPricer(leg, pricer)``.
    Non-FloatingRateCoupon entries are skipped silently.
    """
    # Local import to avoid a cycle with floating_rate_coupon.py.
    from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon  # noqa: PLC0415

    for cf in leg:
        if isinstance(cf, FloatingRateCoupon):
            cf.set_pricer(pricer)
