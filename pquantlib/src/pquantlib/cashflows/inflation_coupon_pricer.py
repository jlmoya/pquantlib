"""InflationCouponPricer — abstract base for inflation-coupon pricers.

# C++ parity: ql/cashflows/inflationcouponpricer.{hpp,cpp} (v1.42.1).

The C++ class is an abstract pricer with cap / floor / swaplet methods
and an ``initialize(InflationCoupon)`` hook. Concrete subclasses live
in:

* ``cpi_coupon_pricer.CPICouponPricer`` (swaplet + cap/floor through
  CPI volatility surface — vol path stubbed pending L7-D).
* ``yoy_inflation_coupon_pricer.YoYInflationCouponPricer`` (analogous,
  YoY).

Python divergences from C++:

- The C++ pricer derives from ``Observer + Observable``. We mirror with
  ``Observable`` (matches the L2-D ``CouponPricer`` choice — observer
  registrations against vol-curve handles are not needed until the
  vol-dependent code paths land in L7-D).
- ``setCouponPricer(Leg&, shared_ptr<InflationCouponPricer>)`` is
  ported as a free function on the module (``set_coupon_pricer``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pquantlib.patterns.observer import Observable
from pquantlib.time.date import Date

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pquantlib.cashflows.cash_flow import CashFlow
    from pquantlib.cashflows.inflation_coupon import InflationCoupon


class InflationCouponPricer(ABC, Observable):
    """Abstract pricer for ``InflationCoupon`` subtypes.

    # C++ parity: ``InflationCouponPricer`` in inflationcouponpricer.hpp.
    """

    def __init__(self) -> None:
        super().__init__()
        # C++ parity: protected ``Date paymentDate_`` member used by
        # subclasses' discount lookups (cpicouponpricer.cpp:113 etc.).
        self._payment_date: Date = Date()

    @abstractmethod
    def swaplet_price(self) -> float:
        """Discounted swaplet (no cap/floor) price."""

    @abstractmethod
    def swaplet_rate(self) -> float:
        """Swaplet rate (undiscounted)."""

    @abstractmethod
    def caplet_price(self, effective_cap: float) -> float:
        """Discounted caplet price at the effective cap level."""

    @abstractmethod
    def caplet_rate(self, effective_cap: float) -> float:
        """Caplet rate (undiscounted) at the effective cap level."""

    @abstractmethod
    def floorlet_price(self, effective_floor: float) -> float:
        """Discounted floorlet price at the effective floor level."""

    @abstractmethod
    def floorlet_rate(self, effective_floor: float) -> float:
        """Floorlet rate (undiscounted) at the effective floor level."""

    @abstractmethod
    def initialize(self, coupon: InflationCoupon) -> None:
        """Capture per-coupon state needed by the rate / price methods."""

    # Observer.update() — propagate to observers (mirrors C++).
    def update(self) -> None:
        self.notify_observers()


def set_coupon_pricer(leg: Iterable[CashFlow], pricer: InflationCouponPricer) -> None:
    """Attach an inflation pricer to every ``InflationCoupon`` in ``leg``.

    # C++ parity: ql/cashflows/inflationcouponpricer.cpp:27-34 — the
    # function silently skips non-inflation entries.
    """
    # Local import to avoid the coupon ↔ pricer cycle at module load.
    from pquantlib.cashflows.inflation_coupon import InflationCoupon  # noqa: PLC0415

    for cf in leg:
        if isinstance(cf, InflationCoupon):
            cf.set_pricer(pricer)
