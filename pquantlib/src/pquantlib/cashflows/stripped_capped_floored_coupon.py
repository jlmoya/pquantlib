"""StrippedCappedFlooredCoupon — extract the embedded optionality value.

# C++ parity: ql/experimental/coupons/strippedcapflooredcoupon.{hpp,cpp}
# (v1.42.1, 099987f0).

Given a :class:`~pquantlib.cashflows.capped_floored_coupon.CappedFlooredCoupon`,
this coupon's ``rate()`` returns *only* the value of the embedded cap/floor —
the swaplet rate is stripped out. For a collar the value is
``floorletRate - capletRate`` (a long floor + short cap); for a single cap or
floor it is the long-option value (``floorletRate`` or ``capletRate``).

# C++ parity divergences: LazyObject cache + Observer wiring + Visitability
# (``accept``) are not ported (consistent with the cashflows port).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.capped_floored_coupon import CappedFlooredCoupon
from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pquantlib.cashflows.cash_flow import CashFlow
    from pquantlib.cashflows.coupon_pricer import FloatingRateCouponPricer


class StrippedCappedFlooredCoupon(FloatingRateCoupon):
    """The embedded cap/floor optionality of a ``CappedFlooredCoupon``.

    # C++ parity: ql/experimental/coupons/strippedcapflooredcoupon.hpp:31-72 +
    # .cpp:26-113.
    """

    def __init__(self, underlying: CappedFlooredCoupon) -> None:
        super().__init__(
            underlying.date(),
            underlying.nominal(),
            underlying.accrual_start_date(),
            underlying.accrual_end_date(),
            underlying.fixing_days(),
            underlying.index(),
            underlying.gearing(),
            underlying.spread(),
            underlying.reference_period_start(),
            underlying.reference_period_end(),
            underlying.day_counter(),
            underlying.is_in_arrears(),
        )
        self._underlying: CappedFlooredCoupon = underlying

    # --- inspectors ----------------------------------------------------

    def underlying(self) -> CappedFlooredCoupon:
        """# C++ parity: ``StrippedCappedFlooredCoupon::underlying``."""
        return self._underlying

    def is_cap(self) -> bool:
        """# C++ parity: ql/experimental/coupons/strippedcapflooredcoupon.cpp:97-99."""
        return self._underlying.is_capped()

    def is_floor(self) -> bool:
        """# C++ parity: ql/experimental/coupons/strippedcapflooredcoupon.cpp:101-103."""
        return self._underlying.is_floored()

    def is_collar(self) -> bool:
        """# C++ parity: ql/experimental/coupons/strippedcapflooredcoupon.cpp:105-107."""
        return self.is_cap() and self.is_floor()

    def cap(self) -> float | None:
        """# C++ parity: strippedcapflooredcoupon.cpp:74."""
        return self._underlying.cap()

    def floor(self) -> float | None:
        """# C++ parity: strippedcapflooredcoupon.cpp:76-78."""
        return self._underlying.floor()

    def effective_cap(self) -> float | None:
        """# C++ parity: strippedcapflooredcoupon.cpp:80-82."""
        return self._underlying.effective_cap()

    def effective_floor(self) -> float | None:
        """# C++ parity: strippedcapflooredcoupon.cpp:84-86."""
        return self._underlying.effective_floor()

    # --- pricer wiring -------------------------------------------------

    def set_pricer(self, pricer: FloatingRateCouponPricer | None) -> None:
        """# C++ parity: strippedcapflooredcoupon.cpp:109-113."""
        super().set_pricer(pricer)
        self._underlying.set_pricer(pricer)

    # --- Coupon interface ----------------------------------------------

    def rate(self) -> float:
        """Stripped cap/floor value.

        # C++ parity: ql/experimental/coupons/strippedcapflooredcoupon.cpp:45-63.

        Collar → ``floorletRate - capletRate``; single cap/floor →
        ``floorletRate + capletRate`` (one term is zero).
        """
        inner = self._underlying.underlying()
        pricer = inner.pricer()
        qassert.require(pricer is not None, "pricer not set")
        assert pricer is not None
        pricer.initialize(inner)
        floorlet_rate = 0.0
        if self._underlying.is_floored():
            eff_floor = self._underlying.effective_floor()
            assert eff_floor is not None
            floorlet_rate = pricer.floorlet_rate(eff_floor)
        caplet_rate = 0.0
        if self._underlying.is_capped():
            eff_cap = self._underlying.effective_cap()
            assert eff_cap is not None
            caplet_rate = pricer.caplet_rate(eff_cap)
        if self._underlying.is_floored() and self._underlying.is_capped():
            return floorlet_rate - caplet_rate
        return floorlet_rate + caplet_rate

    def convexity_adjustment(self) -> float:
        """# C++ parity: strippedcapflooredcoupon.cpp:70-72."""
        return self._underlying.convexity_adjustment()


def stripped_capped_floored_coupon_leg(underlying_leg: Sequence[CashFlow]) -> list[CashFlow]:
    """Wrap every ``CappedFlooredCoupon`` in ``underlying_leg`` with a stripper.

    # C++ parity: ql/experimental/coupons/strippedcapflooredcoupon.cpp:115-131
    # (``StrippedCappedFlooredCouponLeg::operator Leg``). Non-CappedFloored
    # entries pass through unchanged.
    """
    result: list[CashFlow] = []
    for cf in underlying_leg:
        if isinstance(cf, CappedFlooredCoupon):
            result.append(StrippedCappedFlooredCoupon(cf))
        else:
            result.append(cf)
    return result


__all__ = ["StrippedCappedFlooredCoupon", "stripped_capped_floored_coupon_leg"]
