"""CappedFlooredCoupon family — floating coupons wrapped with a cap/floor.

# C++ parity: ql/cashflows/capflooredcoupon.{hpp,cpp} (v1.42.1, 099987f0) +
# ql/cashflows/overnightindexedcoupon.{hpp,cpp} (the overnight variant).

A :class:`CappedFlooredCoupon` wraps a
:class:`~pquantlib.cashflows.floating_rate_coupon.FloatingRateCoupon` and
caps and/or floors its rate. The payoff of a capped coupon is
``N * T * min(aL + b, C)``; floored is ``N * T * max(aL + b, F)``; collared
combines both. Internally::

    rate = swapletRate + floorletRate - capletRate

with the caplet/floorlet rates supplied by the underlying coupon's pricer
(a vol-aware :class:`~pquantlib.cashflows.coupon_pricer.BlackIborCouponPricer`
for the Ibor case — W12-B upgraded that pricer to price optionlets).

Concrete variants build their own underlying:

* :class:`CappedFlooredIborCoupon` — IborCoupon underlying.
* :class:`CappedFlooredCmsCoupon` — CmsCoupon underlying (W12-A).
* :class:`CappedFlooredOvernightIndexedCoupon` — OvernightIndexedCoupon
  underlying (W12-C).

# C++ parity divergences:
# - The C++ ``LazyObject`` deferred-calculation cache (``performCalculations`` /
#   ``deepUpdate``) is not ported; ``rate()`` recomputes on each call (matching
#   FloatingRateCoupon's port). Observer wiring (``registerWith(underlying_)``)
#   is omitted — consistent with the rest of the cashflows port.
# - ``accept`` / ``AcyclicVisitor`` Visitability is not ported.
# - The overnight variant's ``dailyCapFloor`` / ``compoundSpreadDaily`` /
#   effective-volatility machinery (which needs ``BlackOvernightIndexedCouponPricer``,
#   not in this port) is a documented carve-out — the non-daily / non-compound
#   ``effective_cap`` / ``effective_floor`` path is ported, and ``rate()``
#   delegates cap/floor to the underlying pricer (so the uncapped path works;
#   the capped path works once a vol-aware overnight pricer is wired).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
from pquantlib.time.business_day_convention import BusinessDayConvention

if TYPE_CHECKING:
    from pquantlib.cashflows.coupon_pricer import FloatingRateCouponPricer
    from pquantlib.cashflows.overnight_indexed_coupon import OvernightIndexedCoupon
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.indexes.swap_index import SwapIndex
    from pquantlib.termstructures.protocols import IborIndexProtocol
    from pquantlib.time.date import Date


class CappedFlooredCoupon(FloatingRateCoupon):
    """Floating coupon wrapped with a cap and/or floor.

    # C++ parity: ql/cashflows/capflooredcoupon.hpp:57-101 + .cpp:27-145.
    """

    def __init__(
        self,
        underlying: FloatingRateCoupon,
        cap: float | None = None,
        floor: float | None = None,
    ) -> None:
        # Copy every field off the underlying (C++ capflooredcoupon.cpp:30-43).
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
            underlying.ex_coupon_date(),
            underlying.fixing_convention(),
        )
        self._underlying: FloatingRateCoupon = underlying
        self._is_capped: bool = False
        self._is_floored: bool = False
        self._cap: float = 0.0
        self._floor: float = 0.0
        # Sign-aware cap/floor storage (C++ capflooredcoupon.cpp:46-70).
        if self._gearing > 0.0:
            if cap is not None:
                self._is_capped = True
                self._cap = cap
            if floor is not None:
                self._floor = floor
                self._is_floored = True
        else:
            if cap is not None:
                self._floor = cap
                self._is_floored = True
            if floor is not None:
                self._is_capped = True
                self._cap = floor
        if self._is_capped and self._is_floored:
            assert cap is not None
            assert floor is not None
            qassert.require(
                cap >= floor,
                f"cap level ({cap}) less than floor level ({floor})",
            )

    # --- inspectors ----------------------------------------------------

    def underlying(self) -> FloatingRateCoupon:
        """# C++ parity: ``CappedFlooredCoupon::underlying``."""
        return self._underlying

    def is_capped(self) -> bool:
        """# C++ parity: ``CappedFlooredCoupon::isCapped`` (inline)."""
        return self._is_capped

    def is_floored(self) -> bool:
        """# C++ parity: ``CappedFlooredCoupon::isFloored`` (inline)."""
        return self._is_floored

    def cap(self) -> float | None:
        """User-visible cap (sign-aware).

        # C++ parity: ql/cashflows/capflooredcoupon.cpp:107-113.
        """
        if self._gearing > 0.0 and self._is_capped:
            return self._cap
        if self._gearing < 0.0 and self._is_floored:
            return self._floor
        return None

    def floor(self) -> float | None:
        """User-visible floor (sign-aware).

        # C++ parity: ql/cashflows/capflooredcoupon.cpp:115-121.
        """
        if self._gearing > 0.0 and self._is_floored:
            return self._floor
        if self._gearing < 0.0 and self._is_capped:
            return self._cap
        return None

    def effective_cap(self) -> float | None:
        """``(cap - spread) / gearing`` (or ``None`` if uncapped).

        # C++ parity: ql/cashflows/capflooredcoupon.cpp:123-128.
        """
        if self._is_capped:
            return (self._cap - self.spread()) / self.gearing()
        return None

    def effective_floor(self) -> float | None:
        """``(floor - spread) / gearing`` (or ``None`` if unfloored).

        # C++ parity: ql/cashflows/capflooredcoupon.cpp:130-135.
        """
        if self._is_floored:
            return (self._floor - self.spread()) / self.gearing()
        return None

    # --- pricer wiring -------------------------------------------------

    def set_pricer(self, pricer: FloatingRateCouponPricer | None) -> None:
        """Attach the pricer to both this coupon and the underlying.

        # C++ parity: ql/cashflows/capflooredcoupon.cpp:75-79.
        """
        super().set_pricer(pricer)
        self._underlying.set_pricer(pricer)

    # --- Coupon interface ----------------------------------------------

    def rate(self) -> float:
        """``swapletRate + floorletRate - capletRate``.

        # C++ parity: ql/cashflows/capflooredcoupon.cpp:86-101
        # (performCalculations + rate).
        """
        underlying_pricer = self._underlying.pricer()
        qassert.require(underlying_pricer is not None, "pricer not set")
        assert underlying_pricer is not None
        swaplet_rate = self._underlying.rate()
        # The underlying's pricer was just initialized by ``underlying.rate()``.
        floorlet_rate = 0.0
        if self._is_floored:
            eff_floor = self.effective_floor()
            assert eff_floor is not None
            floorlet_rate = underlying_pricer.floorlet_rate(eff_floor)
        caplet_rate = 0.0
        if self._is_capped:
            eff_cap = self.effective_cap()
            assert eff_cap is not None
            caplet_rate = underlying_pricer.caplet_rate(eff_cap)
        return swaplet_rate + floorlet_rate - caplet_rate

    def convexity_adjustment(self) -> float:
        """# C++ parity: ql/cashflows/capflooredcoupon.cpp:103-105."""
        return self._underlying.convexity_adjustment()


class CappedFlooredIborCoupon(CappedFlooredCoupon):
    """Capped/floored Ibor coupon.

    # C++ parity: ql/cashflows/capflooredcoupon.hpp:103-135 — builds an
    # ``IborCoupon`` underlying then wraps it with cap/floor.
    """

    def __init__(
        self,
        payment_date: Date,
        nominal: float,
        start_date: Date,
        end_date: Date,
        fixing_days: int,
        index: IborIndexProtocol,
        gearing: float = 1.0,
        spread: float = 0.0,
        cap: float | None = None,
        floor: float | None = None,
        ref_period_start: Date | None = None,
        ref_period_end: Date | None = None,
        day_counter: DayCounter | None = None,
        is_in_arrears: bool = False,
        ex_coupon_date: Date | None = None,
        fixing_convention: BusinessDayConvention = BusinessDayConvention.Preceding,
    ) -> None:
        from pquantlib.cashflows.ibor_coupon import IborCoupon  # noqa: PLC0415

        underlying = IborCoupon(
            payment_date,
            nominal,
            start_date,
            end_date,
            fixing_days,
            index,
            gearing,
            spread,
            ref_period_start,
            ref_period_end,
            day_counter,
            is_in_arrears,
            ex_coupon_date,
            fixing_convention,
        )
        super().__init__(underlying, cap, floor)


class CappedFlooredCmsCoupon(CappedFlooredCoupon):
    """Capped/floored CMS coupon.

    # C++ parity: ql/cashflows/capflooredcoupon.hpp:137-169 — builds a
    # ``CmsCoupon`` underlying (W12-A) then wraps it with cap/floor.

    .. note:: cap/floor pricing of CMS coupons requires a CMS optionlet pricer
       (a Hagan-replication caplet/floorlet), which the
       :class:`~pquantlib.cashflows.cms_coupon_pricer.CmsCouponPricer` base does
       not provide. The uncapped path (cap=floor=None) prices via the Hagan CMS
       swaplet rate; the capped path needs a pricer implementing
       ``caplet_rate`` / ``floorlet_rate``.
    """

    def __init__(
        self,
        payment_date: Date,
        nominal: float,
        start_date: Date,
        end_date: Date,
        fixing_days: int,
        swap_index: SwapIndex,
        gearing: float = 1.0,
        spread: float = 0.0,
        cap: float | None = None,
        floor: float | None = None,
        ref_period_start: Date | None = None,
        ref_period_end: Date | None = None,
        day_counter: DayCounter | None = None,
        is_in_arrears: bool = False,
        ex_coupon_date: Date | None = None,
        fixing_convention: BusinessDayConvention = BusinessDayConvention.Preceding,
    ) -> None:
        from pquantlib.cashflows.cms_coupon import CmsCoupon  # noqa: PLC0415

        underlying = CmsCoupon(
            payment_date,
            nominal,
            start_date,
            end_date,
            fixing_days,
            swap_index,
            gearing,
            spread,
            ref_period_start,
            ref_period_end,
            day_counter,
            is_in_arrears,
            ex_coupon_date,
            fixing_convention,
        )
        super().__init__(underlying, cap, floor)


class CappedFlooredOvernightIndexedCoupon(CappedFlooredCoupon):
    """Capped/floored compounded overnight-indexed coupon.

    # C++ parity: ql/cashflows/overnightindexedcoupon.hpp:144-208 + .cpp:274-396.

    Wraps an :class:`~pquantlib.cashflows.overnight_indexed_coupon.OvernightIndexedCoupon`.
    Unlike the base ``CappedFlooredCoupon`` it supports a ``naked_option`` flag
    (strip the swaplet, keep only the optionality).

    # C++ parity divergence: ``daily_cap_floor`` and ``compound_spread_daily``
    # are NOT ported (they require ``BlackOvernightIndexedCouponPricer`` +
    # ``OvernightIndexedCoupon.effectiveSpread`` — both deferred). The
    # non-daily ``effective_cap``/``effective_floor`` = ``(cap-spread)/gearing``
    # path is ported. Cap/floor pricing delegates to the underlying pricer's
    # ``caplet_rate``/``floorlet_rate``; the uncapped path always works.
    """

    def __init__(
        self,
        underlying: OvernightIndexedCoupon,
        cap: float | None = None,
        floor: float | None = None,
        naked_option: bool = False,
    ) -> None:
        super().__init__(underlying, cap, floor)
        self._naked_option: bool = naked_option

    def naked_option(self) -> bool:
        """# C++ parity: ``CappedFlooredOvernightIndexedCoupon::nakedOption``."""
        return self._naked_option

    def rate(self) -> float:
        """``swapletRate + floorletRate - capletRate`` with naked-option support.

        # C++ parity: ql/cashflows/overnightindexedcoupon.cpp:317-335.

        When ``naked_option`` is set the swaplet rate is dropped; and a
        cap-only naked option flips the caplet sign (so the result is a long
        call rather than a short cap).
        """
        underlying_pricer = self._underlying.pricer()
        qassert.require(underlying_pricer is not None, "underlying coupon pricer not set")
        assert underlying_pricer is not None
        swaplet_rate = 0.0 if self._naked_option else self._underlying.rate()
        if self._naked_option:
            # Still need the pricer initialized against the underlying.
            underlying_pricer.initialize(self._underlying)
        floorlet_rate = 0.0
        if self._is_floored:
            eff_floor = self.effective_floor()
            assert eff_floor is not None
            floorlet_rate = underlying_pricer.floorlet_rate(eff_floor)
        caplet_rate = 0.0
        if self._is_capped:
            eff_cap = self.effective_cap()
            assert eff_cap is not None
            sign = -1.0 if (self._naked_option and not self._is_floored) else 1.0
            caplet_rate = sign * underlying_pricer.caplet_rate(eff_cap)
        return swaplet_rate + floorlet_rate - caplet_rate


__all__ = [
    "CappedFlooredCmsCoupon",
    "CappedFlooredCoupon",
    "CappedFlooredIborCoupon",
    "CappedFlooredOvernightIndexedCoupon",
]
