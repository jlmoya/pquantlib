"""FloatingRateCoupon — coupon paying an index-based rate.

# C++ parity: ql/cashflows/floatingratecoupon.hpp + .cpp (v1.42.1).

The C++ class binds an InterestRateIndex pointer plus fixingDays /
gearing / spread / isInArrears / fixingConvention; ``rate()`` delegates
to a CouponPricer set via ``setPricer``.

Python port:
- ``index`` is typed as ``IborIndexProtocol`` (cross-cluster Protocol).
  Concrete OvernightIndexedCoupon narrows this to ``OvernightIndexProtocol``.
  We use a runtime check + a Union of the two Protocols to keep typing
  loose enough to accommodate both.
- ``rate()`` is implemented to use ``pricer.swaplet_rate()`` (set via
  ``set_pricer``); the C++ ``performCalculations()`` deferred-execution
  cache is replaced by recomputing on every call (Python perf is
  adequate at L2-D scale).
- ``adjusted_fixing``, ``convexity_adjustment`` are concrete here mirroring
  the inline definitions in ql/cashflows/floatingratecoupon.hpp:132-148.
- The C++ ``Settings.evaluationDate()`` observer registration is
  omitted (no global eval-date state in this port).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.coupon import Coupon
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.cashflows.coupon_pricer import FloatingRateCouponPricer
    from pquantlib.termstructures.protocols import (
        IborIndexProtocol,
        OvernightIndexProtocol,
    )


class FloatingRateCoupon(Coupon):
    """Base class for coupons paying an index-derived rate.

    Concrete subclasses (IborCoupon, OvernightIndexedCoupon) override
    ``index_fixing()`` and may set a different pricer.
    """

    def __init__(
        self,
        payment_date: Date,
        nominal: float,
        accrual_start_date: Date,
        accrual_end_date: Date,
        fixing_days: int,
        index: IborIndexProtocol | OvernightIndexProtocol,
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
            accrual_start_date,
            accrual_end_date,
            ref_period_start,
            ref_period_end,
            ex_coupon_date,
        )
        qassert.require(gearing != 0.0, "Null gearing not allowed")
        self._index: IborIndexProtocol | OvernightIndexProtocol = index
        self._day_counter: DayCounter = (
            day_counter if day_counter is not None else index.day_counter()
        )
        self._fixing_days: int = fixing_days
        self._gearing: float = gearing
        self._spread: float = spread
        self._is_in_arrears: bool = is_in_arrears
        self._fixing_convention: BusinessDayConvention = fixing_convention
        self._pricer: FloatingRateCouponPricer | None = None

    # --- inspectors ----------------------------------------------------

    def index(self) -> IborIndexProtocol | OvernightIndexProtocol:
        return self._index

    def fixing_days(self) -> int:
        return self._fixing_days

    def gearing(self) -> float:
        return self._gearing

    def spread(self) -> float:
        return self._spread

    def is_in_arrears(self) -> bool:
        return self._is_in_arrears

    def fixing_convention(self) -> BusinessDayConvention:
        return self._fixing_convention

    def day_counter(self) -> DayCounter:
        return self._day_counter

    def pricer(self) -> FloatingRateCouponPricer | None:
        return self._pricer

    # --- index / fixing dates ------------------------------------------

    def fixing_date(self) -> Date:
        """Fixing date (start - fixingDays for in-advance, end - fixingDays for in-arrears).

        C++ parity: ql/cashflows/floatingratecoupon.cpp:81-86.
        """
        ref_date = self._accrual_end_date if self._is_in_arrears else self._accrual_start_date
        return self._index.fixing_calendar().advance(
            ref_date,
            -self._fixing_days,
            TimeUnit.Days,
            self._fixing_convention,
            False,
        )

    def index_fixing(self) -> float:
        """Underlying-index fixing for this coupon.

        C++ parity: ql/cashflows/floatingratecoupon.cpp:103-105.
        """
        return self._index.fixing(self.fixing_date(), False)

    # --- pricer wiring -------------------------------------------------

    def set_pricer(self, pricer: FloatingRateCouponPricer | None) -> None:
        """Attach (or detach) a coupon pricer.

        C++ parity: ql/cashflows/floatingratecoupon.cpp:62-70.
        """
        self._pricer = pricer

    # --- Coupon interface ----------------------------------------------

    def rate(self) -> float:
        """Floating rate as set by the pricer.

        C++ parity: ql/cashflows/floatingratecoupon.cpp:88-91.
        """
        qassert.require(self._pricer is not None, "pricer not set")
        assert self._pricer is not None
        self._pricer.initialize(self)
        return self._pricer.swaplet_rate()

    def amount(self) -> float:
        """amount = nominal * rate * accrual_period.

        C++ parity: ql/cashflows/floatingratecoupon.hpp:69 (inline).
        """
        return self._nominal * self.rate() * self.accrual_period()

    def accrued_amount(self, d: Date) -> float:
        """nominal * rate * accrued_period(d), zeroed outside the accrual window.

        C++ parity: ql/cashflows/floatingratecoupon.cpp:72-79.
        """
        if d <= self._accrual_start_date or d > self._payment_date:
            return 0.0
        return self._nominal * self.rate() * self.accrued_period(d)

    # --- derived ------------------------------------------------------

    def convexity_adjustment(self) -> float:
        """Default convexity adjustment via the pricer.

        C++ parity: ql/cashflows/floatingratecoupon.hpp:132-134.
        """
        return self._convexity_adjustment_impl(self.index_fixing())

    def _convexity_adjustment_impl(self, fixing: float) -> float:
        """gearing == 0 ? 0 : (adjusted_fixing - fixing).

        C++ parity: ql/cashflows/floatingratecoupon.hpp:145-148.
        """
        if self._gearing == 0.0:
            return 0.0
        return self.adjusted_fixing() - fixing

    def adjusted_fixing(self) -> float:
        """(rate - spread) / gearing.

        C++ parity: ql/cashflows/floatingratecoupon.hpp:136-138.
        """
        return (self.rate() - self._spread) / self._gearing
