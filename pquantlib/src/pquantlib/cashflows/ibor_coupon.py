"""IborCoupon — coupon paying a Libor-type index.

# C++ parity: ql/cashflows/iborcoupon.hpp + .cpp (v1.42.1).

Simplifications vs C++:
- ``IborCoupon::Settings`` global toggle (par vs indexed coupons) is NOT
  ported — coupons here behave as "indexed coupons" (forecast the
  fixing for the actual accrual period, not for the index's natural
  tenor period). This matches QL_USE_INDEXED_COUPON=OFF behaviour,
  which is the C++ default.
- ``fixingValueDate`` / ``fixingMaturityDate`` / ``fixingEndDate`` /
  ``spanningTime`` cached-data accessors used by the C++ pricer are
  not exposed (no pricer in this port consults them).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date

if TYPE_CHECKING:
    from pquantlib.termstructures.protocols import IborIndexProtocol


class IborCoupon(FloatingRateCoupon):
    """Coupon paying a Libor-type index fixing.

    Inherits all behaviour from FloatingRateCoupon; the type narrows the
    index slot to ``IborIndexProtocol`` (strictly: the C++ ``IborIndex``).
    """

    def __init__(
        self,
        payment_date: Date,
        nominal: float,
        accrual_start_date: Date,
        accrual_end_date: Date,
        fixing_days: int,
        index: IborIndexProtocol,
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
        # Cache the fixing date (C++ computes once in ctor).
        self._fixing_date_cached: Date = super().fixing_date()

    def ibor_index(self) -> IborIndexProtocol:
        """Narrowed accessor returning the IborIndexProtocol-typed index.

        C++ parity: ql/cashflows/iborcoupon.hpp:59 ``iborIndex() const``.
        """
        # Already narrowed at construction via the typed parameter.
        return self._index  # type: ignore[return-value]

    def fixing_date(self) -> Date:
        """Return the cached fixing date (computed once at construction).

        C++ parity: ql/cashflows/iborcoupon.cpp:89-91.
        """
        return self._fixing_date_cached
