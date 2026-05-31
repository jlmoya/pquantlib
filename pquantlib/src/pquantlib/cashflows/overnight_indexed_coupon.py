"""OvernightIndexedCoupon — daily-compounded overnight rate coupon.

# C++ parity: ql/cashflows/overnightindexedcoupon.hpp + .cpp (v1.42.1).

Simplified port:
- ``RateAveraging`` enum (Compound vs Simple): only Compound is
  implemented inline here. Simple averaging is a deferred carve-out
  (no use case in L2-D test surface).
- Lookback days / lockout days / observation shift / compound-spread-daily
  flags: all deferred — set to zero / False / not exposed.
- ``telescopicValueDates`` optimization (C++ uses it to avoid building
  the full daily-fixing schedule when ``QL_USE_SHARED_PTR_FROM_BOOST`` is
  on): NOT ported. We always build the full series of business-day
  value dates in the accrual window. For real-world settlement
  schedules of ~30-90 days this is negligible.

Algorithm (mirrors ql/cashflows/overnightindexedcouponpricer.cpp:121-198
``CompoundingOvernightIndexedCouponPricer::compute``):

    compound_factor = 1.0
    for i in range(n):                            # n = number of fixings
        fixing = index.fixing(fixing_dates[i])
        span   = day_counter.year_fraction(interest_dates[i], interest_dates[i+1])
        compound_factor *= (1.0 + fixing * span)
    average_rate = (compound_factor - 1.0) / accrual_period

The coupon's ``rate()`` and ``amount()`` are then:
    rate   = gearing * average_rate + spread
    amount = nominal * rate * accrual_period

This is wired in via a built-in pricer (``CompoundingOvernightIndexedCouponPricer``)
attached at coupon construction. Callers may override via ``set_pricer``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
from pquantlib.cashflows.overnight_indexed_coupon_pricer import (
    CompoundingOvernightIndexedCouponPricer,
)
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.time.date import Date
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.termstructures.protocols import OvernightIndexProtocol


class OvernightIndexedCoupon(FloatingRateCoupon):
    """Coupon paying daily-compounded overnight rates over [start, end].

    Constructor builds the full series of business-day value dates by
    walking the index's fixing calendar one day at a time.
    """

    def __init__(
        self,
        payment_date: Date,
        nominal: float,
        accrual_start_date: Date,
        accrual_end_date: Date,
        overnight_index: OvernightIndexProtocol,
        gearing: float = 1.0,
        spread: float = 0.0,
        ref_period_start: Date | None = None,
        ref_period_end: Date | None = None,
        day_counter: DayCounter | None = None,
    ) -> None:
        qassert.require(
            payment_date >= accrual_end_date,
            "Payment date cannot be earlier than accrual end date",
        )
        # FloatingRateCoupon expects fixing_days; for plain OIS it's 0
        # (the fixing happens on the same business day as the value date).
        super().__init__(
            payment_date,
            nominal,
            accrual_start_date,
            accrual_end_date,
            0,  # fixing_days
            overnight_index,
            gearing,
            spread,
            ref_period_start,
            ref_period_end,
            day_counter,
            False,  # is_in_arrears
        )
        # Build value dates by walking the fixing calendar one business
        # day at a time from start to end inclusive. C++ uses MakeSchedule
        # with a 1-day tenor backwards; the result is the same for plain
        # daily generation when no lookback / lockout is active.
        cal = overnight_index.fixing_calendar()
        # Adjust the start date forward to a business day if needed.
        value_dates: list[Date] = []
        d = cal.adjust(accrual_start_date)
        end_adj = cal.adjust(accrual_end_date)
        while d <= end_adj:
            value_dates.append(d)
            d = cal.advance(d, 1, TimeUnit.Days)
        qassert.require(len(value_dates) >= 2, "degenerate schedule (fewer than 2 fixings)")
        self._value_dates: list[Date] = value_dates
        self._fixing_dates: list[Date] = value_dates[:-1]
        self._interest_dates: list[Date] = list(value_dates)
        self._n: int = len(value_dates) - 1
        # dt = year fractions between consecutive interest dates
        dc = overnight_index.day_counter()
        self._dt: list[float] = [
            dc.year_fraction(value_dates[i], value_dates[i + 1]) for i in range(self._n)
        ]
        # Attach default pricer.
        self.set_pricer(CompoundingOvernightIndexedCouponPricer())

    # --- inspectors ----------------------------------------------------

    def value_dates(self) -> list[Date]:
        return list(self._value_dates)

    def fixing_dates(self) -> list[Date]:
        return list(self._fixing_dates)

    def interest_dates(self) -> list[Date]:
        return list(self._interest_dates)

    def dt(self) -> list[float]:
        return list(self._dt)

    def n(self) -> int:
        return self._n

    def overnight_index(self) -> OvernightIndexProtocol:
        return self._index  # type: ignore[return-value]


# The default pricer (CompoundingOvernightIndexedCouponPricer) and the
# arithmetic-average / base variants now live in
# ``overnight_indexed_coupon_pricer`` (the full-fidelity port with
# past-fixing handling). It is imported above and attached in __init__.
# It is re-exported here for backwards-compatible imports.
__all__ = [
    "CompoundingOvernightIndexedCouponPricer",
    "OvernightIndexedCoupon",
]
