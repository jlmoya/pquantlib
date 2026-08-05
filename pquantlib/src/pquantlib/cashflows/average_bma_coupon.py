"""AverageBMACoupon — coupon paying a weighted average of BMA-index fixings.

# C++ parity: ql/cashflows/averagebmacoupon.{hpp,cpp} (v1.42.1, 099987f0).

The coupon rate is a calendar-day-weighted average of the relevant BMA
(SIFMA) fixings over the interest period. The weighting uses the actual
calendar days for which each fixing is valid and contributing to the period.
Before weights are computed the fixing schedule is adjusted for the index's
one-day fixing gap (see :meth:`AverageBMACouponPricer.swaplet_rate`).

Components:

- :class:`AverageBMACoupon` — the coupon (a ``FloatingRateCoupon`` whose
  ``fixing_date()`` / ``index_fixing()`` raise; the multiple fixings are
  exposed via ``fixing_dates()`` / ``index_fixings()``).
- :class:`AverageBMACouponPricer` — the built-in pricer computing the
  weighted-average rate (attached at construction).
- :func:`average_bma_leg` — builds a leg of average-BMA coupons from a
  schedule (the C++ ``AverageBMALeg`` fluent builder collapsed into a
  keyword-argument function, matching this port's ``ibor_leg`` /
  ``overnight_leg`` convention).

Python divergences from C++:

- ``fixingDate()`` / ``indexFixing()`` / ``convexityAdjustment()`` raise
  (C++ ``QL_FAIL``) because a single fixing is undefined for this coupon.
- ``accept(AcyclicVisitor&)`` is omitted (Visitor carve-out).
- The ``AverageBMALeg`` builder class is replaced by ``average_bma_leg(...)``.
- ``bmaCutoffDays`` is 0 (the C++ "to be verified" placeholder), matching C++.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows import cash_flow_vectors as cfv
from pquantlib.cashflows.coupon_pricer import FloatingRateCouponPricer
from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
from pquantlib.exceptions import LibraryException
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.cashflows.cash_flow import CashFlow
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.indexes.bma_index import BMAIndex
    from pquantlib.time.schedule import Schedule

# C++ parity: averagebmacoupon.cpp:30 — ``Integer bmaCutoffDays = 0;``.
_BMA_CUTOFF_DAYS = 0


def _adjust_to_previous_valid_fixing_date(d: Date, index: BMAIndex) -> Date:
    """Roll ``d`` back to the nearest earlier valid BMA fixing date.

    # C++ parity: anonymous ``adjustToPreviousValidFixingDate``
    (averagebmacoupon.cpp:96-99).
    """
    while not index.is_valid_fixing_date(d) and d > Date.min_date():
        d = d - 1
    return d


class AverageBMACoupon(FloatingRateCoupon):
    """Coupon paying the calendar-day-weighted average of BMA fixings.

    # C++ parity: ``AverageBMACoupon`` in averagebmacoupon.hpp:44-79.
    """

    def __init__(
        self,
        payment_date: Date,
        nominal: float,
        start_date: Date,
        end_date: Date,
        index: BMAIndex,
        gearing: float = 1.0,
        spread: float = 0.0,
        ref_period_start: Date | None = None,
        ref_period_end: Date | None = None,
        day_counter: DayCounter | None = None,
    ) -> None:
        super().__init__(
            payment_date,
            nominal,
            start_date,
            end_date,
            index.fixing_days(),
            index,
            gearing,
            spread,
            ref_period_start,
            ref_period_end,
            day_counter,
            False,  # is_in_arrears
        )
        cal = index.fixing_calendar()
        fixing_days = int(index.fixing_days()) + _BMA_CUTOFF_DAYS
        fixing_start = cal.advance(
            start_date, -fixing_days, TimeUnit.Days, BusinessDayConvention.Preceding
        )
        # make sure that the value date associated to fixingStart is <= startDate
        fixing_start = _adjust_to_previous_valid_fixing_date(fixing_start, index)
        while index.value_date(fixing_start) > start_date and fixing_start > Date.min_date():
            fixing_start = _adjust_to_previous_valid_fixing_date(fixing_start - 1, index)

        self._fixing_schedule: Schedule = index.fixing_schedule(fixing_start, end_date)
        self.set_pricer(AverageBMACouponPricer())

    # --- FloatingRateCoupon interface ----------------------------------

    def fixing_date(self) -> Date:
        """No single fixing date for an average-BMA coupon.

        # C++ parity: ``AverageBMACoupon::fixingDate`` (averagebmacoupon.cpp:132-134).
        """
        msg = "no single fixing date for average-BMA coupon"
        raise LibraryException(msg)

    def fixing_dates(self) -> list[Date]:
        """The fixing dates of the rates to be averaged.

        # C++ parity: ``AverageBMACoupon::fixingDates`` (averagebmacoupon.cpp:136-138).
        """
        s = self._fixing_schedule
        return [s.date(i) for i in range(s.size())]

    def index_fixing(self) -> float:
        """No single fixing for an average-BMA coupon.

        # C++ parity: ``AverageBMACoupon::indexFixing`` (averagebmacoupon.cpp:140-142).
        """
        msg = "no single fixing for average-BMA coupon"
        raise LibraryException(msg)

    def index_fixings(self) -> list[float]:
        """Fixings of the underlying index to be averaged.

        # C++ parity: ``AverageBMACoupon::indexFixings`` (averagebmacoupon.cpp:144-149).
        """
        s = self._fixing_schedule
        return [self._index.fixing(s.date(i), False) for i in range(s.size())]

    def convexity_adjustment(self) -> float:
        """Not defined for an average-BMA coupon.

        # C++ parity: ``AverageBMACoupon::convexityAdjustment`` (averagebmacoupon.cpp:151-153).
        """
        msg = "not defined for average-BMA coupon"
        raise LibraryException(msg)

    def bma_index(self) -> BMAIndex:
        """The underlying BMA index (narrowed return type)."""
        return self._index  # type: ignore[return-value]


class AverageBMACouponPricer(FloatingRateCouponPricer):
    """Calendar-day-weighted-average BMA rate pricer.

    # C++ parity: anonymous ``AverageBMACouponPricer`` (averagebmacoupon.cpp:32-91).
    """

    def __init__(self) -> None:
        super().__init__()
        self._coupon: AverageBMACoupon | None = None

    def initialize(self, coupon: FloatingRateCoupon) -> None:
        qassert.require(isinstance(coupon, AverageBMACoupon), "wrong coupon type")
        assert isinstance(coupon, AverageBMACoupon)
        self._coupon = coupon

    def swaplet_rate(self) -> float:
        """Calendar-day-weighted average of the BMA fixings over the period.

        # C++ parity: ``AverageBMACouponPricer::swapletRate``
        (averagebmacoupon.cpp:38-81). Restricted to ``cutoffDays == 0``
        (the C++ "to be verified" placeholder).
        """
        qassert.require(self._coupon is not None, "coupon not set")
        assert self._coupon is not None
        coupon = self._coupon
        index = coupon.bma_index()
        fixing_dates = coupon.fixing_dates()

        cutoff_days = 0
        start_date = coupon.accrual_start_date() - cutoff_days
        end_date = coupon.accrual_end_date() - cutoff_days
        d1 = start_date
        d2 = start_date

        qassert.require(len(fixing_dates) > 0, "fixing date list empty")
        qassert.require(
            index.value_date(fixing_dates[0]) <= start_date,
            "first fixing date valid after period start",
        )
        qassert.require(
            index.value_date(fixing_dates[-1]) >= end_date,
            "last fixing date valid before period end",
        )

        avg_bma = 0.0
        days = 0
        for i in range(len(fixing_dates) - 1):
            value_date = index.value_date(fixing_dates[i])
            next_value_date = index.value_date(fixing_dates[i + 1])

            if fixing_dates[i] >= end_date or value_date >= end_date:
                break
            if fixing_dates[i + 1] < start_date or next_value_date <= start_date:
                continue

            d2 = min(next_value_date, end_date)
            avg_bma += index.fixing(fixing_dates[i], False) * (d2 - d1)
            days += d2 - d1
            d1 = d2

        avg_bma /= end_date - start_date
        qassert.require(
            days == end_date - start_date,
            f"averaging days {days} differ from interest days {end_date - start_date}",
        )
        return coupon.gearing() * avg_bma + coupon.spread()

    # The remaining FloatingRateCouponPricer interface is not available
    # (C++ QL_FAIL "not available").
    def swaplet_price(self) -> float:
        msg = "not available"
        raise LibraryException(msg)

    def caplet_price(self, effective_cap: float) -> float:
        del effective_cap
        msg = "not available"
        raise LibraryException(msg)

    def caplet_rate(self, effective_cap: float) -> float:
        del effective_cap
        msg = "not available"
        raise LibraryException(msg)

    def floorlet_price(self, effective_floor: float) -> float:
        del effective_floor
        msg = "not available"
        raise LibraryException(msg)

    def floorlet_rate(self, effective_floor: float) -> float:
        del effective_floor
        msg = "not available"
        raise LibraryException(msg)


class AverageBMALeg:
    """Chained builder for a sequence of average-BMA coupons.

    # C++ parity: ``AverageBMALeg`` (averagebmacoupon.hpp:82-101,
    # .cpp:180-251).

    Every C++ ``withXxx`` setter is present as ``with_xxx`` and returns
    ``self``; C++'s ``operator Leg()`` is :meth:`build`. Note that
    ``AverageBMALeg`` has **no** payment-lag, payment-calendar, fixing-days,
    cap/floor or ex-coupon setter: the C++ class is the smallest of the
    family, and its payment date is a plain ``calendar.adjust(end, adj)``.
    """

    def __init__(self, schedule: Schedule, index: BMAIndex) -> None:
        self._schedule: Schedule = schedule
        self._index: BMAIndex = index
        self._notionals: list[float] = []
        self._payment_day_counter: DayCounter | None = None
        self._payment_adjustment: BusinessDayConvention = BusinessDayConvention.Following
        self._gearings: list[float] = []
        self._spreads: list[float] = []

    # --- chained setters -----------------------------------------------

    def with_notionals(self, notionals: float | Sequence[float]) -> AverageBMALeg:
        """# C++ parity: ``withNotionals`` (.cpp:183-192)."""
        self._notionals = cfv.as_float_list(notionals)
        return self

    def with_payment_day_counter(self, day_counter: DayCounter) -> AverageBMALeg:
        """# C++ parity: ``withPaymentDayCounter`` (.cpp:194-198)."""
        self._payment_day_counter = day_counter
        return self

    def with_payment_adjustment(self, convention: BusinessDayConvention) -> AverageBMALeg:
        """# C++ parity: ``withPaymentAdjustment`` (.cpp:200-204)."""
        self._payment_adjustment = convention
        return self

    def with_gearings(self, gearings: float | Sequence[float]) -> AverageBMALeg:
        """# C++ parity: ``withGearings`` (.cpp:206-215)."""
        self._gearings = cfv.as_float_list(gearings)
        return self

    def with_spreads(self, spreads: float | Sequence[float]) -> AverageBMALeg:
        """# C++ parity: ``withSpreads`` (.cpp:217-226)."""
        self._spreads = cfv.as_float_list(spreads)
        return self

    # --- operator Leg() -------------------------------------------------

    def build(self) -> list[CashFlow]:
        """Build the leg.

        # C++ parity: ``AverageBMALeg::operator Leg()`` (.cpp:228-251).

        Note the C++ stub-reference blocks use ``paymentAdjustment_`` rather
        than the schedule's own business-day convention (.cpp:239-245), and
        that the notional falls back to ``notionals_.back()`` rather than to
        ``1.0`` (.cpp:248).
        """
        qassert.require(len(self._notionals) > 0, "no notional given")

        schedule = self._schedule
        # NB: Schedule exposes ``calendar`` / ``tenor`` / ``is_regular`` as
        # properties (no parens) in this port; ``size()`` / ``date(i)`` /
        # ``has_*()`` remain methods.
        calendar = schedule.calendar
        cashflows: list[CashFlow] = []
        n = schedule.size() - 1
        for i in range(n):
            start = schedule.date(i)
            end = schedule.date(i + 1)
            ref_start = start
            ref_end = end
            payment_date = calendar.adjust(end, self._payment_adjustment)
            if schedule.has_is_regular() and schedule.has_tenor():
                if i == 0 and not schedule.is_regular[i]:
                    ref_start = calendar.adjust(
                        end - schedule.tenor, self._payment_adjustment
                    )
                if i == n - 1 and not schedule.is_regular[i]:
                    ref_end = calendar.adjust(
                        start + schedule.tenor, self._payment_adjustment
                    )

            cashflows.append(
                AverageBMACoupon(
                    payment_date,
                    cfv.get(self._notionals, i, self._notionals[-1]),
                    start,
                    end,
                    self._index,
                    cfv.get(self._gearings, i, 1.0),
                    cfv.get(self._spreads, i, 0.0),
                    ref_start,
                    ref_end,
                    self._payment_day_counter,
                )
            )
        return cashflows


def average_bma_leg(
    schedule: Schedule,
    index: BMAIndex,
    notionals: list[float],
    payment_day_counter: DayCounter | None = None,
    payment_adjustment: BusinessDayConvention = BusinessDayConvention.Following,
    gearings: list[float] | None = None,
    spreads: list[float] | None = None,
) -> list[CashFlow]:
    """Build a leg of :class:`AverageBMACoupon` from ``schedule``.

    Keyword-argument façade over :class:`AverageBMALeg`; the leg logic lives
    there, so there is exactly one implementation.
    """
    builder = (
        AverageBMALeg(schedule, index)
        .with_notionals(notionals)
        .with_payment_adjustment(payment_adjustment)
    )
    if payment_day_counter is not None:
        builder = builder.with_payment_day_counter(payment_day_counter)
    if gearings is not None:
        builder = builder.with_gearings(gearings)
    if spreads is not None:
        builder = builder.with_spreads(spreads)
    return builder.build()


__all__ = [
    "AverageBMACoupon",
    "AverageBMACouponPricer",
    "AverageBMALeg",
    "average_bma_leg",
]
