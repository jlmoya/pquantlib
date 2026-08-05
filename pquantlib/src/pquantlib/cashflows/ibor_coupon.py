"""IborCoupon — coupon paying a Libor-type index.

# C++ parity: ql/cashflows/iborcoupon.hpp + .cpp (v1.43).

Par vs indexed coupons is a property of the attached ``IborCouponPricer``
(C++ ``useIndexedCoupons_``), exactly as in v1.43 — C++ removed the old
``IborCoupon::Settings`` global. The default is par coupons.

The cached-data accessors ``fixing_value_date`` / ``fixing_end_date`` /
``fixing_maturity_date`` / ``spanning_time`` /
``spanning_time_index_maturity`` mirror C++
``IborCouponPricer::initializeCachedData`` (couponpricer.cpp:56-94) and,
like C++, require an ``IborCouponPricer`` to be attached: the par-coupon
branch is only defined relative to a pricer's ``useIndexedCoupons`` flag.
Unlike C++ they are recomputed on demand rather than memoised on first
use, so swapping the pricer cannot leave a stale answer behind.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

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

    # --- cached data (C++ IborCouponPricer::initializeCachedData) ---------

    def _uses_indexed_coupons(self) -> bool:
        """Read the par/indexed flag off the attached IborCouponPricer.

        C++ parity: ``IborCoupon::initializeCachedData`` (iborcoupon.cpp:57-62)
        delegates to the pricer and QL_REQUIREs it to be an
        ``IborCouponPricer``.
        """
        # Local import: coupon_pricer imports ibor_coupon lazily too, so
        # keeping this lazy avoids spelling out the cycle.
        from pquantlib.cashflows.coupon_pricer import IborCouponPricer  # noqa: PLC0415

        pricer = self.pricer()
        qassert.require(
            isinstance(pricer, IborCouponPricer),
            "IborCoupon: pricer not set or not derived from IborCouponPricer",
        )
        assert isinstance(pricer, IborCouponPricer)
        return pricer.uses_indexed_coupons()

    def fixing_value_date(self) -> Date:
        """Start of the period the index fixing spans.

        C++ parity: ``couponpricer.cpp:61-62``.
        """
        idx = self.ibor_index()
        return idx.fixing_calendar().advance(
            self._fixing_date_cached, idx.fixing_days(), TimeUnit.Days
        )

    def fixing_maturity_date(self) -> Date:
        """Natural maturity of the index period starting at the value date.

        C++ parity: ``couponpricer.cpp:63``.
        """
        return self.ibor_index().maturity_date(self.fixing_value_date())

    def fixing_end_date(self) -> Date:
        """End of the period the fixing is forecast over.

        C++ parity: ``couponpricer.cpp:65-78``. With indexed coupons (or an
        in-arrears coupon) that is the index's natural maturity; with par
        coupons it is the accrual end date snapped onto the index's
        fixing-day grid, floored at one day past the value date.
        """
        if self.is_in_arrears() or self._uses_indexed_coupons():
            return self.fixing_maturity_date()
        idx = self.ibor_index()
        cal = idx.fixing_calendar()
        # The back-step uses the COUPON's fixing days, the forward step the
        # INDEX's — the asymmetry is C++'s and it matters whenever the coupon
        # overrode fixing_days (e.g. withFixingDays(0) on a 2-day index).
        next_fixing_date = cal.advance(
            self.accrual_end_date(), -self.fixing_days(), TimeUnit.Days
        )
        end = cal.advance(next_fixing_date, idx.fixing_days(), TimeUnit.Days)
        # Make sure the estimation period contains at least one day.
        return max(end, self.fixing_value_date() + Period(1, TimeUnit.Days))

    def spanning_time(self) -> float:
        """Year fraction from the value date to the fixing end date.

        C++ parity: ``couponpricer.cpp:81-82``.
        """
        return self.ibor_index().day_counter().year_fraction(
            self.fixing_value_date(), self.fixing_end_date()
        )

    def spanning_time_index_maturity(self) -> float:
        """Year fraction from the value date to the index's natural maturity.

        C++ parity: ``couponpricer.cpp:90-91``.
        """
        return self.ibor_index().day_counter().year_fraction(
            self.fixing_value_date(), self.fixing_maturity_date()
        )

    # --- fixing -----------------------------------------------------------

    def index_fixing(self) -> float:
        """Recorded fixing if this coupon has fixed, otherwise the forecast.

        C++ parity: ``IborCoupon::indexFixing`` (iborcoupon.cpp:110-128) —
        a recorded fixing wins; otherwise the rate is forecast over
        ``(fixing_value_date, fixing_end_date)`` using the par/indexed span
        rather than the index's canonical period.

        Divergence, deliberate: C++ decides "has fixed" by comparing the
        fixing date against ``Settings::evaluationDate()`` and only then
        consults the history. This port has no mandatory global evaluation
        date — curves carry their own reference date — so, exactly as
        ``InterestRateIndex.fixing`` already does, the presence of a recorded
        fixing IS the test. The two differ only when the history holds a
        fixing for a date the global "today" considers future.

        The index slot is typed as ``IborIndexProtocol``, which guarantees
        only ``fixing(date)`` — not a fixing history and not a forecast
        curve. Both C++ branches are therefore probed for, and an index
        offering neither falls back to its own ``fixing``, which is the only
        thing the protocol promises.
        """
        idx = self.ibor_index()
        fixing_date = self._fixing_date_cached

        has_history = getattr(idx, "has_historical_fixing", None)
        past_fixing = getattr(idx, "past_fixing", None)
        if has_history is not None and past_fixing is not None and has_history(fixing_date):
            return float(past_fixing(fixing_date))

        get_ts = getattr(idx, "forecast_term_structure", None)
        ts = get_ts() if get_ts is not None else None
        if ts is None:
            return idx.fixing(fixing_date)

        value_date = self.fixing_value_date()
        end_date = self.fixing_end_date()
        spanning_time = self.spanning_time()
        qassert.require(
            spanning_time > 0.0,
            f"cannot calculate forward rate between {value_date} and {end_date}: "
            f"non positive time ({spanning_time}) using "
            f"{idx.day_counter().name()} daycounter",
        )
        # C++ IborIndex::forecastFixing(d1, d2, t) — iborindex.hpp:140-148.
        return (ts.discount(value_date) / ts.discount(end_date) - 1.0) / spanning_time
