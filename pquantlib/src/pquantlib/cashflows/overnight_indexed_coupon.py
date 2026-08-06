"""OvernightIndexedCoupon + OvernightLeg — overnight coupon and its leg builder.

# C++ parity: ql/cashflows/overnightindexedcoupon.hpp + .cpp (v1.43).

Simplified port:
- ``RateAveraging`` enum (Compound vs Simple): both are selectable — the
  averaging method picks the coupon's default pricer, exactly as in C++
  (overnightindexedcoupon.cpp:180-192).
- Lookback days / lockout days / observation shift / compound-spread-daily /
  rounding precision: all deferred — set to zero / False / not exposed.
- ``telescopicValueDates`` optimization (C++ uses it to build only a front
  and back stub of the daily value-date series): NOT ported. We always
  build the full series of business-day value dates in the rate-computation
  window. For real-world settlement schedules of ~30-90 days this is
  negligible, and the C++ optimisation is documented as producing the same
  numbers within its grace period.

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

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows import cash_flow_vectors as cfv
from pquantlib.cashflows.capped_floored_coupon import CappedFlooredOvernightIndexedCoupon
from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.fixed_rate_coupon import FixedRateCoupon
from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
from pquantlib.cashflows.overnight_indexed_coupon_pricer import (
    ArithmeticAveragedOvernightIndexedCouponPricer,
    CompoundingOvernightIndexedCouponPricer,
    OvernightIndexedCouponPricer,
)
from pquantlib.cashflows.rate_averaging import RateAveraging
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.termstructures.protocols import OvernightIndexProtocol
    from pquantlib.time.calendar import Calendar
    from pquantlib.time.period import Period
    from pquantlib.time.schedule import Schedule


class OvernightIndexedCoupon(FloatingRateCoupon):
    """Coupon paying daily-compounded overnight rates over [start, end].

    Constructor builds the full series of business-day value dates by
    walking the index's fixing calendar one day at a time.

    ``rate_computation_start_date`` / ``rate_computation_end_date`` decouple
    the window the rate is observed over from the accrual window; they are
    what an *in-advance* fixing (``OvernightLeg.in_arrears(False)``) and a
    ``last_recent_period`` need. Both default to the accrual dates.

    # C++ parity: overnightindexedcoupon.cpp:52-193.
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
        averaging_method: RateAveraging = RateAveraging.Compound,
        rate_computation_start_date: Date | None = None,
        rate_computation_end_date: Date | None = None,
        ex_coupon_date: Date | None = None,
    ) -> None:
        qassert.require(
            accrual_start_date < accrual_end_date, "startDate must be less than endDate"
        )
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
            ex_coupon_date,
        )
        rate_calc_start = (
            rate_computation_start_date
            if rate_computation_start_date is not None
            else accrual_start_date
        )
        rate_calc_end = (
            rate_computation_end_date
            if rate_computation_end_date is not None
            else accrual_end_date
        )
        self._rate_computation_start_date: Date = rate_calc_start
        self._rate_computation_end_date: Date = rate_calc_end
        self._averaging_method: RateAveraging = averaging_method

        # C++ parity: overnightindexedcoupon.cpp:117-119 —
        # ``valueDates_ = fixingCal.businessDayList(adjust(start, Preceding),
        # adjust(end, Following))``. Walking the calendar one business day at
        # a time is the same list.
        cal = overnight_index.fixing_calendar()
        value_dates: list[Date] = []
        d = cal.adjust(rate_calc_start, BusinessDayConvention.Preceding)
        end_adj = cal.adjust(rate_calc_end, BusinessDayConvention.Following)
        while d <= end_adj:
            value_dates.append(d)
            d = cal.advance(d, 1, TimeUnit.Days)
        qassert.require(len(value_dates) >= 2, "degenerate schedule (fewer than 2 fixings)")
        self._value_dates: list[Date] = value_dates
        self._fixing_dates: list[Date] = value_dates[:-1]
        # C++ parity: cpp:137-139 — the interest series is the value series
        # with the *unadjusted* rate-computation dates at both ends.
        interest_dates = list(value_dates)
        interest_dates[0] = rate_calc_start
        interest_dates[-1] = rate_calc_end
        self._interest_dates: list[Date] = interest_dates
        self._n: int = len(value_dates) - 1
        # dt = year fractions between consecutive interest dates
        dc = overnight_index.day_counter()
        self._dt: list[float] = [
            dc.year_fraction(interest_dates[i], interest_dates[i + 1]) for i in range(self._n)
        ]
        # Attach the default pricer for the averaging method (C++ cpp:180-192).
        if averaging_method == RateAveraging.Simple:
            self.set_pricer(ArithmeticAveragedOvernightIndexedCouponPricer())
        else:
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

    def averaging_method(self) -> RateAveraging:
        """# C++ parity: ``averagingMethod()`` (overnightindexedcoupon.hpp:118)."""
        return self._averaging_method

    def lockout_days(self) -> int:
        """Rate-cutoff length, always 0 in this port.

        # C++ parity: ``lockoutDays()`` (overnightindexedcoupon.hpp:120).
        # Lockout is a deferred carve-out and is deliberately NOT a
        # constructor argument, so the value cannot be accepted and then
        # dropped; the accessor exists because the coupon pricers branch on
        # it (``blackovernightindexedcouponpricer.cpp:228-231``).
        """
        return 0

    def compound_spread_daily(self) -> bool:
        """Whether the spread compounds daily; always ``False`` in this port.

        # C++ parity: ``compoundSpreadDaily()`` (overnightindexedcoupon.hpp:126).
        # Same carve-out reasoning as :meth:`lockout_days`. With ``False``,
        # ``effectiveSpread() == spread()`` and the effective cap / floor of a
        # capped-floored overnight coupon reduce to
        # ``(level - spread) / gearing``.
        """
        return False

    def rate_computation_start_date(self) -> Date:
        """# C++ parity: ``rateComputationStartDate()`` (hpp:133)."""
        return self._rate_computation_start_date

    def rate_computation_end_date(self) -> Date:
        """# C++ parity: ``rateComputationEndDate()`` (hpp:135)."""
        return self._rate_computation_end_date

    def fixing_date(self) -> Date:
        """The date the coupon is fully determined — the last fixing date.

        # C++ parity: ``fixingDate()`` (overnightindexedcoupon.hpp:141).
        """
        return self._fixing_dates[-1]

    def overnight_index(self) -> OvernightIndexProtocol:
        return self._index  # type: ignore[return-value]


class OvernightLeg:
    """Chained builder for a sequence of overnight-indexed coupons.

    # C++ parity: ``OvernightLeg`` (overnightindexedcoupon.hpp:210-266,
    # .cpp:419-687).

    **Setters deliberately not exposed.** Seven C++ setters configure coupon
    machinery this port does not have; rather than accept the argument and
    drop it, they are absent, and the reason is recorded here:

    - ``withTelescopicValueDates`` — the value-date stub optimisation is not
      ported (this port always builds the full daily series), so the flag
      would have nothing to switch.
    - ``withLookbackDays`` / ``withLockoutDays`` / ``withObservationShift`` —
      need the coupon's lookback / lockout / observation-shift fixing-date
      machinery (C++ overnightindexedcoupon.cpp:141-176), a deferred carve-out.
    - ``compoundingSpreadDaily`` — needs ``OvernightIndexedCoupon::
      effectiveSpread`` / ``effectiveIndexFixing``, a deferred carve-out.
    - ``withRoundingPrecision`` — needs ``ClosestRounding`` inside
      ``OvernightIndexedCoupon::amount()``, a deferred carve-out.
    - ``withDailyCapFloor`` — needs ``BlackOvernightIndexedCouponPricer``;
      ``CappedFlooredOvernightIndexedCoupon`` does not port the daily variant.
    """

    def __init__(self, schedule: Schedule, overnight_index: OvernightIndexProtocol) -> None:
        # C++ parity: .cpp:419-422 — the payment calendar defaults to the
        # schedule's calendar.
        self._schedule: Schedule = schedule
        self._overnight_index: OvernightIndexProtocol = overnight_index
        self._notionals: list[float] = []
        self._payment_day_counter: DayCounter | None = None
        self._payment_calendar: Calendar = schedule.calendar
        self._payment_adjustment: BusinessDayConvention = BusinessDayConvention.Following
        self._payment_lag: int = 0
        self._gearings: list[float] = []
        self._spreads: list[float] = []
        self._averaging_method: RateAveraging = RateAveraging.Compound
        self._caps: list[float] = []
        self._floors: list[float] = []
        self._naked_option: bool = False
        self._in_arrears: bool = True
        self._last_recent_period: Period | None = None
        self._last_recent_period_calendar: Calendar | None = None
        self._payment_dates: list[Date] = []
        self._coupon_pricer: OvernightIndexedCouponPricer | None = None

    # --- chained setters -----------------------------------------------

    def with_notionals(self, notionals: float | Sequence[float]) -> OvernightLeg:
        """# C++ parity: ``withNotionals`` (.cpp:424-432)."""
        self._notionals = cfv.as_float_list(notionals)
        return self

    def with_payment_day_counter(self, day_counter: DayCounter) -> OvernightLeg:
        """# C++ parity: ``withPaymentDayCounter`` (.cpp:434-437)."""
        self._payment_day_counter = day_counter
        return self

    def with_payment_adjustment(self, convention: BusinessDayConvention) -> OvernightLeg:
        """# C++ parity: ``withPaymentAdjustment`` (.cpp:439-443)."""
        self._payment_adjustment = convention
        return self

    def with_payment_calendar(self, calendar: Calendar) -> OvernightLeg:
        """# C++ parity: ``withPaymentCalendar`` (.cpp:445-448)."""
        self._payment_calendar = calendar
        return self

    def with_payment_lag(self, lag: int) -> OvernightLeg:
        """# C++ parity: ``withPaymentLag`` (.cpp:450-453)."""
        self._payment_lag = lag
        return self

    def with_gearings(self, gearings: float | Sequence[float]) -> OvernightLeg:
        """# C++ parity: ``withGearings`` (.cpp:455-463)."""
        self._gearings = cfv.as_float_list(gearings)
        return self

    def with_spreads(self, spreads: float | Sequence[float]) -> OvernightLeg:
        """# C++ parity: ``withSpreads`` (.cpp:465-473)."""
        self._spreads = cfv.as_float_list(spreads)
        return self

    def with_averaging_method(self, averaging_method: RateAveraging) -> OvernightLeg:
        """# C++ parity: ``withAveragingMethod`` (.cpp:480-483).

        Selects the coupon's default pricer: compounded or arithmetically
        averaged daily fixings.
        """
        self._averaging_method = averaging_method
        return self

    def with_caps(self, caps: float | Sequence[float]) -> OvernightLeg:
        """# C++ parity: ``withCaps`` (.cpp:507-515)."""
        self._caps = cfv.as_float_list(caps)
        return self

    def with_floors(self, floors: float | Sequence[float]) -> OvernightLeg:
        """# C++ parity: ``withFloors`` (.cpp:517-525)."""
        self._floors = cfv.as_float_list(floors)
        return self

    def with_naked_option(self, naked_option: bool = True) -> OvernightLeg:
        """# C++ parity: ``withNakedOption`` (.cpp:527-530)."""
        self._naked_option = naked_option
        return self

    def in_arrears(self, in_arrears: bool = True) -> OvernightLeg:
        """# C++ parity: ``inArrears`` (.cpp:537-540).

        Default **True** (as in C++): the rate is observed over the coupon's
        own accrual period. Setting it False moves the observation window
        back to the previous period (an *in-advance* fixing).
        """
        self._in_arrears = in_arrears
        return self

    def with_last_recent_period(self, last_recent_period: Period | None) -> OvernightLeg:
        """# C++ parity: ``withLastRecentPeriod`` (.cpp:542-545).

        Shortens the observation window to the last ``period`` before the
        rate-computation end date.
        """
        self._last_recent_period = last_recent_period
        return self

    def with_last_recent_period_calendar(self, calendar: Calendar) -> OvernightLeg:
        """# C++ parity: ``withLastRecentPeriodCalendar`` (.cpp:547-550)."""
        self._last_recent_period_calendar = calendar
        return self

    def with_payment_dates(self, payment_dates: Sequence[Date]) -> OvernightLeg:
        """# C++ parity: ``withPaymentDates`` (.cpp:553-556).

        Overrides the computed payment dates entirely; must supply exactly
        one date per calculation period.
        """
        self._payment_dates = list(payment_dates)
        return self

    def with_coupon_pricer(self, coupon_pricer: OvernightIndexedCouponPricer) -> OvernightLeg:
        """# C++ parity: ``withCouponPricer`` (.cpp:558-561).

        The pricer must match the averaging method, as in C++.
        """
        self._coupon_pricer = coupon_pricer
        return self

    # --- operator Leg() -------------------------------------------------

    def _rate_computation_dates(
        self, i: int, start: Date, end: Date, calendar: Calendar
    ) -> tuple[Date, Date]:
        """# C++ parity: .cpp:625-651 — in-arrears vs in-advance observation."""
        schedule = self._schedule
        if self._in_arrears:
            rate_start, rate_end = start, end
        elif i > 0:
            rate_start, rate_end = schedule.date(i - 1), schedule.date(i)
        else:
            rate_end = start
            if schedule.has_tenor() and schedule.tenor.length != 0:
                rate_start = calendar.adjust(
                    start - schedule.tenor, BusinessDayConvention.Preceding
                )
            else:
                rate_start = calendar.adjust(
                    start - (end - start), BusinessDayConvention.Preceding
                )
        if self._last_recent_period is not None:
            cal = (
                self._last_recent_period_calendar
                if self._last_recent_period_calendar is not None
                else calendar
            )
            rate_start = cal.advance_period(rate_end, -self._last_recent_period)
        return rate_start, rate_end

    def build(self) -> list[CashFlow]:
        """Build the leg.

        # C++ parity: ``OvernightLeg::operator Leg()`` (.cpp:563-687).
        """
        qassert.require(len(self._notionals) > 0, "no notional given")
        if self._coupon_pricer is not None:
            if self._averaging_method == RateAveraging.Compound:
                qassert.require(
                    isinstance(self._coupon_pricer, CompoundingOvernightIndexedCouponPricer),
                    "Wrong coupon pricer provided, provide a "
                    "CompoundingOvernightIndexedCouponPricer",
                )
            else:
                qassert.require(
                    isinstance(
                        self._coupon_pricer, ArithmeticAveragedOvernightIndexedCouponPricer
                    ),
                    "Wrong coupon pricer provided, provide a "
                    "ArithmeticAveragedOvernightIndexedCouponPricer",
                )

        schedule = self._schedule
        calendar = schedule.calendar
        payment_calendar = self._payment_calendar
        n = len(schedule) - 1
        if self._payment_dates:
            qassert.require(
                len(self._payment_dates) == n,
                f"Expected the number of explicit payment dates "
                f"({len(self._payment_dates)}) to equal the number of "
                f"calculation periods ({n})",
            )

        day_counter = (
            self._payment_day_counter
            if self._payment_day_counter is not None
            else self._overnight_index.day_counter()
        )
        cashflows: list[CashFlow] = []
        for i in range(n):
            start = schedule.date(i)
            end = schedule.date(i + 1)
            payment_date = (
                self._payment_dates[i]
                if self._payment_dates
                else payment_calendar.advance(
                    end, self._payment_lag, TimeUnit.Days, self._payment_adjustment
                )
            )
            # C++ parity: .cpp:615-621 — the stub reference periods use the
            # *payment* adjustment here, not the schedule's own convention.
            ref_start, ref_end = start, end
            if schedule.has_is_regular() and schedule.has_tenor():
                if i == 0 and not schedule.is_regular_at(1):
                    ref_start = calendar.adjust(
                        end - schedule.tenor, self._payment_adjustment
                    )
                if i == n - 1 and not schedule.is_regular_at(n):
                    ref_end = calendar.adjust(
                        start + schedule.tenor, self._payment_adjustment
                    )
            rate_start, rate_end = self._rate_computation_dates(i, start, end, calendar)

            gearing = cfv.get(self._gearings, i, 1.0)
            nominal = cfv.get(self._notionals, i, 1.0)
            if gearing == 0.0:
                cashflows.append(
                    FixedRateCoupon.from_rate(
                        payment_date,
                        nominal,
                        cfv.effective_fixed_rate(self._spreads, self._caps, self._floors, i),
                        day_counter,
                        start,
                        end,
                        ref_start,
                        ref_end,
                    )
                )
                continue

            coupon = OvernightIndexedCoupon(
                payment_date,
                nominal,
                start,
                end,
                self._overnight_index,
                gearing,
                cfv.get(self._spreads, i, 0.0),
                ref_start,
                ref_end,
                day_counter,
                self._averaging_method,
                rate_start,
                rate_end,
            )
            if self._coupon_pricer is not None:
                coupon.set_pricer(self._coupon_pricer)
            cap = cfv.get_or_none(self._caps, i)
            floor = cfv.get_or_none(self._floors, i)
            if cap is None and floor is None:
                cashflows.append(coupon)
            else:
                capped = CappedFlooredOvernightIndexedCoupon(
                    coupon, cap, floor, self._naked_option
                )
                if self._coupon_pricer is not None:
                    capped.set_pricer(self._coupon_pricer)
                cashflows.append(capped)
        return cashflows


# The default pricer (CompoundingOvernightIndexedCouponPricer) and the
# arithmetic-average / base variants now live in
# ``overnight_indexed_coupon_pricer`` (the full-fidelity port with
# past-fixing handling). It is imported above and attached in __init__.
# It is re-exported here for backwards-compatible imports.
__all__ = [
    "CompoundingOvernightIndexedCouponPricer",
    "OvernightIndexedCoupon",
    "OvernightLeg",
]
