"""MultipleResetsCoupon — coupon compounding or averaging multiple fixings.

# C++ parity: ql/cashflows/multipleresetscoupon.hpp + .cpp (v1.43).

A ``MultipleResetsCoupon`` accrues over one coupon period but takes several
index fixings inside it, one per period of a *reset schedule*. The reset
schedule's first and last dates are the coupon's accrual start and end; each
schedule period contributes one fixing, taken ``fixing_days`` before the start
of that period.

Components:

- :class:`MultipleResetsCoupon` — the coupon.
- :class:`MultipleResetsPricer` — abstract base holding the sub-period fixings.
  Everything except ``swaplet_rate`` is a C++ ``QL_FAIL`` and raises
  :class:`~pquantlib.exceptions.LibraryException` here.
- :class:`AveragingMultipleResetsPricer` — ``RateAveraging.Simple``.
- :class:`CompoundingMultipleResetsPricer` — ``RateAveraging.Compound``.
- :class:`MultipleResetsLeg` — the chained builder.

Python divergences from C++:

- ``MultipleResetsLeg::operator Leg()`` becomes :meth:`MultipleResetsLeg.build`
  (with ``__call__`` delegating to it), mirroring ``MakeSchedule.build`` /
  ``MakeSchedule.__call__`` in ``pquantlib.time.schedule``. The ``with*``
  setters stay chained, unlike ``ibor_leg`` / ``overnight_leg``, because this
  builder carries enough state that keyword arguments would be unwieldy and the
  C++ shape is worth preserving here.
- The C++ scalar/vector ``withNotionals`` (and friends) overload pairs collapse
  into one setter taking ``float | Sequence[float]`` via
  ``cash_flow_vectors.as_float_list`` / ``as_int_list``; a scalar is stored as a
  one-element list exactly as C++ does, so ``cash_flow_vectors.get`` (C++
  ``detail::get``) falling through to ``back()`` reproduces the "same value for
  every coupon" behaviour.
- ``MultipleResetsLeg``'s ``index`` parameter is typed ``... | None`` because
  the C++ ``QL_REQUIRE(index_, "no index provided")`` is a real runtime guard
  against a null ``shared_ptr``, and this port keeps that guard.
- An empty C++ ``Calendar`` (``exCouponCalendar_``) is spelled ``None``.
- ``accept(AcyclicVisitor&)`` is omitted (Visitor carve-out, as elsewhere in
  ``pquantlib.cashflows``).
- ``MultipleResetsPricer::initialize``'s ``dynamic_pointer_cast<IborIndex>``
  guard ("IborIndex required") is unreachable here: the coupon's ``index``
  parameter is already typed ``IborIndexProtocol``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows import cash_flow_vectors as cfv
from pquantlib.cashflows.coupon_pricer import (
    FloatingRateCouponPricer,
    set_coupon_pricer,
)
from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
from pquantlib.cashflows.rate_averaging import RateAveraging
from pquantlib.exceptions import LibraryException
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pquantlib.cashflows.cash_flow import CashFlow
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.termstructures.protocols import IborIndexProtocol
    from pquantlib.time.calendar import Calendar
    from pquantlib.time.date import Date
    from pquantlib.time.schedule import Schedule


class MultipleResetsCoupon(FloatingRateCoupon):
    """Coupon paying a rate compounded or averaged over multiple fixings.

    # C++ parity: ``MultipleResetsCoupon`` (multipleresetscoupon.hpp:40-90,
    .cpp:30-79).

    ``reset_schedule`` drives everything: its first and last dates are the
    coupon's accrual start / end, and each of its periods contributes one
    fixing taken ``fixing_days`` before the period start.
    """

    def __init__(
        self,
        payment_date: Date,
        nominal: float,
        reset_schedule: Schedule,
        fixing_days: int,
        index: IborIndexProtocol,
        gearing: float = 1.0,
        coupon_spread: float = 0.0,
        rate_spread: float = 0.0,
        ref_period_start: Date | None = None,
        ref_period_end: Date | None = None,
        day_counter: DayCounter | None = None,
        ex_coupon_date: Date | None = None,
    ) -> None:
        super().__init__(
            payment_date,
            nominal,
            reset_schedule.front(),
            reset_schedule.back(),
            fixing_days,
            index,
            gearing,
            coupon_spread,
            ref_period_start,
            ref_period_end,
            day_counter,
            False,  # is_in_arrears — C++ passes a hard-coded false
            ex_coupon_date,
        )
        self._rate_spread: float = rate_spread
        self._value_dates: list[Date] = list(reset_schedule.dates)
        self._n: int = len(self._value_dates) - 1

        # C++ parity: multipleresetscoupon.cpp:48-56 — with zero fixing days the
        # fixing dates ARE the value dates (no calendar roll at all), which is
        # not the same as rolling back by zero business days.
        if self._fixing_days == 0:
            self._fixing_dates: list[Date] = list(self._value_dates[:-1])
        else:
            self._fixing_dates = [self._fixing_date_of(d) for d in self._value_dates[:-1]]

        # accrual times of the sub-periods, on the INDEX's day counter
        # (C++ multipleresetscoupon.cpp:58-62) — not the coupon's.
        index_dc = index.day_counter()
        self._dt: list[float] = [
            index_dc.year_fraction(self._value_dates[i], self._value_dates[i + 1]) for i in range(self._n)
        ]

    def _fixing_date_of(self, value_date: Date) -> Date:
        """Fixing date for a sub-period starting on ``value_date``.

        # C++ parity: private ``MultipleResetsCoupon::fixingDate(const Date&)``
        (multipleresetscoupon.cpp:73-77).
        """
        return self._index.fixing_calendar().advance(value_date, -self._fixing_days, TimeUnit.Days)

    # --- inspectors ----------------------------------------------------

    def fixing_dates(self) -> list[Date]:
        """Fixing dates for the rates to be compounded / averaged."""
        return list(self._fixing_dates)

    def value_dates(self) -> list[Date]:
        """Value dates for the rates to be compounded / averaged."""
        return list(self._value_dates)

    def dt(self) -> list[float]:
        """Accrual (compounding) periods of the sub-periods."""
        return list(self._dt)

    def rate_spread(self) -> float:
        """Spread added to each of the underlying fixings."""
        return self._rate_spread

    def ibor_index(self) -> IborIndexProtocol:
        """Narrowed accessor returning the ``IborIndexProtocol``-typed index."""
        # Already narrowed at construction via the typed parameter.
        return self._index  # type: ignore[return-value]

    # --- FloatingRateCoupon interface -----------------------------------

    def fixing_date(self) -> Date:
        """The date when the coupon is fully determined — the LAST fixing date.

        # C++ parity: multipleresetscoupon.hpp:78 (inline override).
        """
        return self._fixing_dates[-1]


class MultipleResetsPricer(FloatingRateCouponPricer):
    """Base pricer for :class:`MultipleResetsCoupon`.

    # C++ parity: ``MultipleResetsPricer`` (multipleresetscoupon.hpp:93-106,
    .cpp:81-118).

    Abstract exactly as in C++: ``swapletRate`` stays pure virtual, so only
    :class:`AveragingMultipleResetsPricer` and
    :class:`CompoundingMultipleResetsPricer` can be instantiated. Every other
    price/rate method is a C++ ``QL_FAIL``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._coupon: MultipleResetsCoupon | None = None
        self._sub_period_fixings: list[float] = []

    def initialize(self, coupon: FloatingRateCoupon) -> None:
        """Cache the coupon and its rate-spread-shifted sub-period fixings.

        # C++ parity: multipleresetscoupon.cpp:81-99.
        """
        qassert.require(isinstance(coupon, MultipleResetsCoupon), "sub-periods coupon required")
        assert isinstance(coupon, MultipleResetsCoupon)
        self._coupon = coupon
        qassert.require(coupon.accrual_period() != 0.0, "null accrual period")
        index = coupon.ibor_index()
        rate_spread = coupon.rate_spread()
        self._sub_period_fixings = [index.fixing(d, False) + rate_spread for d in coupon.fixing_dates()]

    def _coupon_or_fail(self) -> MultipleResetsCoupon:
        qassert.require(self._coupon is not None, "pricer not initialized")
        assert self._coupon is not None
        return self._coupon

    # --- CouponPricer interface (all QL_FAIL in C++) ---------------------

    def swaplet_price(self) -> float:
        msg = "MultipleResetsPricer::swapletPrice not implemented"
        raise LibraryException(msg)

    def caplet_price(self, effective_cap: float) -> float:
        del effective_cap
        msg = "MultipleResetsPricer::capletPrice not implemented"
        raise LibraryException(msg)

    def caplet_rate(self, effective_cap: float) -> float:
        del effective_cap
        msg = "MultipleResetsPricer::capletRate not implemented"
        raise LibraryException(msg)

    def floorlet_price(self, effective_floor: float) -> float:
        del effective_floor
        msg = "MultipleResetsPricer::floorletPrice not implemented"
        raise LibraryException(msg)

    def floorlet_rate(self, effective_floor: float) -> float:
        del effective_floor
        msg = "MultipleResetsPricer::floorletRate not implemented"
        raise LibraryException(msg)


class AveragingMultipleResetsPricer(MultipleResetsPricer):
    """``RateAveraging.Simple`` — time-weighted average of the sub-period rates.

    # C++ parity: ``AveragingMultipleResetsPricer::swapletRate``
    (multipleresetscoupon.cpp:120-133).
    """

    def swaplet_rate(self) -> float:
        coupon = self._coupon_or_fail()
        sub_period_fractions = coupon.dt()
        aggregate_factor = 0.0
        for fixing, tau in zip(self._sub_period_fixings, sub_period_fractions, strict=True):
            aggregate_factor += fixing * tau
        rate = aggregate_factor / coupon.accrual_period()
        return coupon.gearing() * rate + coupon.spread()


class CompoundingMultipleResetsPricer(MultipleResetsPricer):
    """``RateAveraging.Compound`` — the sub-period rates compounded.

    # C++ parity: ``CompoundingMultipleResetsPricer::swapletRate``
    (multipleresetscoupon.cpp:135-147).
    """

    def swaplet_rate(self) -> float:
        coupon = self._coupon_or_fail()
        sub_period_fractions = coupon.dt()
        compound_factor = 1.0
        for fixing, tau in zip(self._sub_period_fixings, sub_period_fractions, strict=True):
            compound_factor *= 1.0 + fixing * tau
        rate = (compound_factor - 1.0) / coupon.accrual_period()
        return coupon.gearing() * rate + coupon.spread()


class MultipleResetsLeg:
    """Chained builder for a sequence of :class:`MultipleResetsCoupon`.

    # C++ parity: ``MultipleResetsLeg`` (multipleresetscoupon.hpp:113-172,
    .cpp:151-291).

    ``full_reset_schedule`` specifies the reset periods of *all* coupons;
    ``resets_per_coupon`` consecutive periods make up one coupon, so the number
    of schedule periods must be an exact multiple of it.
    """

    def __init__(
        self,
        full_reset_schedule: Schedule,
        index: IborIndexProtocol | None,
        resets_per_coupon: int,
    ) -> None:
        self._schedule: Schedule = full_reset_schedule
        self._resets_per_coupon: int = resets_per_coupon
        self._notionals: list[float] = []
        self._payment_day_counter: DayCounter | None = None
        self._payment_calendar: Calendar = full_reset_schedule.calendar
        self._payment_adjustment: BusinessDayConvention = BusinessDayConvention.Following
        self._payment_lag: int = 0
        self._fixing_days: list[int] = []
        self._gearings: list[float] = []
        self._coupon_spreads: list[float] = []
        self._rate_spreads: list[float] = []
        self._averaging_method: RateAveraging = RateAveraging.Compound
        self._ex_coupon_period: Period = Period()
        self._ex_coupon_calendar: Calendar | None = None
        self._ex_coupon_adjustment: BusinessDayConvention = BusinessDayConvention.Unadjusted
        self._ex_coupon_end_of_month: bool = False

        # C++ parity: multipleresetscoupon.cpp:155-160.
        qassert.require(index is not None, "no index provided")
        assert index is not None
        self._index: IborIndexProtocol = index
        qassert.require(not full_reset_schedule.empty(), "empty schedule provided")
        qassert.require(
            (full_reset_schedule.size() - 1) % resets_per_coupon == 0,
            "number of resets per coupon does not divide exactly number of periods in schedule",
        )

    # --- chained setters -------------------------------------------------

    def with_notionals(self, notionals: float | Sequence[float]) -> MultipleResetsLeg:
        self._notionals = cfv.as_float_list(notionals)
        return self

    def with_payment_day_counter(self, day_counter: DayCounter) -> MultipleResetsLeg:
        self._payment_day_counter = day_counter
        return self

    def with_payment_adjustment(self, convention: BusinessDayConvention) -> MultipleResetsLeg:
        self._payment_adjustment = convention
        return self

    def with_payment_calendar(self, calendar: Calendar) -> MultipleResetsLeg:
        self._payment_calendar = calendar
        return self

    def with_payment_lag(self, lag: int) -> MultipleResetsLeg:
        self._payment_lag = lag
        return self

    def with_fixing_days(self, fixing_days: int | Sequence[int]) -> MultipleResetsLeg:
        self._fixing_days = cfv.as_int_list(fixing_days)
        return self

    def with_gearings(self, gearings: float | Sequence[float]) -> MultipleResetsLeg:
        self._gearings = cfv.as_float_list(gearings)
        return self

    def with_coupon_spreads(self, spreads: float | Sequence[float]) -> MultipleResetsLeg:
        self._coupon_spreads = cfv.as_float_list(spreads)
        return self

    def with_rate_spreads(self, spreads: float | Sequence[float]) -> MultipleResetsLeg:
        self._rate_spreads = cfv.as_float_list(spreads)
        return self

    def with_ex_coupon_period(
        self,
        period: Period,
        calendar: Calendar | None,
        convention: BusinessDayConvention,
        end_of_month: bool = False,
    ) -> MultipleResetsLeg:
        """``calendar=None`` mirrors an empty C++ ``Calendar()``.

        With no calendar the ex-coupon date is rolled on the *schedule's*
        calendar (C++ multipleresetscoupon.cpp:264-271).
        """
        self._ex_coupon_period = period
        self._ex_coupon_calendar = calendar
        self._ex_coupon_adjustment = convention
        self._ex_coupon_end_of_month = end_of_month
        return self

    def with_averaging_method(self, averaging_method: RateAveraging) -> MultipleResetsLeg:
        self._averaging_method = averaging_method
        return self

    # --- build -----------------------------------------------------------

    def build(self) -> list[CashFlow]:
        """Build the leg and attach the averaging-method-selected pricer.

        # C++ parity: ``MultipleResetsLeg::operator Leg()``
        (multipleresetscoupon.cpp:251-291).
        """
        calendar = self._schedule.calendar
        n = (self._schedule.size() - 1) // self._resets_per_coupon
        qassert.require(len(self._notionals) > 0, "no notional given")
        qassert.require(
            len(self._notionals) <= n,
            f"too many nominals ({len(self._notionals)}), only {n} required",
        )
        qassert.require(
            len(self._gearings) <= n,
            f"too many gearings ({len(self._gearings)}), only {n} required",
        )
        qassert.require(
            len(self._coupon_spreads) <= n,
            f"too many coupon spreads ({len(self._coupon_spreads)}), only {n} required",
        )
        qassert.require(
            len(self._rate_spreads) <= n,
            f"too many rate spreads ({len(self._rate_spreads)}), only {n} required",
        )
        qassert.require(
            len(self._fixing_days) <= n,
            f"too many fixing days ({len(self._fixing_days)}), only {n} required",
        )

        cashflows: list[CashFlow] = []
        for i in range(n):
            start = self._schedule.date(i * self._resets_per_coupon)
            end = self._schedule.date((i + 1) * self._resets_per_coupon)
            sub_schedule = self._schedule.after(start).until(end)
            payment_date = self._payment_calendar.advance(
                end, self._payment_lag, TimeUnit.Days, self._payment_adjustment
            )
            ex_coupon_date: Date | None = None
            # C++ tests ``exCouponPeriod_ != Period()``; a C++ Period compares
            # equal to the default one iff its length is zero (period.cpp
            # operator< short-circuits on zero length), whereas Python's
            # dataclass __eq__ also compares the unit — so test the length.
            if self._ex_coupon_period.length != 0:
                ex_cal = self._ex_coupon_calendar if self._ex_coupon_calendar is not None else calendar
                ex_coupon_date = ex_cal.advance_period(
                    payment_date,
                    -self._ex_coupon_period,
                    self._ex_coupon_adjustment,
                    self._ex_coupon_end_of_month,
                )
            cashflows.append(
                MultipleResetsCoupon(
                    payment_date,
                    cfv.get(self._notionals, i, self._notionals[-1]),
                    sub_schedule,
                    cfv.get(self._fixing_days, i, self._index.fixing_days()),
                    self._index,
                    cfv.get(self._gearings, i, 1.0),
                    cfv.get(self._coupon_spreads, i, 0.0),
                    cfv.get(self._rate_spreads, i, 0.0),
                    start,
                    end,
                    self._payment_day_counter,
                    ex_coupon_date,
                )
            )

        pricer: FloatingRateCouponPricer
        if self._averaging_method == RateAveraging.Simple:
            pricer = AveragingMultipleResetsPricer()
        elif self._averaging_method == RateAveraging.Compound:
            pricer = CompoundingMultipleResetsPricer()
        else:
            msg = f"unknown compounding convention ({int(self._averaging_method)})"
            raise LibraryException(msg)
        set_coupon_pricer(cashflows, pricer)
        return cashflows

    def __call__(self) -> list[CashFlow]:
        """Alias for :meth:`build`, mirroring C++ ``operator Leg()``."""
        return self.build()
