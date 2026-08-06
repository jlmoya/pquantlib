"""FixedRateCoupon + FixedRateLeg — fixed-rate coupon and its leg builder.

# C++ parity: ql/cashflows/fixedratecoupon.hpp + .cpp (v1.43).

amount = nominal * (rate.compound_factor(start, end, refStart, refEnd) - 1)

The C++ class has two ctors — one taking a raw Rate + DayCounter (which
internally builds an InterestRate(Simple, Annual)), and one taking a
fully-constructed ``InterestRate``. Python exposes a single ``__init__``
plus ``FixedRateCoupon.from_rate(...)`` classmethod for the raw-rate
variant.

:class:`FixedRateLeg` is the chained builder; ``operator Leg()`` becomes
:meth:`FixedRateLeg.build`. The free function
:func:`pquantlib.cashflows.fixed_rate_leg.fixed_rate_leg` is a thin wrapper
over it, so there is exactly one implementation of the leg logic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.coupon import Coupon
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.interest_rate import InterestRate
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.time.calendar import Calendar
    from pquantlib.time.period import Period
    from pquantlib.time.schedule import Schedule


class FixedRateCoupon(Coupon):
    """Coupon paying a fixed interest rate.

    The interest rate carries the day counter + compounding + frequency,
    so the coupon's ``day_counter()`` is derived from the rate.
    """

    def __init__(
        self,
        payment_date: Date,
        nominal: float,
        interest_rate: InterestRate,
        accrual_start_date: Date,
        accrual_end_date: Date,
        ref_period_start: Date | None = None,
        ref_period_end: Date | None = None,
        ex_coupon_date: Date | None = None,
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
        self._rate: InterestRate = interest_rate

    @classmethod
    def from_rate(
        cls,
        payment_date: Date,
        nominal: float,
        rate: float,
        day_counter: DayCounter,
        accrual_start_date: Date,
        accrual_end_date: Date,
        ref_period_start: Date | None = None,
        ref_period_end: Date | None = None,
        ex_coupon_date: Date | None = None,
    ) -> FixedRateCoupon:
        """C++ ctor 1: build a Simple/Annual InterestRate internally.

        C++ parity: ql/cashflows/fixedratecoupon.cpp:32-43.
        """
        ir = InterestRate(rate, day_counter, Compounding.Simple, Frequency.Annual)
        return cls(
            payment_date,
            nominal,
            ir,
            accrual_start_date,
            accrual_end_date,
            ref_period_start,
            ref_period_end,
            ex_coupon_date,
        )

    # --- Coupon interface ----------------------------------------------

    def rate(self) -> float:
        return self._rate.rate()

    def interest_rate(self) -> InterestRate:
        return self._rate

    def day_counter(self) -> DayCounter:
        return self._rate.day_counter()

    def amount(self) -> float:
        """amount = nominal * (compound_factor(start, end, ref_start, ref_end) - 1).

        C++ parity: ql/cashflows/fixedratecoupon.cpp:62-71.
        """
        return self._nominal * (
            self._rate.compound_factor_dates(
                self._accrual_start_date,
                self._accrual_end_date,
                self._ref_period_start,
                self._ref_period_end,
            )
            - 1.0
        )

    def accrued_amount(self, d: Date) -> float:
        """Accrued amount at the given date.

        C++ parity: ql/cashflows/fixedratecoupon.cpp:73-89.
        """
        if d <= self._accrual_start_date or d > self._payment_date:
            return 0.0
        if self.trading_ex_coupon(d):
            end = max(d, self._accrual_end_date)
            return -self._nominal * (
                self._rate.compound_factor_dates(
                    d, end, self._ref_period_start, self._ref_period_end
                )
                - 1.0
            )
        end = min(d, self._accrual_end_date)
        return self._nominal * (
            self._rate.compound_factor_dates(
                self._accrual_start_date,
                end,
                self._ref_period_start,
                self._ref_period_end,
            )
            - 1.0
        )


class FixedRateLeg:
    """Chained builder for a sequence of :class:`FixedRateCoupon`.

    # C++ parity: ``FixedRateLeg`` (fixedratecoupon.hpp:89-127, .cpp:92-275).

    Every C++ ``withXxx`` setter is present as ``with_xxx`` and returns
    ``self``; C++'s ``operator Leg()`` is :meth:`build`::

        leg = (
            FixedRateLeg(schedule)
            .with_notionals(1_000_000.0)
            .with_coupon_rates(0.04, Actual360())
            .with_payment_lag(2)
            .build()
        )
    """

    def __init__(self, schedule: Schedule) -> None:
        # C++ parity: fixedratecoupon.cpp:92-93 — the payment calendar
        # defaults to the schedule's calendar.
        self._schedule: Schedule = schedule
        self._notionals: list[float] = []
        self._coupon_rates: list[InterestRate] = []
        self._first_period_day_counter: DayCounter | None = None
        self._last_period_day_counter: DayCounter | None = None
        self._payment_calendar: Calendar = schedule.calendar
        self._payment_adjustment: BusinessDayConvention = BusinessDayConvention.Following
        self._payment_lag: int = 0
        self._ex_coupon_period: Period | None = None
        self._ex_coupon_calendar: Calendar | None = None
        self._ex_coupon_adjustment: BusinessDayConvention = BusinessDayConvention.Following
        self._ex_coupon_end_of_month: bool = False

    # --- chained setters -----------------------------------------------

    def with_notionals(self, notionals: float | Sequence[float]) -> FixedRateLeg:
        """# C++ parity: ``withNotionals`` (both overloads, .cpp:95-103)."""
        self._notionals = (
            [float(notionals)]
            if isinstance(notionals, int | float)
            else [float(x) for x in notionals]
        )
        return self

    def with_coupon_rates(
        self,
        rates: float | InterestRate | Sequence[float] | Sequence[InterestRate],
        payment_day_counter: DayCounter | None = None,
        compounding: Compounding = Compounding.Simple,
        frequency: Frequency = Frequency.Annual,
    ) -> FixedRateLeg:
        """# C++ parity: ``withCouponRates`` (all four overloads, .cpp:105-138).

        Collapses the C++ overload set: ``rates`` may be a single rate, a
        single :class:`~pquantlib.interest_rate.InterestRate`, or a sequence
        of either. ``payment_day_counter`` / ``compounding`` / ``frequency``
        are only consulted for plain-number rates (as in C++, where the
        ``InterestRate`` overloads carry their own convention).
        """
        if isinstance(rates, InterestRate):
            self._coupon_rates = [rates]
            return self
        if isinstance(rates, int | float):
            qassert.require(
                payment_day_counter is not None,
                "a payment day counter is required for plain coupon rates",
            )
            assert payment_day_counter is not None
            self._coupon_rates = [
                InterestRate(float(rates), payment_day_counter, compounding, frequency)
            ]
            return self
        built: list[InterestRate] = []
        for r in rates:
            if isinstance(r, InterestRate):
                built.append(r)
            else:
                qassert.require(
                    payment_day_counter is not None,
                    "a payment day counter is required for plain coupon rates",
                )
                assert payment_day_counter is not None
                built.append(
                    InterestRate(float(r), payment_day_counter, compounding, frequency)
                )
        self._coupon_rates = built
        return self

    def with_payment_adjustment(self, convention: BusinessDayConvention) -> FixedRateLeg:
        """# C++ parity: ``withPaymentAdjustment`` (.cpp:140-144)."""
        self._payment_adjustment = convention
        return self

    def with_first_period_day_counter(self, day_counter: DayCounter) -> FixedRateLeg:
        """# C++ parity: ``withFirstPeriodDayCounter`` (.cpp:146-150)."""
        self._first_period_day_counter = day_counter
        return self

    def with_last_period_day_counter(self, day_counter: DayCounter) -> FixedRateLeg:
        """# C++ parity: ``withLastPeriodDayCounter`` (.cpp:152-156)."""
        self._last_period_day_counter = day_counter
        return self

    def with_payment_calendar(self, calendar: Calendar) -> FixedRateLeg:
        """# C++ parity: ``withPaymentCalendar`` (.cpp:158-161)."""
        self._payment_calendar = calendar
        return self

    def with_payment_lag(self, lag: int) -> FixedRateLeg:
        """# C++ parity: ``withPaymentLag`` (.cpp:163-166)."""
        self._payment_lag = lag
        return self

    def with_ex_coupon_period(
        self,
        period: Period,
        calendar: Calendar,
        convention: BusinessDayConvention,
        end_of_month: bool = False,
    ) -> FixedRateLeg:
        """# C++ parity: ``withExCouponPeriod`` (.cpp:168-178)."""
        self._ex_coupon_period = period
        self._ex_coupon_calendar = calendar
        self._ex_coupon_adjustment = convention
        self._ex_coupon_end_of_month = end_of_month
        return self

    # --- operator Leg() -------------------------------------------------

    def _payment_date(self, end: Date) -> Date:
        return self._payment_calendar.advance(
            end, self._payment_lag, TimeUnit.Days, self._payment_adjustment
        )

    def _ex_coupon_date(self, payment_date: Date) -> Date | None:
        if self._ex_coupon_period is None:
            return None
        cal = self._ex_coupon_calendar
        assert cal is not None
        return cal.advance(
            payment_date,
            -self._ex_coupon_period.length,
            self._ex_coupon_period.units,
            self._ex_coupon_adjustment,
            self._ex_coupon_end_of_month,
        )

    def _rate_at(self, i: int) -> InterestRate:
        return self._coupon_rates[min(i, len(self._coupon_rates) - 1)]

    def _nominal_at(self, i: int) -> float:
        return self._notionals[min(i, len(self._notionals) - 1)]

    def _with_day_counter(self, rate: InterestRate, dc: DayCounter | None) -> InterestRate:
        if dc is None:
            return rate
        return InterestRate(rate.rate(), dc, rate.compounding(), rate.frequency())

    def build(self) -> list[CashFlow]:
        """Build the leg.

        # C++ parity: ``FixedRateLeg::operator Leg()`` (.cpp:180-275).

        The first and last periods are handled separately from the regular
        ones so that a short/long stub gets a widened reference period and
        the first/last period day counters can override the rate's own.
        """
        qassert.require(len(self._coupon_rates) > 0, "no coupon rates given")
        qassert.require(len(self._notionals) > 0, "no notional given")
        schedule = self._schedule
        n_dates = len(schedule)
        qassert.require(n_dates >= 2, "schedule has fewer than 2 dates")

        leg: list[CashFlow] = []

        # first period might be short or long
        start = schedule.date(0)
        end = schedule.date(1)
        payment_date = self._payment_date(end)
        ex_coupon_date = self._ex_coupon_date(payment_date)
        rate = self._rate_at(0)
        nominal = self._nominal_at(0)
        if schedule.has_tenor() and schedule.has_is_regular() and not schedule.is_regular_at(1):
            ref = schedule.calendar.advance(
                end,
                -schedule.tenor.length,
                schedule.tenor.units,
                schedule.business_day_convention,
                schedule.end_of_month,
            )
        else:
            ref = start
        leg.append(
            FixedRateCoupon(
                payment_date,
                nominal,
                self._with_day_counter(rate, self._first_period_day_counter),
                start,
                end,
                ref,
                end,
                ex_coupon_date,
            )
        )

        # regular periods
        for i in range(2, n_dates - 1):
            start, end = end, schedule.date(i)
            payment_date = self._payment_date(end)
            ex_coupon_date = self._ex_coupon_date(payment_date)
            leg.append(
                FixedRateCoupon(
                    payment_date,
                    self._nominal_at(i - 1),
                    self._rate_at(i - 1),
                    start,
                    end,
                    start,
                    end,
                    ex_coupon_date,
                )
            )

        if n_dates > 2:
            # last period might be short or long
            start, end = end, schedule.date(n_dates - 1)
            payment_date = self._payment_date(end)
            ex_coupon_date = self._ex_coupon_date(payment_date)
            rate = self._with_day_counter(
                self._rate_at(n_dates - 2), self._last_period_day_counter
            )
            if (
                schedule.has_is_regular() and schedule.is_regular_at(n_dates - 1)
            ) or not schedule.has_tenor():
                ref_end = end
            else:
                ref_end = schedule.calendar.advance(
                    start,
                    schedule.tenor.length,
                    schedule.tenor.units,
                    schedule.business_day_convention,
                    schedule.end_of_month,
                )
            leg.append(
                FixedRateCoupon(
                    payment_date,
                    self._nominal_at(n_dates - 2),
                    rate,
                    start,
                    end,
                    start,
                    ref_end,
                    ex_coupon_date,
                )
            )
        return leg


__all__ = ["FixedRateCoupon", "FixedRateLeg"]
