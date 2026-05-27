"""fixed_rate_leg — free-function FixedRateLeg builder.

# C++ parity: ql/cashflows/fixedratecoupon.hpp class ``FixedRateLeg``
   (the chained-builder ``operator Leg()`` is folded into a single free
   function — see L2-D design carve-out).

Python-idiomatic divergence: the C++ class is a chained builder
(``FixedRateLeg(schedule).withNotionals(N).withCouponRates(...).operator Leg()``)
exposing ~10 ``with*`` setters. Each of those setters is reified here as
a keyword argument on a single function. The carve-outs from the full
C++ surface are:

- ``with_first_period_day_counter`` / ``with_last_period_day_counter``
  — deferred (rarely used in tests).
- ``with_payment_lag`` — deferred (always 0 in default L2-D coverage).
- ``with_ex_coupon_period`` + variants — deferred.
- ``with_indexed_coupons`` / ``with_at_par_coupons`` — deferred (no IborCoupon
  par/indexed flag yet).

The retained surface is the minimum sufficient to build a vanilla
fixed-rate bond leg.
"""

from __future__ import annotations

from collections.abc import Sequence

from pquantlib import qassert
from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.fixed_rate_coupon import FixedRateCoupon
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.interest_rate import InterestRate
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.schedule import Schedule


def _pick_nominal(seq: Sequence[float], i: int) -> float:
    return float(seq[i] if i < len(seq) else seq[-1])


def _pick_rate(seq: Sequence[InterestRate], i: int) -> InterestRate:
    return seq[i] if i < len(seq) else seq[-1]


def _normalise_rates(
    rates: Sequence[float] | Sequence[InterestRate],
    day_counter: DayCounter | None,
    compounding: Compounding,
    frequency: Frequency,
) -> list[InterestRate]:
    first = rates[0]
    if isinstance(first, InterestRate):
        # All elements must be InterestRate; cast wholesale.
        return [r for r in rates if isinstance(r, InterestRate)]
    qassert.require(day_counter is not None, "day_counter required for plain rates")
    # narrowed: ``rates`` is Sequence[float], day_counter is not None
    assert day_counter is not None
    return [
        InterestRate(float(r), day_counter, compounding, frequency)
        for r in rates
        if not isinstance(r, InterestRate)
    ]


def _compute_ref_period(
    schedule: Schedule,
    n_periods: int,
    i: int,
    start: Date,
    end: Date,
) -> tuple[Date, Date]:
    """Replicate C++ FixedRateLeg ref-period logic for first/last periods."""
    ref_start = start
    ref_end = end
    if (
        i == 0
        and schedule.has_tenor()
        and schedule.has_is_regular()
        and not schedule.is_regular_at(1)
    ):
        tenor = schedule.tenor
        ref_start = schedule.calendar.advance(
            end,
            -tenor.length,
            tenor.units,
            schedule.business_day_convention,
            schedule.end_of_month,
        )
    elif (
        i == n_periods - 1
        and schedule.has_tenor()
        and schedule.has_is_regular()
        and not schedule.is_regular_at(n_periods)
    ):
        tenor = schedule.tenor
        ref_end = schedule.calendar.advance(
            start,
            tenor.length,
            tenor.units,
            schedule.business_day_convention,
            schedule.end_of_month,
        )
    return ref_start, ref_end


def fixed_rate_leg(
    schedule: Schedule,
    nominals: Sequence[float],
    rates: Sequence[float] | Sequence[InterestRate],
    day_counter: DayCounter | None = None,
    compounding: Compounding = Compounding.Simple,
    frequency: Frequency = Frequency.Annual,
    payment_adjustment: BusinessDayConvention = BusinessDayConvention.Following,
    payment_calendar: Calendar | None = None,
) -> list[CashFlow]:
    """Build a leg of FixedRateCoupons from a schedule.

    C++ parity: ql/cashflows/fixedratecoupon.cpp:176-275 ``FixedRateLeg::operator Leg()``.

    Either ``rates`` is a sequence of plain floats — in which case
    ``day_counter`` (required) + ``compounding`` + ``frequency`` define
    the InterestRate convention — or ``rates`` is a sequence of
    fully-built ``InterestRate`` instances (in which case ``day_counter``
    is ignored).

    Sequence semantics mirror C++: if the rates / nominals list is
    shorter than the number of coupon periods, the last value is
    repeated. Empty rates or empty nominals raises.
    """
    qassert.require(len(rates) > 0, "no coupon rates given")
    qassert.require(len(nominals) > 0, "no notional given")
    qassert.require(len(schedule) >= 2, "schedule has fewer than 2 dates")

    cal = payment_calendar if payment_calendar is not None else schedule.calendar
    coupon_rates: list[InterestRate] = _normalise_rates(
        rates, day_counter, compounding, frequency
    )

    leg: list[CashFlow] = []
    n_periods = len(schedule) - 1

    for i in range(n_periods):
        start = schedule.date(i)
        end = schedule.date(i + 1)
        payment_date = cal.adjust(end, payment_adjustment)
        rate = _pick_rate(coupon_rates, i)
        nominal_val = _pick_nominal(nominals, i)
        ref_start, ref_end = _compute_ref_period(schedule, n_periods, i, start, end)

        leg.append(
            FixedRateCoupon(
                payment_date,
                nominal_val,
                rate,
                start,
                end,
                ref_start,
                ref_end,
            )
        )

    return leg
