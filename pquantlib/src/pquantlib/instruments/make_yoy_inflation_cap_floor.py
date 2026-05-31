"""make_yoy_inflation_cap_floor — builder for standard YoY cap/floors.

# C++ parity: ql/instruments/makeyoyinflationcapfloor.{hpp,cpp} +
   ql/cashflows/yoyinflationcoupon.cpp (``yoyInflationLeg``) (v1.42.1).

PQuantLib exposes the C++ ``MakeYoYInflationCapFloor`` builder + its
fluent ``with*`` setters as a single keyword-argument factory function
(no separate fluent object). The leg is a plain YoY swaplet leg (one
``YoYInflationCoupon`` per annual period) — the cap/floor option payoff
is applied by the pricing engine, not by the coupons. This matches the
``noOption`` branch of ``yoyInflationLeg::operator Leg()``.
"""

from __future__ import annotations

from typing import cast

from pquantlib import qassert
from pquantlib.cashflows.yoy_inflation_coupon import YoYInflationCoupon
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.indexes.inflation.cpi import InterpolationType
from pquantlib.indexes.inflation.inflation_index import YoYInflationIndex
from pquantlib.instruments.yoy_inflation_capfloor import (
    YoYInflationCapFloor,
    YoYInflationCapFloorType,
    YoYInflationCouponLike,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.pricing_engine import PricingEngine
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit


def yoy_inflation_leg(
    schedule: Schedule,
    calendar: Calendar,
    index: YoYInflationIndex,
    observation_lag: Period,
    interpolation: InterpolationType,
    *,
    notional: float,
    payment_day_counter: DayCounter,
    payment_adjustment: BusinessDayConvention = BusinessDayConvention.ModifiedFollowing,
    fixing_days: int = 0,
    gearing: float = 1.0,
    spread: float = 0.0,
) -> list[YoYInflationCouponLike]:
    """Build a YoY swaplet leg over ``schedule``.

    # C++ parity: ``yoyInflationLeg::operator Leg()`` (noOption swaplet branch).
    """
    n = schedule.size() - 1
    qassert.require(n > 0, "schedule too short for a YoY leg")
    leg: list[YoYInflationCouponLike] = []
    for i in range(n):
        start = schedule.date(i)
        end = schedule.date(i + 1)
        payment_date = calendar.adjust(end, payment_adjustment)
        coupon = YoYInflationCoupon(
            payment_date=payment_date,
            nominal=notional,
            accrual_start_date=start,
            accrual_end_date=end,
            fixing_days=fixing_days,
            index=index,
            observation_lag=observation_lag,
            interpolation=interpolation,
            day_counter=payment_day_counter,
            gearing=gearing,
            spread=spread,
            ref_period_start=start,
            ref_period_end=end,
        )
        # YoYInflationCoupon structurally satisfies YoYInflationCouponLike;
        # the Protocol types register_with(observer: object) (broader) while
        # the concrete narrows it to Observer — an accepted contravariance
        # mismatch under pyright, so we cast.
        leg.append(cast(YoYInflationCouponLike, coupon))
    return leg


def make_yoy_inflation_cap_floor(
    cap_floor_type: YoYInflationCapFloorType,
    index: YoYInflationIndex,
    length: int,
    calendar: Calendar,
    observation_lag: Period,
    interpolation: InterpolationType,
    *,
    strike: float,
    nominal: float = 1_000_000.0,
    effective_date: Date | None = None,
    fixing_days: int = 0,
    forward_start: Period | None = None,
    payment_day_counter: DayCounter | None = None,
    payment_adjustment: BusinessDayConvention = BusinessDayConvention.ModifiedFollowing,
    first_caplet_excluded: bool = False,
    as_optionlet: bool = False,
    pricing_engine: PricingEngine | None = None,
) -> YoYInflationCapFloor:
    """Build a standard YoY inflation cap/floor.

    # C++ parity: ``MakeYoYInflationCapFloor::operator
    # ext::shared_ptr<YoYInflationCapFloor>()`` (makeyoyinflationcapfloor.cpp:46-89).
    """
    if effective_date is not None:
        start_date = effective_date
    else:
        ref = ObservableSettings().evaluation_date_or_today()
        spot = calendar.advance(ref, fixing_days, TimeUnit.Days)
        start_date = spot + forward_start if forward_start is not None else spot

    end_date = calendar.advance(
        start_date, length, TimeUnit.Years, BusinessDayConvention.Unadjusted
    )
    schedule = Schedule.from_rule(
        start_date,
        end_date,
        Period.from_frequency(Frequency.Annual),
        calendar,
        BusinessDayConvention.Unadjusted,
        BusinessDayConvention.Unadjusted,
        DateGeneration.Forward,
        False,
    )
    dc = (
        payment_day_counter
        if payment_day_counter is not None
        else Thirty360(Thirty360Convention.BondBasis)
    )
    leg = yoy_inflation_leg(
        schedule,
        calendar,
        index,
        observation_lag,
        interpolation,
        notional=nominal,
        payment_day_counter=dc,
        payment_adjustment=payment_adjustment,
        fixing_days=fixing_days,
    )

    if first_caplet_excluded:
        leg = leg[1:]
    if as_optionlet and len(leg) > 1:
        leg = leg[-1:]

    cap_floor = YoYInflationCapFloor.from_strikes(cap_floor_type, leg, [strike])
    if pricing_engine is not None:
        cap_floor.set_pricing_engine(pricing_engine)
    return cap_floor
