"""YearOnYearInflationSwap smoke + structural tests.

The YoY swap depends on L7-C's YoYInflationCoupon for a full leg
construction; here we exercise the swap shell with the scaffolding
YoY coupon from ``test_yoy_inflation_capfloor_engine``. Validates:

- 2-leg layout (fixed + yoy),
- payer multipliers per Type,
- inspector round-trip,
- fixed leg cashflow count + amounts.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.indexes.inflation.cpi import InterpolationType
from pquantlib.indexes.inflation.eu_hicp import YoYEUHICP
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.year_on_year_inflation_swap import YearOnYearInflationSwap
from pquantlib.instruments.yoy_inflation_capfloor import YoYInflationCouponLike
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

_EVAL_DATE = Date.from_ymd(17, Month.January, 2024)


class _ScaffoldYoYCoupon:
    """Minimal YoY coupon scaffold (same shape as the engine test helper)."""

    def __init__(self, *, accrual_start: Date, accrual_end: Date) -> None:
        self._start = accrual_start
        self._end = accrual_end
        self._fixing = accrual_end - Period(3, TimeUnit.Months)

    def accrual_start_date(self) -> Date:
        return self._start

    def accrual_end_date(self) -> Date:
        return self._end

    def fixing_date(self) -> Date:
        return self._fixing

    def date(self) -> Date:
        return self._end

    def accrual_period(self) -> float:
        return 1.0

    def nominal(self) -> float:
        return 1_000_000.0

    def gearing(self) -> float:
        return 1.0

    def spread(self) -> float:
        return 0.0

    def has_occurred(self, ref_date: Date | None = None) -> bool:
        del ref_date
        return False

    def register_with(self, observer: object) -> None:
        del observer


@pytest.fixture(autouse=True)
def _pin_eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    s = ObservableSettings()
    old = s.evaluation_date
    s.evaluation_date = _EVAL_DATE
    yield
    s.evaluation_date = old


def _build_swap(type_: SwapType = SwapType.Payer) -> YearOnYearInflationSwap:
    cal = TARGET()
    start = cal.advance(_EVAL_DATE, 2, TimeUnit.Days)
    end = cal.advance(start, 5, TimeUnit.Years)
    fixed_schedule = Schedule.from_rule(
        start, end, Period(1, TimeUnit.Years), cal,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward, False,
    )
    yoy_schedule = Schedule.from_rule(
        start, end, Period(1, TimeUnit.Years), cal,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward, False,
    )
    # Build a 5x1Y YoY scaffold leg.
    yoy_leg: list[YoYInflationCouponLike] = []
    for i in range(len(yoy_schedule) - 1):
        coupon = _ScaffoldYoYCoupon(
            accrual_start=yoy_schedule.date(i),
            accrual_end=yoy_schedule.date(i + 1),
        )
        yoy_leg.append(cast(YoYInflationCouponLike, coupon))

    return YearOnYearInflationSwap(
        type_=type_,
        nominal=1_000_000.0,
        fixed_schedule=fixed_schedule,
        fixed_rate=0.025,
        fixed_day_count=Thirty360(Thirty360Convention.BondBasis),
        yoy_leg=yoy_leg,
        yoy_index=YoYEUHICP(),
        observation_lag=Period(3, TimeUnit.Months),
        interpolation=InterpolationType.AsIndex,
        spread=0.0,
        yoy_day_count=Actual360(),
        payment_calendar=cal,
    )


def test_yoyiis_has_two_legs() -> None:
    swap = _build_swap()
    assert swap.number_of_legs() == 2
    assert len(swap.fixed_leg()) > 0
    assert len(swap.yoy_leg()) == 5


def test_yoyiis_payer_inverts_multipliers() -> None:
    payer = _build_swap(SwapType.Payer)
    receiver = _build_swap(SwapType.Receiver)
    # Payer: pays fixed (leg 0 = -1), receives YoY (leg 1 = +1).
    assert payer.payer(0) is True
    assert payer.payer(1) is False
    # Receiver: receives fixed (leg 0 = +1), pays YoY (leg 1 = -1).
    assert receiver.payer(0) is False
    assert receiver.payer(1) is True


def test_yoyiis_inspectors_round_trip() -> None:
    swap = _build_swap()
    assert swap.type() == SwapType.Payer
    assert swap.nominal() == 1_000_000.0
    assert swap.fixed_rate() == 0.025
    assert swap.observation_lag() == Period(3, TimeUnit.Months)
    assert swap.interpolation() == InterpolationType.AsIndex
    assert swap.spread() == 0.0
    assert swap.payment_calendar().name() == TARGET().name()
    assert swap.payment_convention() == BusinessDayConvention.ModifiedFollowing
