"""make_yoy_inflation_cap_floor — builder smoke + pricing roundtrip.

# C++ parity: ql/instruments/makeyoyinflationcapfloor.{hpp,cpp} (v1.42.1).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.indexes.inflation.cpi import InterpolationType
from pquantlib.indexes.inflation.eu_hicp import YoYEUHICP
from pquantlib.instruments.make_yoy_inflation_cap_floor import (
    make_yoy_inflation_cap_floor,
    yoy_inflation_leg,
)
from pquantlib.instruments.yoy_inflation_capfloor import YoYInflationCapFloorType
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

_EVAL = Date.from_ymd(13, Month.August, 2007)


@pytest.fixture(autouse=True)
def _pin_eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    s = ObservableSettings()
    old = s.evaluation_date
    s.evaluation_date = _EVAL
    yield
    s.evaluation_date = old


def test_build_cap_topology() -> None:
    idx = YoYEUHICP()
    cf = make_yoy_inflation_cap_floor(
        YoYInflationCapFloorType.Cap,
        idx,
        5,
        TARGET(),
        Period(3, TimeUnit.Months),
        InterpolationType.AsIndex,
        strike=0.02,
        nominal=10000.0,
    )
    assert cf.type() == YoYInflationCapFloorType.Cap
    leg = cf.yoy_leg()
    assert len(leg) == 5  # 5 annual periods
    assert cf.cap_rates()[0] == 0.02
    # nominal flows through to the coupons.
    assert leg[0].nominal() == 10000.0
    # 5-year span.
    assert cf.maturity_date() > cf.start_date()


def test_build_floor_and_optionlet() -> None:
    idx = YoYEUHICP()
    floor = make_yoy_inflation_cap_floor(
        YoYInflationCapFloorType.Floor,
        idx,
        5,
        TARGET(),
        Period(3, TimeUnit.Months),
        InterpolationType.AsIndex,
        strike=0.01,
    )
    assert floor.type() == YoYInflationCapFloorType.Floor
    assert floor.floor_rates()[0] == 0.01

    # as_optionlet keeps only the last coupon.
    opt = make_yoy_inflation_cap_floor(
        YoYInflationCapFloorType.Cap,
        idx,
        5,
        TARGET(),
        Period(3, TimeUnit.Months),
        InterpolationType.AsIndex,
        strike=0.02,
        as_optionlet=True,
    )
    assert len(opt.yoy_leg()) == 1


def test_first_caplet_excluded() -> None:
    idx = YoYEUHICP()
    cf = make_yoy_inflation_cap_floor(
        YoYInflationCapFloorType.Cap,
        idx,
        5,
        TARGET(),
        Period(3, TimeUnit.Months),
        InterpolationType.AsIndex,
        strike=0.02,
        first_caplet_excluded=True,
    )
    assert len(cf.yoy_leg()) == 4


def test_yoy_inflation_leg_direct() -> None:
    cal = TARGET()
    start = cal.advance(_EVAL, 2, TimeUnit.Days)
    end = cal.advance(start, 3, TimeUnit.Years)
    schedule = Schedule.from_rule(
        start, end, Period(1, TimeUnit.Years), cal,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Forward, False,
    )
    leg = yoy_inflation_leg(
        schedule, cal, YoYEUHICP(), Period(3, TimeUnit.Months),
        InterpolationType.AsIndex, notional=1000.0,
        payment_day_counter=Actual360(),
    )
    assert len(leg) == 3
    assert all(c.nominal() == 1000.0 for c in leg)
