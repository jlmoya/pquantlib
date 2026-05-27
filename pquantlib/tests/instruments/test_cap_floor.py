"""CapFloor / Cap / Floor / Collar smoke tests.

Builds the 5y cap on Euribor3M @ 4% used by the C++ probe, validates
construction + ``setup_arguments`` field population. Engine-backed NPV
cross-validation lives in the engine tests (``test_black_capfloor_engine.py``
etc.).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

import pytest

from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.coupon_pricer import IborCouponPricer, set_coupon_pricer
from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
from pquantlib.cashflows.ibor_leg import ibor_leg
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.instruments.cap_floor import (
    NULL_RATE,
    Cap,
    CapFloor,
    CapFloorArguments,
    CapFloorType,
    Collar,
    Floor,
)
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

_REF_PATH = (
    Path(__file__).resolve().parents[3] / "migration-harness/references/cluster/l4e.json"
)


@pytest.fixture(scope="module")
def cluster_refs() -> dict[str, dict[str, float]]:
    return json.loads(_REF_PATH.read_text())


def _five_year_cap_leg() -> list[CashFlow]:
    """Build the floating leg used by the probe (5y, Euribor3M, 1M EUR nominal)."""
    eval_date = Date.from_ymd(17, Month.January, 2024)
    curve = cast(
        YieldTermStructureProtocol,
        FlatForward.from_rate(
            eval_date, 0.05, Actual360(), Compounding.Continuous, Frequency.Annual
        ),
    )
    cal = TARGET()
    start = cal.advance(eval_date, 2, TimeUnit.Days)
    end = cal.advance(start, 5, TimeUnit.Years)
    idx = Euribor(Period(3, TimeUnit.Months), curve)
    schedule = Schedule.from_rule(
        start, end, Period(3, TimeUnit.Months), cal,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward, False,
    )
    leg = ibor_leg(
        schedule, idx, [1_000_000.0],
        payment_adjustment=BusinessDayConvention.ModifiedFollowing,
        fixing_days=2,
    )
    # The cap engine path needs IborCoupon.adjusted_fixing(), which
    # delegates to a pricer. Attach the default IborCouponPricer (the
    # same plain-IBOR pricer VanillaSwap wires on its floating leg).
    set_coupon_pricer(leg, IborCouponPricer())
    return leg


# --- Construction --------------------------------------------------------


def test_cap_constructs(cluster_refs: dict[str, dict[str, float]]) -> None:
    leg = _five_year_cap_leg()
    cap = Cap(leg, [0.04])
    expected = cluster_refs["cap_setup"]
    assert cap.type() == CapFloorType.Cap
    assert len(cap.floating_leg()) == expected["leg_size"]
    # Strike repeated to leg length.
    assert len(cap.cap_rates()) == len(leg)
    assert all(r == 0.04 for r in cap.cap_rates())
    assert cap.floor_rates() == []


def test_floor_constructs() -> None:
    leg = _five_year_cap_leg()
    floor = Floor(leg, [0.04])
    assert floor.type() == CapFloorType.Floor
    assert floor.cap_rates() == []
    assert len(floor.floor_rates()) == len(leg)


def test_collar_constructs() -> None:
    leg = _five_year_cap_leg()
    collar = Collar(leg, cap_rates=[0.05], floor_rates=[0.03])
    assert collar.type() == CapFloorType.Collar
    assert len(collar.cap_rates()) == len(leg)
    assert len(collar.floor_rates()) == len(leg)


def test_cap_from_strikes_classmethod() -> None:
    """``CapFloor.from_strikes`` matches the C++ 3-arg constructor."""
    leg = _five_year_cap_leg()
    cap = CapFloor.from_strikes(CapFloorType.Cap, leg, [0.04])
    assert cap.type() == CapFloorType.Cap
    assert all(r == 0.04 for r in cap.cap_rates())


def test_cap_from_strikes_rejects_collar() -> None:
    leg = _five_year_cap_leg()
    with pytest.raises(Exception, match="only Cap/Floor"):
        CapFloor.from_strikes(CapFloorType.Collar, leg, [0.04])


def test_cap_requires_at_least_one_strike() -> None:
    leg = _five_year_cap_leg()
    with pytest.raises(Exception, match="no cap rates"):
        CapFloor(CapFloorType.Cap, leg, cap_rates=[])


def test_floor_requires_at_least_one_strike() -> None:
    leg = _five_year_cap_leg()
    with pytest.raises(Exception, match="no floor rates"):
        CapFloor(CapFloorType.Floor, leg, floor_rates=[])


# --- start_date / maturity_date -----------------------------------------


def test_cap_start_and_maturity_dates_match_leg() -> None:
    leg = _five_year_cap_leg()
    cap = Cap(leg, [0.04])
    # Start should be the first cashflow's accrual_start_date.
    first = leg[0]
    assert isinstance(first, FloatingRateCoupon)
    assert cap.start_date() == first.accrual_start_date()
    # Maturity = last cashflow's date().
    assert cap.maturity_date() == leg[-1].date()


# --- setup_arguments populates the engine carrier -----------------------


def test_cap_setup_arguments_populates_carrier(
    cluster_refs: dict[str, dict[str, float]],
) -> None:
    leg = _five_year_cap_leg()
    cap = Cap(leg, [0.04])
    args = CapFloorArguments()
    cap.setup_arguments(args)
    expected = cluster_refs["cap_setup"]
    n = expected["leg_size"]

    assert args.type == CapFloorType.Cap
    assert len(args.start_dates) == n
    assert len(args.end_dates) == n
    assert len(args.fixing_dates) == n
    assert len(args.accrual_times) == n
    assert len(args.cap_rates) == n
    # Floor on a Cap stays as null/NaN sentinels.
    assert len(args.floor_rates) == n
    # All accrual times should be positive (5y/4 ≈ 0.25y).
    assert all(t > 0.0 for t in args.accrual_times)
    # Nominals broadcast to 1M.
    assert all(nom == 1_000_000.0 for nom in args.nominals)
    # Cap strikes = (strike - spread) / gearing = 0.04 (zero spread, unit gearing).
    assert all(r == 0.04 for r in args.cap_rates)
    # Floor rates remain NULL (NaN) sentinels on a pure Cap.
    for r in args.floor_rates:
        assert math.isnan(r)


def test_floor_setup_arguments_leaves_cap_rates_null() -> None:
    leg = _five_year_cap_leg()
    floor = Floor(leg, [0.04])
    args = CapFloorArguments()
    floor.setup_arguments(args)
    for r in args.cap_rates:
        assert math.isnan(r)


def test_cap_optionlet_extracts_single_period() -> None:
    leg = _five_year_cap_leg()
    cap = Cap(leg, [0.04])
    opt = cap.optionlet(0)
    assert opt.type() == CapFloorType.Cap
    assert len(opt.floating_leg()) == 1
    assert opt.floating_leg()[0] is leg[0]


def test_cap_optionlet_out_of_range_rejected() -> None:
    leg = _five_year_cap_leg()
    cap = Cap(leg, [0.04])
    with pytest.raises(Exception, match="does not exist"):
        cap.optionlet(len(leg))


def test_null_rate_sentinel_is_nan() -> None:
    """``NULL_RATE`` is the marker stored in the unused-strike slots."""
    assert math.isnan(NULL_RATE)
