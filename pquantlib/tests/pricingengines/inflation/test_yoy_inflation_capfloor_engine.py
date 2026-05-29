"""YoY inflation Black / Bachelier / UnitDisplaced engines cross-validation.

C++ reference values live in ``migration-harness/references/cluster/l7d.json``.

The probe builds:

- a flat YoY curve at 2.5% on the index,
- a flat 3% nominal discount curve,
- a 5y annual YoY cap (or floor) at 2.5% strike on YYEUHICP,
- and prices it with each of the 3 engines (Black @ vol=20%, Bachelier
  @ vol=0.5%, UnitDisplacedBlack @ vol=20%).

These tests mirror the same setup using the L7-D engine + scaffolding
``_FlatYoYTermStructure`` (a minimal YoYInflationTermStructure concrete
the engine uses until L7-B's curves land). LOOSE tier for the NPV: the
chain of accrual computations + ActualActual::ISDA year-fraction
arithmetic propagates ~1e-13 rounding; LOOSE captures that comfortably.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.actual_actual import ActualActual
from pquantlib.daycounters.actual_actual import Convention as ActualActualConvention
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.inflation.eu_hicp import YoYEUHICP
from pquantlib.instruments.yoy_inflation_capfloor import (
    YoYInflationCap,
    YoYInflationCapFloorArguments,
    YoYInflationCapFloorType,
    YoYInflationCollar,
    YoYInflationCouponLike,
    YoYInflationFloor,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.inflation.yoy_inflation_capfloor_engine import (
    YoYInflationBachelierCapFloorEngine,
    YoYInflationBlackCapFloorEngine,
    YoYInflationUnitDisplacedBlackCapFloorEngine,
)
from pquantlib.termstructures.inflation.yoy_inflation_term_structure import (
    YoYInflationTermStructure,
)
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.volatility.inflation.constant_yoy_optionlet_volatility import (
    ConstantYoYOptionletVolatility,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.tolerance import loose
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
    Path(__file__).resolve().parents[4] / "migration-harness/references/cluster/l7d.json"
)
_EVAL_DATE = Date.from_ymd(17, Month.January, 2024)


# ---------------------------------------------------------------------------
# Scaffolding: a flat-rate YoY term structure + a minimal YoY coupon.
#
# These stand in for L7-B's InterpolatedYoYInflationCurve + L7-C's
# YoYInflationCoupon. Once those clusters merge, the scaffolds remain as
# test-only helpers and the engines are usable with the production types.
# ---------------------------------------------------------------------------


class _FlatYoYTermStructure(YoYInflationTermStructure):
    """Minimal flat-rate YoYInflationTermStructure.

    # Used as a stand-in for L7-B's InterpolatedYoYInflationCurve / Piecewise
    # variants for L7-D engine tests. Always returns the constructor rate.
    """

    def __init__(self, *, base_date: Date, rate: float) -> None:
        super().__init__(
            base_date=base_date,
            base_yoy_rate=rate,
            frequency=Frequency.Monthly,
            day_counter=ActualActual(ActualActualConvention.ISDA),
            observation_lag=Period(3, TimeUnit.Months),
            reference_date=_EVAL_DATE,
            calendar=TARGET(),
        )
        self._rate = rate

    def _yoy_rate_impl(self, t: float) -> float:
        del t
        return self._rate

    def max_date(self) -> Date:
        return Date.max_date()


class _ScaffoldYoYCoupon:
    """Minimal YoY coupon that satisfies ``YoYInflationCouponLike``.

    # Carries the fields the YoY cap/floor engine reads (accrual dates,
    # fixing date, accrual period, nominal, gearing, spread). Date/payment
    # computations are precomputed by the test builder so the engine can
    # just read them.
    """

    def __init__(
        self,
        *,
        accrual_start: Date,
        accrual_end: Date,
        payment_date: Date,
        fixing_date: Date,
        accrual_period: float,
        nominal: float,
        gearing: float = 1.0,
        spread: float = 0.0,
    ) -> None:
        self._accrual_start = accrual_start
        self._accrual_end = accrual_end
        self._payment_date = payment_date
        self._fixing_date = fixing_date
        self._accrual_period = accrual_period
        self._nominal = nominal
        self._gearing = gearing
        self._spread = spread

    def accrual_start_date(self) -> Date:
        return self._accrual_start

    def accrual_end_date(self) -> Date:
        return self._accrual_end

    def fixing_date(self) -> Date:
        return self._fixing_date

    def date(self) -> Date:
        return self._payment_date

    def accrual_period(self) -> float:
        return self._accrual_period

    def nominal(self) -> float:
        return self._nominal

    def gearing(self) -> float:
        return self._gearing

    def spread(self) -> float:
        return self._spread

    def has_occurred(self, ref_date: Date | None = None) -> bool:
        if ref_date is None:
            return False
        return self._payment_date <= ref_date

    def register_with(self, observer: object) -> None:
        del observer


def _build_yoy_leg(
    *,
    nominal: float,
    payment_day_counter: DayCounter,
) -> list[YoYInflationCouponLike]:
    """Build the 5y annual YoY leg the C++ probe uses.

    Mirrors the probe's ``yoyInflationLeg(yoySchedule, cal, yyeu, obsLag,
    CPI::AsIndex)`` chain: 5 annual coupons starting 2 business days after
    the eval date.
    """
    cal = TARGET()
    start = cal.advance(_EVAL_DATE, 2, TimeUnit.Days)
    end = cal.advance(start, 5, TimeUnit.Years)
    schedule = Schedule.from_rule(
        start, end, Period(1, TimeUnit.Years), cal,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward, False,
    )
    # # C++ parity: yoyInflationLeg + .withFixingDays(0) — the YoY fixing
    # # date == accrual start; observation lag is folded into the engine's
    # # forward fetch via the index's curve. (The probe explicitly uses
    # # the default fixingDays which the C++ leg builder resolves to 0.)
    # # C++ parity: YoYInflationCoupon.fixing_date = accrual_end - observation_lag
    # # (via InflationCoupon::fixingDate at inflationcoupon.cpp:87-92, with
    # # fixingDays = 0 default in yoyInflationLeg).
    obs_lag = Period(3, TimeUnit.Months)
    leg: list[YoYInflationCouponLike] = []
    for i in range(len(schedule) - 1):
        accrual_start = schedule.date(i)
        accrual_end = schedule.date(i + 1)
        payment_date = cal.adjust(
            accrual_end, BusinessDayConvention.ModifiedFollowing
        )
        accrual_period = payment_day_counter.year_fraction(
            accrual_start, accrual_end
        )
        # # InflationCoupon::fixingDate: NullCalendar.advance(refEnd - obsLag, ...).
        # # With fixingDays=0 the result is just (refEnd - obsLag).
        fixing_date = accrual_end - obs_lag
        coupon = _ScaffoldYoYCoupon(
            accrual_start=accrual_start,
            accrual_end=accrual_end,
            payment_date=payment_date,
            fixing_date=fixing_date,
            accrual_period=accrual_period,
            nominal=nominal,
        )
        leg.append(cast(YoYInflationCouponLike, coupon))
    return leg


# ---------------------------------------------------------------------------
# Fixtures + test cases.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cluster_refs() -> dict[str, dict[str, float]]:
    return json.loads(_REF_PATH.read_text())


@pytest.fixture(autouse=True)
def _pin_eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    s = ObservableSettings()
    old = s.evaluation_date
    s.evaluation_date = _EVAL_DATE
    yield
    s.evaluation_date = old


def _nominal_curve() -> YieldTermStructureProtocol:
    return cast(
        YieldTermStructureProtocol,
        FlatForward.from_rate(
            _EVAL_DATE, 0.03, Actual365Fixed(), Compounding.Continuous, Frequency.Annual
        ),
    )


def _vol_surface(vol: float, vol_type: VolatilityType, displacement: float = 0.0):
    return ConstantYoYOptionletVolatility(
        vol=vol,
        settlement_days=0,
        calendar=TARGET(),
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        day_counter=Actual360(),
        observation_lag=Period(3, TimeUnit.Months),
        frequency=Frequency.Monthly,
        index_is_interpolated=False,
        volatility_type=vol_type,
        displacement=displacement,
    )


def _yoy_index() -> YoYEUHICP:
    # Build a flat YoY curve at 2.5% and attach to YoYEUHICP.
    yoy_curve = _FlatYoYTermStructure(
        base_date=_EVAL_DATE - Period(3, TimeUnit.Months),
        rate=0.025,
    )
    return YoYEUHICP(ts=yoy_curve)


def test_yoy_inflation_black_cap_npv(
    cluster_refs: dict[str, dict[str, float]],
) -> None:
    ref = cluster_refs["yoy_inflation_black_cap_5y_2_5pct"]
    leg = _build_yoy_leg(nominal=ref["nominal"], payment_day_counter=Actual360())
    assert len(leg) == int(ref["leg_size"])
    cap = YoYInflationCap(leg, [ref["strike"]])
    vol_surface = _vol_surface(ref["vol"], VolatilityType.ShiftedLognormal)
    eng = YoYInflationBlackCapFloorEngine(_yoy_index(), vol_surface, _nominal_curve())
    cap.set_pricing_engine(eng)
    loose(
        cap.npv(),
        ref["npv"],
        reason="YoY Black cap NPV: chained accrual + discount math vs C++.",
    )


def test_yoy_inflation_bachelier_cap_npv(
    cluster_refs: dict[str, dict[str, float]],
) -> None:
    ref = cluster_refs["yoy_inflation_bachelier_cap_5y_2_5pct"]
    other = cluster_refs["yoy_inflation_black_cap_5y_2_5pct"]
    leg = _build_yoy_leg(nominal=other["nominal"], payment_day_counter=Actual360())
    cap = YoYInflationCap(leg, [other["strike"]])
    vol_surface = _vol_surface(ref["vol"], VolatilityType.Normal)
    eng = YoYInflationBachelierCapFloorEngine(_yoy_index(), vol_surface, _nominal_curve())
    cap.set_pricing_engine(eng)
    loose(
        cap.npv(),
        ref["npv"],
        reason="YoY Bachelier cap NPV vs C++.",
    )


def test_yoy_inflation_unit_displaced_black_cap_npv(
    cluster_refs: dict[str, dict[str, float]],
) -> None:
    ref = cluster_refs["yoy_inflation_unit_displaced_black_cap_5y_2_5pct"]
    other = cluster_refs["yoy_inflation_black_cap_5y_2_5pct"]
    leg = _build_yoy_leg(nominal=other["nominal"], payment_day_counter=Actual360())
    cap = YoYInflationCap(leg, [other["strike"]])
    vol_surface = _vol_surface(
        ref["vol"], VolatilityType.ShiftedLognormal, displacement=1.0
    )
    eng = YoYInflationUnitDisplacedBlackCapFloorEngine(
        _yoy_index(), vol_surface, _nominal_curve()
    )
    cap.set_pricing_engine(eng)
    loose(
        cap.npv(),
        ref["npv"],
        reason="YoY UnitDisplacedBlack cap NPV: large-displacement shift dominates intrinsic.",
    )


def test_yoy_inflation_black_floor_npv(
    cluster_refs: dict[str, dict[str, float]],
) -> None:
    ref = cluster_refs["yoy_inflation_black_floor_5y_2_5pct"]
    other = cluster_refs["yoy_inflation_black_cap_5y_2_5pct"]
    leg = _build_yoy_leg(nominal=other["nominal"], payment_day_counter=Actual360())
    floor_inst = YoYInflationFloor(leg, [other["strike"]])
    vol_surface = _vol_surface(ref["vol"], VolatilityType.ShiftedLognormal)
    eng = YoYInflationBlackCapFloorEngine(_yoy_index(), vol_surface, _nominal_curve())
    floor_inst.set_pricing_engine(eng)
    loose(
        floor_inst.npv(),
        ref["npv"],
        reason="YoY Black floor NPV (put-call parity at strike == forward yields cap == floor).",
    )


def test_yoy_collar_equals_cap_minus_floor_at_same_strike(
    cluster_refs: dict[str, dict[str, float]],
) -> None:
    """Collar(K, K) = Cap(K) - Floor(K). Pure structural identity."""
    ref = cluster_refs["yoy_inflation_black_cap_5y_2_5pct"]
    leg = _build_yoy_leg(nominal=ref["nominal"], payment_day_counter=Actual360())
    vol_surface = _vol_surface(ref["vol"], VolatilityType.ShiftedLognormal)
    nominal_curve = _nominal_curve()
    yoy_index = _yoy_index()

    cap = YoYInflationCap(leg, [ref["strike"]])
    cap.set_pricing_engine(
        YoYInflationBlackCapFloorEngine(yoy_index, vol_surface, nominal_curve)
    )

    floor_inst = YoYInflationFloor(leg, [ref["strike"]])
    floor_inst.set_pricing_engine(
        YoYInflationBlackCapFloorEngine(yoy_index, vol_surface, nominal_curve)
    )

    collar = YoYInflationCollar(leg, [ref["strike"]], [ref["strike"]])
    collar.set_pricing_engine(
        YoYInflationBlackCapFloorEngine(yoy_index, vol_surface, nominal_curve)
    )

    loose(
        collar.npv(),
        cap.npv() - floor_inst.npv(),
        reason="Collar = Cap - Floor at the same strike (long cap, short floor).",
    )


def test_yoy_capfloor_setup_arguments_populates_fields() -> None:
    leg = _build_yoy_leg(nominal=1_000_000.0, payment_day_counter=Actual360())
    cap = YoYInflationCap(leg, [0.025])
    args = YoYInflationCapFloorArguments()
    cap.setup_arguments(args)
    args.validate()
    assert args.type == YoYInflationCapFloorType.Cap
    assert len(args.start_dates) == len(leg)
    assert len(args.cap_rates) == len(leg)
    # All floor rates are NULL (NaN) on a pure cap.
    assert all(math.isnan(r) for r in args.floor_rates)
    # cap rate is (strike - spread) / gearing = 0.025 (since spread=0, gearing=1).
    for r in args.cap_rates:
        assert math.isclose(r, 0.025, abs_tol=1e-14, rel_tol=1e-12)
