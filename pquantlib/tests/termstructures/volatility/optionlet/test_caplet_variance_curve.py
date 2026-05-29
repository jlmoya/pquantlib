"""Tests for CapletVarianceCurve."""

from __future__ import annotations

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.termstructures.volatility.optionlet.caplet_variance_curve import (
    CapletVarianceCurve,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.testing import tolerance
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.time_unit import TimeUnit


def _new_curve() -> CapletVarianceCurve:
    ref = Date.from_ymd(15, Month.January, 2024)
    dates = [
        TARGET().advance(ref, 1, TimeUnit.Years),
        TARGET().advance(ref, 2, TimeUnit.Years),
        TARGET().advance(ref, 3, TimeUnit.Years),
    ]
    return CapletVarianceCurve(
        reference_date=ref,
        dates=dates,
        caplet_vol_curve=[0.20, 0.18, 0.16],
        day_counter=Actual365Fixed(),
    )


def test_volatility_at_node() -> None:
    curve = _new_curve()
    ref = Date.from_ymd(15, Month.January, 2024)
    d2y = TARGET().advance(ref, 2, TimeUnit.Years)
    # The black-variance-curve linearly interpolates variance and
    # then derives vol = sqrt(var/t). At a node, this returns the
    # input vol (modulo tiny rounding).
    tolerance.loose(curve.volatility(d2y, 0.04, True), 0.18)


def test_default_vol_type_and_displacement() -> None:
    curve = _new_curve()
    assert curve.volatility_type() == VolatilityType.ShiftedLognormal
    assert curve.displacement() == 0.0


def test_overrides() -> None:
    ref = Date.from_ymd(15, Month.January, 2024)
    dates = [
        TARGET().advance(ref, 1, TimeUnit.Years),
        TARGET().advance(ref, 2, TimeUnit.Years),
    ]
    curve = CapletVarianceCurve(
        reference_date=ref,
        dates=dates,
        caplet_vol_curve=[0.005, 0.006],
        day_counter=Actual365Fixed(),
        volatility_type=VolatilityType.Normal,
        displacement=0.0,
    )
    assert curve.volatility_type() == VolatilityType.Normal


def test_max_date_returns_last_pillar() -> None:
    curve = _new_curve()
    ref = Date.from_ymd(15, Month.January, 2024)
    last = TARGET().advance(ref, 3, TimeUnit.Years)
    assert curve.max_date() == last


def test_black_variance_time_scaling_at_node() -> None:
    curve = _new_curve()
    ref = Date.from_ymd(15, Month.January, 2024)
    d2y = TARGET().advance(ref, 2, TimeUnit.Years)
    v = curve.volatility(d2y, 0.04, True)
    t = Actual365Fixed().year_fraction(ref, d2y)
    tolerance.loose(curve.black_variance(d2y, 0.04, True), v * v * t)
