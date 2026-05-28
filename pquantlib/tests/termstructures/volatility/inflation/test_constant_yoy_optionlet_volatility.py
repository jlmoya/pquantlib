"""ConstantYoYOptionletVolatility cross-validation against C++ probe.

C++ reference values live in ``migration-harness/references/cluster/l7d.json``.

Asserts:
* the constant surface returns the configured vol regardless of ``(date, strike)``
  (Tier: EXACT — direct quote read).
* ``total_variance(date, strike) = vol^2 * time_from_base`` matches C++
  (Tier: TIGHT — multiplication of an ActualActual year fraction).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from pquantlib.daycounters.actual_actual import ActualActual
from pquantlib.daycounters.actual_actual import Convention as ActualActualConvention
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.termstructures.volatility.inflation.constant_yoy_optionlet_volatility import (
    ConstantYoYOptionletVolatility,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.testing.tolerance import exact, tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_REF_PATH = (
    Path(__file__).resolve().parents[5] / "migration-harness/references/cluster/l7d.json"
)


@pytest.fixture(scope="module")
def cluster_refs() -> dict[str, dict[str, float]]:
    return json.loads(_REF_PATH.read_text())


@pytest.fixture(autouse=True)
def _pin_eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Pin the global evaluation date to the C++ probe's date (2024-01-17)."""
    s = ObservableSettings()
    old = s.evaluation_date
    s.evaluation_date = Date.from_ymd(17, Month.January, 2024)
    yield
    s.evaluation_date = old


def _make_surface(
    vol: float = 0.005,
    volatility_type: VolatilityType = VolatilityType.ShiftedLognormal,
    displacement: float = 0.0,
) -> ConstantYoYOptionletVolatility:
    return ConstantYoYOptionletVolatility(
        vol=vol,
        settlement_days=0,
        calendar=TARGET(),
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        day_counter=ActualActual(ActualActualConvention.ISDA),
        observation_lag=Period(3, TimeUnit.Months),
        frequency=Frequency.Monthly,
        index_is_interpolated=False,
        volatility_type=volatility_type,
        displacement=displacement,
    )


def test_constant_yoy_returns_configured_vol(
    cluster_refs: dict[str, dict[str, float]],
) -> None:
    ref = cluster_refs["constant_yoy_optionlet_volatility"]
    surface = _make_surface(vol=ref["vol"])
    eval_date = Date.from_ymd(17, Month.January, 2024)
    d1 = eval_date + Period(1, TimeUnit.Years)
    d5 = eval_date + Period(5, TimeUnit.Years)
    exact(surface.volatility(d1, 0.025), ref["vol_at_1y_strike_2_5pct"])
    exact(surface.volatility(d5, 0.00), ref["vol_at_5y_strike_0pct"])


def test_constant_yoy_total_variance_matches_cpp(
    cluster_refs: dict[str, dict[str, float]],
) -> None:
    ref = cluster_refs["constant_yoy_optionlet_volatility"]
    surface = _make_surface(vol=ref["vol"])
    eval_date = Date.from_ymd(17, Month.January, 2024)
    d1 = eval_date + Period(1, TimeUnit.Years)
    d5 = eval_date + Period(5, TimeUnit.Years)
    # C++ probe passes Period(0, Days) explicitly as obs_lag override (the
    # engine path also does this). TIGHT — vol^2 * year_fraction(ActualActual::ISDA).
    zero_lag = Period(0, TimeUnit.Days)
    tight(surface.total_variance(d1, 0.025, zero_lag), ref["total_variance_1y_2_5pct"])
    tight(surface.total_variance(d5, 0.025, zero_lag), ref["total_variance_5y_2_5pct"])


def test_constant_yoy_displacement_must_be_0_or_1() -> None:
    # Construction succeeds with 0 or 1.
    _make_surface(displacement=0.0)
    _make_surface(displacement=1.0)
    # Anything else raises.
    with pytest.raises(Exception, match="displacement"):
        _make_surface(displacement=0.5)


def test_constant_yoy_volatility_type_and_displacement_inspectors() -> None:
    s_log = _make_surface(volatility_type=VolatilityType.ShiftedLognormal, displacement=1.0)
    assert s_log.volatility_type() == VolatilityType.ShiftedLognormal
    assert s_log.displacement() == 1.0
    s_normal = _make_surface(volatility_type=VolatilityType.Normal)
    assert s_normal.volatility_type() == VolatilityType.Normal
    assert s_normal.displacement() == 0.0


def test_constant_yoy_strike_limits_from_constructor() -> None:
    surface = ConstantYoYOptionletVolatility(
        vol=0.005,
        settlement_days=0,
        calendar=TARGET(),
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        day_counter=ActualActual(ActualActualConvention.ISDA),
        observation_lag=Period(3, TimeUnit.Months),
        frequency=Frequency.Monthly,
        index_is_interpolated=False,
        min_strike=-0.10,
        max_strike=0.50,
    )
    assert surface.min_strike() == -0.10
    assert surface.max_strike() == 0.50


def test_constant_yoy_max_date_is_date_max() -> None:
    surface = _make_surface()
    assert surface.max_date() == Date.max_date()
