"""Tests for SmileSection abstract + FlatSmileSection concrete leaf."""

from __future__ import annotations

import math

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.termstructures.volatility.flat_smile_section import FlatSmileSection
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.month import Month

_REF = reference_reader.load("cluster/l2e")
_FLAT = _REF["flat_smile_section"]


def test_cannot_instantiate_abstract() -> None:
    with pytest.raises(TypeError):
        SmileSection(exercise_time=1.0)  # type: ignore[abstract]


def test_flat_smile_section_date_anchored_construction() -> None:
    """C++ cross-validation: exercise_time of 1y under Actual/365 Fixed = 1.0."""
    ref = Date.from_ymd(15, Month.June, 2026)
    exercise = Date.from_ymd(15, Month.June, 2027)
    smile = FlatSmileSection(
        exercise_date=exercise,
        volatility=0.20,
        day_counter=Actual365Fixed(),
        reference_date=ref,
        atm_level=100.0,
    )
    tolerance.exact(smile.exercise_time(), _FLAT["exercise_time"])
    tolerance.exact(smile.atm_level(), _FLAT["atm_level"])
    tolerance.exact(smile.shift(), _FLAT["shift"])


def test_flat_smile_section_vol_is_flat_across_strikes() -> None:
    smile = FlatSmileSection(
        exercise_time=1.0,
        volatility=0.20,
        atm_level=100.0,
    )
    tolerance.exact(smile.volatility(100.0), _FLAT["vol_at_strike_100"])
    tolerance.exact(smile.volatility(120.0), _FLAT["vol_at_strike_120"])
    tolerance.exact(smile.volatility(50.0), _FLAT["vol_at_strike_100"])


def test_flat_smile_section_variance() -> None:
    """variance = vol^2 * exercise_time."""
    smile = FlatSmileSection(
        exercise_time=1.0,
        volatility=0.20,
        atm_level=100.0,
    )
    tolerance.tight(smile.variance(100.0), _FLAT["variance_at_strike_100"])


def test_flat_smile_section_min_max_strike() -> None:
    smile = FlatSmileSection(
        exercise_time=1.0,
        volatility=0.20,
        atm_level=100.0,
    )
    assert smile.min_strike() == -math.inf
    assert smile.max_strike() == math.inf


def test_flat_smile_section_atm_level_can_be_unset() -> None:
    smile = FlatSmileSection(exercise_time=1.0, volatility=0.20)
    assert math.isnan(smile.atm_level())


def test_flat_smile_section_default_volatility_type_is_shifted_lognormal() -> None:
    smile = FlatSmileSection(exercise_time=1.0, volatility=0.20)
    assert smile.volatility_type() == VolatilityType.ShiftedLognormal


def test_flat_smile_section_normal_volatility_type_round_trip() -> None:
    smile = FlatSmileSection(
        exercise_time=1.0,
        volatility=0.20,
        volatility_type=VolatilityType.Normal,
    )
    assert smile.volatility_type() == VolatilityType.Normal


def test_flat_smile_section_with_shift() -> None:
    smile = FlatSmileSection(
        exercise_time=1.0,
        volatility=0.20,
        shift=0.05,
    )
    assert smile.shift() == 0.05


def test_smile_section_date_mode_requires_reference_date() -> None:
    with pytest.raises(LibraryException, match="reference_date"):
        FlatSmileSection(
            exercise_date=Date.from_ymd(15, Month.June, 2027),
            volatility=0.20,
            day_counter=Actual365Fixed(),
            atm_level=100.0,
        )


def test_smile_section_date_mode_requires_day_counter() -> None:
    with pytest.raises(LibraryException, match="day_counter"):
        FlatSmileSection(
            exercise_date=Date.from_ymd(15, Month.June, 2027),
            volatility=0.20,
            reference_date=Date.from_ymd(15, Month.June, 2026),
            atm_level=100.0,
        )


def test_smile_section_date_mode_requires_exercise_ge_reference() -> None:
    with pytest.raises(LibraryException, match="must be greater than or equal"):
        FlatSmileSection(
            exercise_date=Date.from_ymd(15, Month.June, 2025),
            volatility=0.20,
            day_counter=Actual365Fixed(),
            reference_date=Date.from_ymd(15, Month.June, 2026),
            atm_level=100.0,
        )


def test_smile_section_time_mode_requires_non_negative_time() -> None:
    with pytest.raises(LibraryException, match="non-negative"):
        FlatSmileSection(exercise_time=-1.0, volatility=0.20)


def test_smile_section_time_mode_requires_exercise_time_argument() -> None:
    with pytest.raises(LibraryException, match="exercise_time"):
        FlatSmileSection(volatility=0.20)


def test_smile_section_update_notifies_observers() -> None:
    smile = FlatSmileSection(exercise_time=1.0, volatility=0.20)
    counts = [0]

    class _Counter:
        def update(self) -> None:
            counts[0] += 1

    obs = _Counter()
    smile.register_with(obs)
    smile.update()
    assert counts[0] == 1


def test_smile_section_exercise_date_unavailable_in_time_mode() -> None:
    smile = FlatSmileSection(exercise_time=1.0, volatility=0.20)
    with pytest.raises(LibraryException, match="exercise date not available"):
        smile.exercise_date()


def test_smile_section_reference_date_unavailable_in_time_mode() -> None:
    smile = FlatSmileSection(exercise_time=1.0, volatility=0.20)
    with pytest.raises(LibraryException, match="referenceDate not available"):
        smile.reference_date()


def test_smile_section_day_counter_unavailable_in_time_mode_when_unspecified() -> None:
    smile = FlatSmileSection(exercise_time=1.0, volatility=0.20)
    with pytest.raises(LibraryException, match="day counter not available"):
        smile.day_counter()


def test_smile_section_day_counter_available_in_time_mode_when_supplied() -> None:
    dc = Actual365Fixed()
    smile = FlatSmileSection(exercise_time=1.0, volatility=0.20, day_counter=dc)
    assert smile.day_counter() is dc
