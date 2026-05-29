"""Tests for SabrSmileSection (closed-form Hagan 2002 vol)."""

from __future__ import annotations

import math

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.math.interpolations.sabr_formula import sabr_volatility
from pquantlib.termstructures.volatility.sabr_smile_section import SabrSmileSection
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.month import Month

_REF = reference_reader.load("cluster/l9c")
_SABR_SEC = _REF["sabr_smile_section"]


def _params() -> tuple[float, float, float, float]:
    return (_SABR_SEC["alpha"], _SABR_SEC["beta"], _SABR_SEC["nu"], _SABR_SEC["rho"])


# --- construction ------------------------------------------------------


def test_sabr_smile_section_time_anchored_construction() -> None:
    section = SabrSmileSection(
        forward=_SABR_SEC["atm_level"],
        sabr_params=_params(),
        exercise_time=_SABR_SEC["exercise_time"],
    )
    tolerance.exact(section.exercise_time(), _SABR_SEC["exercise_time"])
    tolerance.exact(section.atm_level(), _SABR_SEC["atm_level"])
    tolerance.exact(section.alpha(), _SABR_SEC["alpha"])
    tolerance.exact(section.beta(), _SABR_SEC["beta"])
    tolerance.exact(section.nu(), _SABR_SEC["nu"])
    tolerance.exact(section.rho(), _SABR_SEC["rho"])


def test_sabr_smile_section_date_anchored_uses_default_dc() -> None:
    """C++ default day-counter for the date overload is Actual365Fixed."""
    ref = Date.from_ymd(15, Month.June, 2026)
    exercise = Date.from_ymd(15, Month.June, 2027)
    section = SabrSmileSection(
        forward=0.05,
        sabr_params=(0.04, 0.5, 0.4, -0.1),
        exercise_date=exercise,
        reference_date=ref,
    )
    # Actual/365 Fixed on a 365-day span = exactly 1.0.
    tolerance.tight(section.exercise_time(), 1.0)


def test_sabr_smile_section_date_anchored_custom_dc() -> None:
    ref = Date.from_ymd(15, Month.June, 2026)
    exercise = Date.from_ymd(15, Month.June, 2027)
    section = SabrSmileSection(
        forward=0.05,
        sabr_params=(0.04, 0.5, 0.4, -0.1),
        exercise_date=exercise,
        reference_date=ref,
        day_counter=Actual365Fixed(),
    )
    tolerance.tight(section.exercise_time(), 1.0)


# --- volatility / variance --------------------------------------------


def test_sabr_smile_section_volatility_matches_direct_sabr_call() -> None:
    section = SabrSmileSection(
        forward=_SABR_SEC["atm_level"],
        sabr_params=_params(),
        exercise_time=_SABR_SEC["exercise_time"],
    )
    direct = sabr_volatility(
        _SABR_SEC["atm_level"], _SABR_SEC["atm_level"],
        _SABR_SEC["exercise_time"],
        *_params(),
    )
    tolerance.exact(section.volatility(_SABR_SEC["atm_level"]), direct)


def test_sabr_smile_section_vol_at_atm_matches_probe() -> None:
    section = SabrSmileSection(
        forward=_SABR_SEC["atm_level"],
        sabr_params=_params(),
        exercise_time=_SABR_SEC["exercise_time"],
    )
    tolerance.tight(section.volatility(_SABR_SEC["atm_level"]), _SABR_SEC["vol_atm"])


def test_sabr_smile_section_vol_at_strike_4pct_matches_probe() -> None:
    section = SabrSmileSection(
        forward=_SABR_SEC["atm_level"],
        sabr_params=_params(),
        exercise_time=_SABR_SEC["exercise_time"],
    )
    tolerance.tight(section.volatility(0.04), _SABR_SEC["vol_strike_4pct"])


def test_sabr_smile_section_vol_at_strike_6pct_matches_probe() -> None:
    section = SabrSmileSection(
        forward=_SABR_SEC["atm_level"],
        sabr_params=_params(),
        exercise_time=_SABR_SEC["exercise_time"],
    )
    tolerance.tight(section.volatility(0.06), _SABR_SEC["vol_strike_6pct"])


def test_sabr_smile_section_variance_at_atm() -> None:
    section = SabrSmileSection(
        forward=_SABR_SEC["atm_level"],
        sabr_params=_params(),
        exercise_time=_SABR_SEC["exercise_time"],
    )
    tolerance.tight(section.variance(_SABR_SEC["atm_level"]), _SABR_SEC["variance_atm"])


# --- min/max strike ----------------------------------------------------


def test_sabr_smile_section_min_strike_is_minus_shift() -> None:
    section = SabrSmileSection(
        forward=0.05, sabr_params=(0.04, 0.5, 0.4, -0.1),
        exercise_time=1.0, shift=0.01,
    )
    tolerance.exact(section.min_strike(), -0.01)


def test_sabr_smile_section_max_strike_is_infinity() -> None:
    section = SabrSmileSection(
        forward=0.05, sabr_params=(0.04, 0.5, 0.4, -0.1),
        exercise_time=1.0,
    )
    assert section.max_strike() == math.inf


# --- input bounds ------------------------------------------------------


def test_sabr_smile_section_rejects_zero_forward_without_shift() -> None:
    with pytest.raises(LibraryException, match="forward"):
        SabrSmileSection(
            forward=0.0, sabr_params=(0.04, 0.5, 0.4, -0.1),
            exercise_time=1.0,
        )


def test_sabr_smile_section_rejects_invalid_sabr_params() -> None:
    with pytest.raises(LibraryException, match="alpha"):
        SabrSmileSection(
            forward=0.05, sabr_params=(0.0, 0.5, 0.4, -0.1),
            exercise_time=1.0,
        )


# --- VolatilityType.Normal arm ----------------------------------------


def test_sabr_smile_section_normal_arm_evaluates() -> None:
    section = SabrSmileSection(
        forward=0.05, sabr_params=(0.04, 0.5, 0.4, -0.1),
        exercise_time=1.0,
        volatility_type=VolatilityType.Normal,
    )
    direct = sabr_volatility(
        0.05, 0.05, 1.0, 0.04, 0.5, 0.4, -0.1,
        VolatilityType.Normal,
    )
    tolerance.exact(section.volatility(0.05), direct)
