"""Cross-validate PiecewiseBlackVarianceSurface against the C++ probe.

Reference: ``migration-harness/references/v143/piecewise/black/variance/surface``.

The surface interpolates *total variance* between tenors, which is the one
thing a port is most likely to get wrong: interpolating volatility instead
matches at every tenor and nowhere between them, and freezing the variance past
the last tenor matches everywhere except beyond it. The probe therefore sweeps
a time ladder that lands inside the ramp to the first tenor, exactly on each
tenor, strictly between them, and well past the last one, across six strikes
and two surfaces (one tenor and two tenors carrying *different* SVI shapes).

Tolerance: TIGHT throughout. Nothing here is iterative or a finite difference —
the surface evaluates a closed-form SVI variance and takes a convex combination
of two of them.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.experimental.volatility.svi_smile_section import SviSmileSection
from pquantlib.math.constants import QL_MAX_REAL
from pquantlib.termstructures.volatility.equity_fx.piecewise_black_variance_surface import (
    PiecewiseBlackVarianceSurface,
)
from pquantlib.termstructures.volatility.flat_smile_section import FlatSmileSection
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.month import Month

TODAY = Date.from_ymd(1, Month.March, 2025)
T1 = Date.from_ymd(1, Month.March, 2026)
T2 = Date.from_ymd(1, Month.March, 2027)

# SVI parameters (a, b, sigma, rho, m) — the v1.43 test-suite's
# testGaussianCopulaSpreadEngineSVI pair, so these are the same shapes the
# spread-engine probe drives its setups D and E with.
SVI_1 = (0.04, 0.10, 0.30, -0.40, 0.0)
SVI_2 = (0.02, 0.08, 0.25, -0.30, 0.0)


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/piecewise/black/variance/surface")


def _svi(t: float, forward: float, params: tuple[float, float, float, float, float]) -> SmileSection:
    return SviSmileSection(forward=forward, svi_params=params, exercise_time=t)


def _single() -> PiecewiseBlackVarianceSurface:
    return PiecewiseBlackVarianceSurface.single_tenor(
        reference_date=TODAY,
        date=T1,
        smile_section=_svi(1.0, 100.0, SVI_1),
        day_counter=Actual365Fixed(),
    )


def _two_tenor() -> PiecewiseBlackVarianceSurface:
    return PiecewiseBlackVarianceSurface(
        reference_date=TODAY,
        dates=[T1, T2],
        smile_sections=[_svi(1.0, 100.0, SVI_1), _svi(2.0, 96.0, SVI_2)],
        day_counter=Actual365Fixed(),
    )


def _surface_for(setup: str) -> PiecewiseBlackVarianceSurface:
    return _single() if setup == "single" else _two_tenor()


# --- structure --------------------------------------------------------------


def test_structure(cpp: dict[str, Any]) -> None:
    expected = cpp["structure"]["expected"]
    tolerance.tight(_single().max_time(), float(expected["max_time_single"]))
    tolerance.tight(_two_tenor().max_time(), float(expected["max_time_two_tenor"]))
    # C++ QL_MIN_REAL is -max(), not the smallest positive normal.
    assert expected["min_strike_is_ql_min_real"] is True
    assert expected["max_strike_is_ql_max_real"] is True
    tolerance.exact(_single().min_strike(), -QL_MAX_REAL)
    tolerance.exact(_single().max_strike(), QL_MAX_REAL)
    assert _single().day_counter().name() == expected["day_counter"]


# --- variance / vol across every time regime --------------------------------


def test_variance_and_vol_ladder(cpp: dict[str, Any]) -> None:
    """Every ``*_variance_*`` case: both surfaces, ten times, six strikes."""
    checked = 0
    for name, case in cpp.items():
        if "_variance_t" not in name:
            continue
        inputs = case["inputs"]
        expected = case["expected"]
        surface = _surface_for(str(inputs["setup"]))
        t = float(inputs["t"])
        strike = float(inputs["strike"])
        tolerance.tight(
            surface.black_variance_at_time(t, strike, True),
            float(expected["black_variance"]),
            reason=name,
        )
        tolerance.tight(
            surface.black_vol_at_time(t, strike, True),
            float(expected["black_vol"]),
            reason=name,
        )
        checked += 1
    assert checked >= 100, f"expected the probe to carry a full ladder, got {checked}"


def test_variance_is_interpolated_in_variance_not_in_vol(cpp: dict[str, Any]) -> None:
    """Between two tenors the surface is linear in *total variance*.

    The distinction only shows up between tenors, so it is asserted directly
    rather than left to the ladder: the midpoint value must be the average of
    the two tenor variances, which is *not* what interpolating volatility (or
    variance rate) would give.
    """
    surface = _two_tenor()
    var1 = surface.black_variance_at_time(1.0, 100.0, True)
    var2 = surface.black_variance_at_time(2.0, 100.0, True)
    mid = surface.black_variance_at_time(1.5, 100.0, True)
    tolerance.tight(mid, 0.5 * (var1 + var2))
    # And it is genuinely a different number from the vol-space interpolation.
    vol_space = (0.5 * (math.sqrt(var1 / 1.0) + math.sqrt(var2 / 2.0))) ** 2 * 1.5
    assert abs(mid - vol_space) > 1e-3, "the two schemes must be distinguishable here"


def test_flat_vol_extrapolation_beyond_last_tenor(cpp: dict[str, Any]) -> None:
    """Past the last tenor the *volatility* is flat, so variance keeps growing."""
    surface = _two_tenor()
    vol_at_last = surface.black_vol_at_time(2.0, 100.0, True)
    for t in (2.5, 4.0):
        tolerance.tight(surface.black_vol_at_time(t, 100.0, True), vol_at_last)
    tolerance.tight(
        surface.black_variance_at_time(4.0, 100.0, True),
        surface.black_variance_at_time(2.0, 100.0, True) * 2.0,
    )


def test_zero_time_is_zero_variance() -> None:
    tolerance.exact(_two_tenor().black_variance_at_time(0.0, 100.0, True), 0.0)


# --- the smile view ---------------------------------------------------------


@pytest.mark.parametrize(
    ("case_name", "setup", "t"),
    [
        ("single_smile_at_tenor_1y", "single", 1.0),
        ("single_smile_between_tenors", "single", 0.5),
        ("two_tenor_smile_at_tenor_1y", "two_tenor", 1.0),
        ("two_tenor_smile_at_tenor_2y", "two_tenor", 2.0),
        ("two_tenor_smile_between_tenors", "two_tenor", 1.5),
    ],
)
def test_smile_section(cpp: dict[str, Any], case_name: str, setup: str, t: float) -> None:
    expected = cpp[case_name]["expected"]
    section = _surface_for(setup).smile_section_at_time(t, True)

    tolerance.tight(section.exercise_time(), float(expected["exercise_time"]))
    # C++ Null<Real>() for a level is NaN here.
    assert math.isnan(section.atm_level()) is bool(expected["atm_level_is_null"])
    if not expected["atm_level_is_null"]:
        tolerance.tight(section.atm_level(), float(expected["atm_level"]))
    for strike in (80.0, 100.0, 120.0):
        tolerance.tight(
            section.volatility(strike),
            float(expected[f"volatility_at_{int(strike)}"]),
            reason=f"{case_name} @ {strike}",
        )
    tolerance.tight(section.variance(100.0), float(expected["variance_at_100"]))


def test_smile_at_tenor_is_the_input_section(cpp: dict[str, Any]) -> None:
    """At a tenor the surface hands back the *same object* it was given.

    That identity is the whole point of the v1.43 ``smileSectionImpl``
    override: a parametric smile survives the round trip instead of being
    resampled through ``black_vol``. Anything weaker — an equal-valued copy —
    would still pass every value assertion above.
    """
    assert cpp["smile_at_tenor_is_the_input_section"]["expected"]["same_object"] is True
    section = _svi(1.0, 100.0, SVI_1)
    surface = PiecewiseBlackVarianceSurface.single_tenor(
        reference_date=TODAY,
        date=T1,
        smile_section=section,
        day_counter=Actual365Fixed(),
    )
    assert surface.smile_section_at_time(1.0, True) is section


def test_smile_off_tenor_falls_back_to_the_adapter() -> None:
    section = _svi(1.0, 100.0, SVI_1)
    surface = PiecewiseBlackVarianceSurface.single_tenor(
        reference_date=TODAY,
        date=T1,
        smile_section=section,
        day_counter=Actual365Fixed(),
    )
    assert surface.smile_section_at_time(0.5, True) is not section


# --- strike range: the section's, not the surface's -------------------------


def test_rejects_strike_below_section_min(cpp: dict[str, Any]) -> None:
    assert cpp["rejects_strike_below_section_min"]["expected"]["throws"] is True
    with pytest.raises(LibraryException, match="outside the range of smile section"):
        _single().black_variance_at_time(1.0, -1.0, False)


def test_allows_strike_below_section_min_when_extrapolating(cpp: dict[str, Any]) -> None:
    expected = cpp["allows_strike_below_section_min_when_extrapolating"]["expected"]
    assert expected["throws"] is False
    surface = _single()
    surface.enable_extrapolation()
    surface.black_variance_at_time(1.0, -1.0, False)


def test_flat_section_accepts_negative_strike(cpp: dict[str, Any]) -> None:
    """A FlatSmileSection reports an unbounded strike range, so nothing rejects."""
    assert cpp["flat_section_accepts_negative_strike"]["expected"]["throws"] is False
    surface = PiecewiseBlackVarianceSurface.single_tenor(
        reference_date=TODAY,
        date=T1,
        smile_section=FlatSmileSection(
            volatility=0.20,
            exercise_time=1.0,
            day_counter=Actual365Fixed(),
            atm_level=100.0,
        ),
        day_counter=Actual365Fixed(),
    )
    surface.black_variance_at_time(1.0, -1.0, False)


# --- constructor guards -----------------------------------------------------


def _expected_throws(cpp: dict[str, Any], case_name: str) -> bool:
    return bool(cpp[case_name]["expected"]["throws"])


def test_ctor_rejects_empty_dates(cpp: dict[str, Any]) -> None:
    assert _expected_throws(cpp, "ctor_rejects_empty_dates")
    with pytest.raises(LibraryException, match="at least one date is required"):
        PiecewiseBlackVarianceSurface(
            reference_date=TODAY,
            dates=[],
            smile_sections=[],
            day_counter=Actual365Fixed(),
        )


def test_ctor_rejects_length_mismatch(cpp: dict[str, Any]) -> None:
    assert _expected_throws(cpp, "ctor_rejects_length_mismatch")
    with pytest.raises(LibraryException, match="mismatch between 2 dates"):
        PiecewiseBlackVarianceSurface(
            reference_date=TODAY,
            dates=[T1, T2],
            smile_sections=[_svi(1.0, 100.0, SVI_1)],
            day_counter=Actual365Fixed(),
        )


def test_ctor_rejects_first_date_at_reference(cpp: dict[str, Any]) -> None:
    assert _expected_throws(cpp, "ctor_rejects_first_date_at_reference")
    with pytest.raises(LibraryException, match="must be after reference date"):
        PiecewiseBlackVarianceSurface(
            reference_date=TODAY,
            dates=[TODAY],
            smile_sections=[_svi(1.0, 100.0, SVI_1)],
            day_counter=Actual365Fixed(),
        )


def test_ctor_rejects_unsorted_dates(cpp: dict[str, Any]) -> None:
    assert _expected_throws(cpp, "ctor_rejects_unsorted_dates")
    with pytest.raises(LibraryException, match="dates must be sorted and unique"):
        PiecewiseBlackVarianceSurface(
            reference_date=TODAY,
            dates=[T2, T1],
            smile_sections=[_svi(2.0, 96.0, SVI_2), _svi(1.0, 100.0, SVI_1)],
            day_counter=Actual365Fixed(),
        )


def test_ctor_rejects_duplicate_dates(cpp: dict[str, Any]) -> None:
    assert _expected_throws(cpp, "ctor_rejects_duplicate_dates")
    with pytest.raises(LibraryException, match="dates must be sorted and unique"):
        PiecewiseBlackVarianceSurface(
            reference_date=TODAY,
            dates=[T1, T1],
            smile_sections=[_svi(1.0, 100.0, SVI_1), _svi(1.0, 96.0, SVI_2)],
            day_counter=Actual365Fixed(),
        )
