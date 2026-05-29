"""Tests for InterpolatedSmileSection (cubic-natural-spline default)."""

from __future__ import annotations

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.array import Array
from pquantlib.math.interpolations.cubic_interpolation import CubicNaturalSpline
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.termstructures.volatility.interpolated_smile_section import (
    InterpolatedSmileSection,
)
from pquantlib.testing import reference_reader, tolerance

_REF = reference_reader.load("cluster/l9c")
_INT = _REF["interpolated_smile_section"]


# --- construction ------------------------------------------------------


def test_interpolated_smile_section_basic_construction() -> None:
    section = InterpolatedSmileSection(
        strikes=_INT["strikes"],
        volatilities=_INT["vols"],
        atm_level=_INT["atm_level"],
        exercise_time=_INT["exercise_time"],
    )
    tolerance.exact(section.exercise_time(), _INT["exercise_time"])
    tolerance.exact(section.atm_level(), _INT["atm_level"])
    tolerance.exact(section.min_strike(), _INT["strikes"][0])
    tolerance.exact(section.max_strike(), _INT["strikes"][-1])


def test_interpolated_smile_section_pillar_strikes_return_input_vols() -> None:
    """At each pillar, the interpolated vol must match the input vol exactly."""
    section = InterpolatedSmileSection(
        strikes=_INT["strikes"],
        volatilities=_INT["vols"],
        atm_level=_INT["atm_level"],
        exercise_time=_INT["exercise_time"],
    )
    for k, v in zip(_INT["strikes"], _INT["vols"], strict=True):
        tolerance.tight(section.volatility(k), v)


def test_interpolated_smile_section_default_uses_cubic_natural_spline() -> None:
    """The default interpolator factory is CubicNaturalSpline (L9-A)."""
    section = InterpolatedSmileSection(
        strikes=_INT["strikes"],
        volatilities=_INT["vols"],
        atm_level=_INT["atm_level"],
        exercise_time=_INT["exercise_time"],
    )
    interp = section._interp  # type: ignore[attr-defined]
    assert isinstance(interp, CubicNaturalSpline)


def test_interpolated_smile_section_pillar_vol_at_5pct_matches_probe_exact() -> None:
    """Pillar vol — same under any interpolator (cubic / linear)."""
    section = InterpolatedSmileSection(
        strikes=_INT["strikes"],
        volatilities=_INT["vols"],
        atm_level=_INT["atm_level"],
        exercise_time=_INT["exercise_time"],
    )
    tolerance.tight(section.volatility(0.05), _INT["vol_at_pillar_strike_5pct"])


# --- intermediate (cubic-interpolated) vols ---------------------------


def test_interpolated_smile_section_intermediate_strikes_are_finite() -> None:
    """Off-pillar strikes should produce finite cubic-interpolated values."""
    section = InterpolatedSmileSection(
        strikes=_INT["strikes"],
        volatilities=_INT["vols"],
        atm_level=_INT["atm_level"],
        exercise_time=_INT["exercise_time"],
    )
    v45 = section.volatility(0.045)
    v55 = section.volatility(0.055)
    # By symmetry of the U-smile (vols are symmetric around 0.05),
    # the cubic should produce equal vols at 0.045 and 0.055.
    tolerance.loose(v45, v55)


def _linear_factory(xs: Array, ys: Array) -> LinearInterpolation:
    """Top-level factory closure for `interpolator=` kwarg testing."""
    return LinearInterpolation(xs, ys)


def test_interpolated_smile_section_linear_arm_matches_probe() -> None:
    """Passing a linear interpolator should reproduce the C++ <Linear> probe."""
    section = InterpolatedSmileSection(
        strikes=_INT["strikes"],
        volatilities=_INT["vols"],
        atm_level=_INT["atm_level"],
        exercise_time=_INT["exercise_time"],
        interpolator=_linear_factory,
    )
    tolerance.tight(section.volatility(0.045), _INT["linear_vol_at_strike_45bp"])
    tolerance.tight(section.volatility(0.055), _INT["linear_vol_at_strike_55bp"])


# --- flat-strike extrapolation -----------------------------------------


def test_interpolated_smile_section_flat_extrapolation_clamps() -> None:
    section = InterpolatedSmileSection(
        strikes=[0.03, 0.05, 0.07],
        volatilities=[0.22, 0.18, 0.22],
        atm_level=0.05,
        exercise_time=1.0,
        flat_strike_extrapolation=True,
    )
    # Strike below pillars — should clamp to vol at 0.03 (= 0.22).
    tolerance.tight(section.volatility(0.01), 0.22)
    # Strike above pillars — should clamp to vol at 0.07 (= 0.22).
    tolerance.tight(section.volatility(0.10), 0.22)


# --- input bounds ------------------------------------------------------


def test_interpolated_smile_section_rejects_unsorted_strikes() -> None:
    with pytest.raises(LibraryException, match="sorted"):
        InterpolatedSmileSection(
            strikes=[0.05, 0.04],
            volatilities=[0.18, 0.20],
            atm_level=0.05,
            exercise_time=1.0,
        )


def test_interpolated_smile_section_rejects_mismatched_lengths() -> None:
    with pytest.raises(LibraryException, match="same length"):
        InterpolatedSmileSection(
            strikes=[0.04, 0.05],
            volatilities=[0.18],
            atm_level=0.05,
            exercise_time=1.0,
        )


def test_interpolated_smile_section_rejects_single_strike() -> None:
    with pytest.raises(LibraryException, match="at least 2"):
        InterpolatedSmileSection(
            strikes=[0.05],
            volatilities=[0.18],
            atm_level=0.05,
            exercise_time=1.0,
        )
