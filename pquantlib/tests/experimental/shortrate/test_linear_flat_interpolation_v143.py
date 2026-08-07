"""Cross-validate LinearFlatInterpolation against C++ QuantLib v1.43.

Probe source: migration-harness/cpp/probes/v143_experimental_interp/probe.cpp
Reference:    migration-harness/references/v143/experimental/interp.json

Covers ``detail::LinearFlatInterpolationImpl``
(ql/experimental/shortrate/generalizedhullwhite.hpp:341), which C++ exposes
only through the ``LinearFlatInterpolation`` facade (line 313) and the
``LinearFlat`` factory/traits class (line 328). Every value here came from
running the C++ binary; the pre-existing
``test_linear_flat_interpolation.py`` carries hand-computed constants and
is deliberately left in place as a redundant algebraic check.

Single-node grids (n == 1) are NOT tested: C++
``Interpolation::templateImpl::locate`` returns ``Size(-1)`` there, so
``primitive``/``derivative`` read out of bounds and the probe's output for
that case changed between consecutive runs. See the probe header.

Tolerance: TIGHT throughout. The implementation is a handful of
multiplications and additions in the same order as C++, so anything looser
would hide a genuine divergence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.experimental.shortrate.linear_flat_interpolation import (
    LinearFlat,
    LinearFlatInterpolation,
)
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight

_CASES = ["lf_a", "lf_b", "lf_c", "lf_d"]


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("v143/experimental/interp")


def _build(cpp_ref: dict[str, Any], tag: str) -> LinearFlatInterpolation:
    xs = np.array(cpp_ref[f"{tag}_x"], dtype=np.float64)
    ys = np.array(cpp_ref[f"{tag}_y"], dtype=np.float64)
    f = LinearFlatInterpolation(xs, ys)
    f.enable_extrapolation()
    return f


@pytest.mark.parametrize("tag", _CASES)
def test_value_matches_cpp(cpp_ref: dict[str, Any], tag: str) -> None:
    """``LinearFlatInterpolationImpl::value`` (generalizedhullwhite.hpp:358-365).

    Includes queries at, just inside and just outside both boundary nodes:
    the C++ flat branches use ``<=`` / ``>=``, so the boundary node itself
    short-circuits before ``locate`` is ever called.
    """
    f = _build(cpp_ref, tag)
    for q, expected in zip(cpp_ref[f"{tag}_q"], cpp_ref[f"{tag}_value"], strict=True):
        tight(f(q, allow_extrapolation=True), expected, reason=f"{tag} value at x={q}")


@pytest.mark.parametrize("tag", _CASES)
def test_primitive_matches_cpp(cpp_ref: dict[str, Any], tag: str) -> None:
    """``LinearFlatInterpolationImpl::primitive`` (generalizedhullwhite.hpp:366-371).

    Note the asymmetry the reference pins: ``primitive`` has no in-range
    guard at all — it calls ``locate`` directly — so outside the node range
    C++ extrapolates the boundary segment's quadratic rather than clamping.
    At ``lf_a`` x=100 that is -8602, not the in-range integral.
    """
    f = _build(cpp_ref, tag)
    for q, expected in zip(cpp_ref[f"{tag}_q"], cpp_ref[f"{tag}_primitive"], strict=True):
        tight(f.primitive(q, allow_extrapolation=True), expected, reason=f"{tag} primitive x={q}")


@pytest.mark.parametrize("tag", _CASES)
def test_derivative_matches_cpp(cpp_ref: dict[str, Any], tag: str) -> None:
    """``LinearFlatInterpolationImpl::derivative`` (generalizedhullwhite.hpp:372-377).

    Zero outside the range (``isInRange`` guard), bracketing-segment slope
    inside. At the upper boundary node itself ``isInRange`` is true, so C++
    returns the LAST segment's slope, not zero.
    """
    f = _build(cpp_ref, tag)
    for q, expected in zip(cpp_ref[f"{tag}_q"], cpp_ref[f"{tag}_derivative"], strict=True):
        tight(f.derivative(q, allow_extrapolation=True), expected, reason=f"{tag} deriv x={q}")


@pytest.mark.parametrize("tag", _CASES)
def test_second_derivative_matches_cpp(cpp_ref: dict[str, Any], tag: str) -> None:
    """``LinearFlatInterpolationImpl::secondDerivative`` → 0.0 unconditionally."""
    f = _build(cpp_ref, tag)
    for q, expected in zip(
        cpp_ref[f"{tag}_q"], cpp_ref[f"{tag}_second_derivative"], strict=True
    ):
        tight(f.second_derivative(q, allow_extrapolation=True), expected)


@pytest.mark.parametrize("tag", _CASES)
def test_range_matches_cpp(cpp_ref: dict[str, Any], tag: str) -> None:
    f = _build(cpp_ref, tag)
    tight(f.x_min, cpp_ref[f"{tag}_xMin"])
    tight(f.x_max, cpp_ref[f"{tag}_xMax"])


def test_traits_match_cpp(cpp_ref: dict[str, Any]) -> None:
    """``LinearFlat::requiredPoints == 1`` and ``LinearFlat::global == false``."""
    assert LinearFlat.required_points == cpp_ref["lf_required_points"]
    assert LinearFlat.global_ == bool(cpp_ref["lf_global"])


def test_factory_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    """``LinearFlat::interpolate`` builds the same interpolation."""
    xs = np.array(cpp_ref["lf_a_x"], dtype=np.float64)
    ys = np.array(cpp_ref["lf_a_y"], dtype=np.float64)
    f = LinearFlat.interpolate(xs, ys)
    f.enable_extrapolation()
    for q, expected in zip(
        [-1.0, 0.5, 2.0, 5.0], cpp_ref["lf_factory_value"], strict=True
    ):
        tight(f(q, allow_extrapolation=True), expected)
