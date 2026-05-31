"""Tests for LinearFlatInterpolation (linear w/ flat extrapolation).

C++ parity: ql/experimental/shortrate/generalizedhullwhite.hpp:310-382
(``LinearFlatInterpolation`` + ``LinearFlat``) @ v1.42.1.

No probe needed — the behaviour is a small closed form verified against
hand-computed values + the analytic linear/flat properties.
"""

from __future__ import annotations

import numpy as np

from pquantlib.experimental.shortrate.linear_flat_interpolation import (
    LinearFlat,
    LinearFlatInterpolation,
)
from pquantlib.testing.tolerance import tight


def _interp() -> LinearFlatInterpolation:
    xs = np.array([0.0, 1.0, 3.0], dtype=np.float64)
    ys = np.array([10.0, 12.0, 8.0], dtype=np.float64)
    interp = LinearFlat.interpolate(xs, ys)
    interp.enable_extrapolation()
    return interp


def test_linear_inside_range() -> None:
    f = _interp()
    # midway 0..1: slope (12-10)/1 = 2 -> at 0.5 -> 11
    tight(f(0.5), 11.0)
    # midway 1..3: slope (8-12)/2 = -2 -> at 2.0 -> 10
    tight(f(2.0), 10.0)
    # nodes exact
    tight(f(0.0), 10.0)
    tight(f(1.0), 12.0)
    tight(f(3.0), 8.0)


def test_flat_extrapolation_value() -> None:
    f = _interp()
    # below x_min -> first y; above x_max -> last y (flat, not linear)
    tight(f(-5.0, allow_extrapolation=True), 10.0)
    tight(f(100.0, allow_extrapolation=True), 8.0)


def test_derivative_zero_outside_range() -> None:
    f = _interp()
    # inside: slope of the bracketing segment
    tight(f.derivative(0.5), 2.0)
    tight(f.derivative(2.0), -2.0)
    # outside: flat -> zero derivative
    tight(f.derivative(-5.0, allow_extrapolation=True), 0.0)
    tight(f.derivative(100.0, allow_extrapolation=True), 0.0)


def test_second_derivative_zero() -> None:
    f = _interp()
    tight(f.second_derivative(0.5), 0.0)


def test_primitive_matches_trapezoid() -> None:
    f = _interp()
    # primitive at x=1 = area under [0,1] = trapezoid (10+12)/2 * 1 = 11
    tight(f.primitive(1.0), 11.0)
    # primitive at x=0 = 0
    tight(f.primitive(0.0), 0.0)


def test_single_point_constant() -> None:
    f = LinearFlatInterpolation(
        np.array([2.0, 3.0], dtype=np.float64), np.array([5.0, 5.0], dtype=np.float64)
    )
    f.enable_extrapolation()
    tight(f(0.0, allow_extrapolation=True), 5.0)
    tight(f(2.5), 5.0)
    tight(f(10.0, allow_extrapolation=True), 5.0)


def test_traits_constants() -> None:
    assert LinearFlat.global_ is False
    assert LinearFlat.required_points == 1
