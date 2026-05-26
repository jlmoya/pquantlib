"""Cross-validate 1-D + 2-D interpolations against the L1-E C++ probe.

Reference: ``migration-harness/references/cluster/e.json``.

Sections:
- ``linear``:        x=[0,1,2,3,4], y=[0,1,4,9,16]
- ``loglinear``:     x=[0,1,2,3,4], y=[1,2,4,9,16] (shifted away from 0)
- ``backward_flat``: x=[0,1,2,3,4], y=[0,1,4,9,16]
- ``forward_flat``:  x=[0,1,2,3,4], y=[0,1,4,9,16]
- ``bilinear``:      xs=ys=[0,1,2], z[i,j] = ys[i] + xs[j]
- ``cholesky``: tested in tests/math/matrixutilities/test_cholesky.py
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.interpolations.backward_flat import BackwardFlatInterpolation
from pquantlib.math.interpolations.bilinear import BilinearInterpolation
from pquantlib.math.interpolations.forward_flat import ForwardFlatInterpolation
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.math.interpolations.log_linear import LogLinearInterpolation
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("cluster/e")


# --- linear ------------------------------------------------------------


def test_linear_matches_cpp(cpp: dict[str, Any]) -> None:
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    ys = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
    interp = LinearInterpolation(xs, ys)
    for case in cpp["linear"]:
        tolerance.tight(interp(float(case["x"])), float(case["v"]))


def test_linear_hits_knots_exactly() -> None:
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    ys = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
    interp = LinearInterpolation(xs, ys)
    for x, y in zip(xs, ys, strict=True):
        tolerance.tight(interp(float(x)), float(y))


def test_linear_derivative_constant_between_knots() -> None:
    xs = np.array([0.0, 1.0, 2.0, 3.0])
    ys = np.array([0.0, 2.0, 6.0, 12.0])
    interp = LinearInterpolation(xs, ys)
    # Slopes are 2, 4, 6 in each segment.
    tolerance.tight(interp.derivative(0.5), 2.0)
    tolerance.tight(interp.derivative(1.5), 4.0)
    tolerance.tight(interp.derivative(2.5), 6.0)


def test_linear_second_derivative_zero() -> None:
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([0.0, 1.0, 4.0])
    interp = LinearInterpolation(xs, ys)
    tolerance.exact(interp.second_derivative(0.5), 0.0)
    tolerance.exact(interp.second_derivative(1.5), 0.0)


def test_linear_rejects_extrapolation_by_default() -> None:
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([0.0, 1.0, 4.0])
    interp = LinearInterpolation(xs, ys)
    with pytest.raises(LibraryException, match="extrapolation"):
        interp(5.0)


def test_linear_allows_extrapolation_when_requested() -> None:
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([0.0, 1.0, 4.0])
    interp = LinearInterpolation(xs, ys)
    # Outside range — C++ clamps to last segment slope: y = 4 + (5-2)*3 = 13.
    tolerance.tight(interp(5.0, allow_extrapolation=True), 13.0)


def test_linear_requires_two_points() -> None:
    with pytest.raises(LibraryException, match="at least 2"):
        LinearInterpolation(np.array([0.0]), np.array([0.0]))


def test_linear_length_mismatch_raises() -> None:
    with pytest.raises(LibraryException, match="same length"):
        LinearInterpolation(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]))


# --- log linear --------------------------------------------------------


def test_log_linear_matches_cpp(cpp: dict[str, Any]) -> None:
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    # C++ probe uses ys=[1,2,4,9,16] — shifted away from zero so log is defined.
    ys2 = np.array([1.0, 2.0, 4.0, 9.0, 16.0])
    interp = LogLinearInterpolation(xs, ys2)
    for case in cpp["loglinear"]:
        tolerance.tight(interp(float(case["x"])), float(case["v"]))


def test_log_linear_hits_knots_exactly() -> None:
    xs = np.array([1.0, 2.0, 4.0])
    ys = np.array([1.0, 4.0, 16.0])
    interp = LogLinearInterpolation(xs, ys)
    for x, y in zip(xs, ys, strict=True):
        tolerance.tight(interp(float(x)), float(y))


def test_log_linear_rejects_non_positive_y() -> None:
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([1.0, 0.0, 1.0])
    with pytest.raises(LibraryException, match="strictly positive"):
        LogLinearInterpolation(xs, ys)


# --- backward flat -----------------------------------------------------


def test_backward_flat_matches_cpp(cpp: dict[str, Any]) -> None:
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    ys = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
    interp = BackwardFlatInterpolation(xs, ys)
    for case in cpp["backward_flat"]:
        tolerance.tight(interp(float(case["x"])), float(case["v"]))


def test_backward_flat_picks_right_knot() -> None:
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([10.0, 20.0, 30.0])
    interp = BackwardFlatInterpolation(xs, ys)
    # At x=0, returns ys[0]=10 (the x <= xs[0] branch).
    tolerance.exact(interp(0.0), 10.0)
    # In the first segment (0 < x < 1), returns ys[1]=20.
    tolerance.exact(interp(0.5), 20.0)
    # At knot xs[1]=1, returns ys[1]=20.
    tolerance.exact(interp(1.0), 20.0)
    # In the second segment, returns ys[2]=30.
    tolerance.exact(interp(1.5), 30.0)


def test_backward_flat_derivative_zero() -> None:
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([0.0, 1.0, 4.0])
    interp = BackwardFlatInterpolation(xs, ys)
    tolerance.exact(interp.derivative(0.5), 0.0)
    tolerance.exact(interp.second_derivative(0.5), 0.0)


# --- forward flat ------------------------------------------------------


def test_forward_flat_matches_cpp(cpp: dict[str, Any]) -> None:
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    ys = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
    interp = ForwardFlatInterpolation(xs, ys)
    for case in cpp["forward_flat"]:
        tolerance.tight(interp(float(case["x"])), float(case["v"]))


def test_forward_flat_picks_left_knot() -> None:
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([10.0, 20.0, 30.0])
    interp = ForwardFlatInterpolation(xs, ys)
    # At x=0 (knot), returns ys[0]=10.
    tolerance.exact(interp(0.0), 10.0)
    # In first segment, returns ys[0]=10.
    tolerance.exact(interp(0.5), 10.0)
    # At knot xs[1]=1, returns ys[1]=20 (locate returns 1).
    tolerance.exact(interp(1.0), 20.0)
    # In second segment, returns ys[1]=20.
    tolerance.exact(interp(1.5), 20.0)
    # At/past xs[-1], returns ys[-1]=30.
    tolerance.exact(interp(2.0), 30.0)


def test_forward_flat_derivative_zero() -> None:
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([0.0, 1.0, 4.0])
    interp = ForwardFlatInterpolation(xs, ys)
    tolerance.exact(interp.derivative(0.5), 0.0)
    tolerance.exact(interp.second_derivative(0.5), 0.0)


# --- bilinear ----------------------------------------------------------


def test_bilinear_matches_cpp(cpp: dict[str, Any]) -> None:
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([0.0, 1.0, 2.0])
    # z[i,j] = ys[i] + xs[j]: rows are y, cols are x.
    z = np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    bilin = BilinearInterpolation(xs, ys, z)
    for case in cpp["bilinear"]:
        tolerance.tight(bilin(float(case["x"]), float(case["y"])), float(case["v"]))


def test_bilinear_hits_grid_nodes_exactly() -> None:
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([0.0, 1.0, 2.0])
    z = np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    bilin = BilinearInterpolation(xs, ys, z)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            tolerance.exact(bilin(float(x), float(y)), float(z[i, j]))


def test_bilinear_shape_mismatch_raises() -> None:
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([0.0, 1.0])
    z = np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])  # 3x3, but ys has 2 entries
    with pytest.raises(LibraryException, match="does not match"):
        BilinearInterpolation(xs, ys, z)


def test_bilinear_rejects_extrapolation_by_default() -> None:
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([0.0, 1.0, 2.0])
    z = np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    bilin = BilinearInterpolation(xs, ys, z)
    with pytest.raises(LibraryException, match="extrapolation"):
        bilin(5.0, 1.0)


def test_bilinear_requires_2d_z() -> None:
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([0.0, 1.0, 2.0])
    z_1d = np.array([0.0, 1.0, 2.0])
    with pytest.raises(LibraryException, match="2-D"):
        BilinearInterpolation(xs, ys, z_1d)
