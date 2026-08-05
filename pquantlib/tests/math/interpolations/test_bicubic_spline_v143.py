"""Cross-validate BicubicSpline + BicubicSplineDerivatives against v1.43 C++.

Reference: ``migration-harness/references/v143/math/interp/kernel.json`` —
``bicubic`` section. The grid is 5 x-points by 4 y-points, both non-uniformly
spaced, with ``z[j, i] = sin(xs[i]) + cos(ys[j])``. Non-square catches a
transposed matrix; non-uniform catches an implementation that assumed even
spacing.

This is the test that replaced a ``scipy.interpolate.RectBivariateSpline``
delegation. That delegation was a tensor-product B-spline with not-a-knot end
conditions, where C++ composes *natural* 1-D cubic splines row-then-column;
it agreed at the pillars and disagreed by ~10 % in between. The old test
asserted only a 0.15 relative-error envelope and described the gap as
"qualitative". With the composition transcribed the whole surface — and all
five partial derivatives, which the delegation could not produce at all —
agrees to TIGHT.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.interpolations.bicubic_spline import (
    Bicubic,
    BicubicSpline,
    BicubicSplineDerivatives,
)
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/interp/kernel")


def _make(block: dict[str, Any]) -> BicubicSpline:
    interp = BicubicSpline(
        np.asarray(block["xs"], dtype=np.float64),
        np.asarray(block["ys"], dtype=np.float64),
        np.asarray(block["z"], dtype=np.float64),
    )
    interp.enable_extrapolation()
    return interp


def test_values_match_cpp_tight(cpp: dict[str, Any]) -> None:
    """Surface values match C++ to TIGHT, at nodes, midpoints and off-grid.

    Both sides build the same object: ``len(ys)`` natural cubic splines over
    ``xs``, then a fresh natural cubic spline over ``ys`` through their
    values at ``x``. The natural interpolating cubic spline through a given
    knot set is unique, so C++'s tridiagonal solve and scipy's banded solve
    describe the same function and differ only in rounding.
    """
    block = cpp["bicubic"]
    interp = _make(block)
    for x, y, expected in block["evals"]:
        tolerance.tight(interp(float(x), float(y)), float(expected))


def test_pillars_roundtrip(cpp: dict[str, Any]) -> None:
    """At the grid nodes the surface reproduces ``z`` — the interpolation property."""
    block = cpp["bicubic"]
    interp = _make(block)
    z = np.asarray(block["z"], dtype=np.float64)
    for j, y in enumerate(block["ys"]):
        for i, x in enumerate(block["xs"]):
            tolerance.tight(interp(float(x), float(y)), float(z[j, i]))


@pytest.mark.parametrize(
    ("column", "method"),
    [
        (2, "derivative_x"),
        (3, "derivative_y"),
        (4, "derivative_xy"),
        (5, "second_derivative_x"),
        (6, "second_derivative_y"),
    ],
)
def test_derivatives_match_cpp_tight(
    cpp: dict[str, Any], column: int, method: str
) -> None:
    """Every ``BicubicSplineDerivatives`` entry point matches C++ to TIGHT.

    ``derivative_xy`` is the interesting one: C++ builds it by sampling
    ``derivativeY`` across the x pillars and differentiating *that* spline,
    which is not the same as differentiating the surface twice in either
    order. Reproducing the composition, not the mathematics, is the point.
    """
    block = cpp["bicubic"]
    assert block["derivs_columns"] == "x,y,dX,dY,dXY,d2X,d2Y"
    interp = _make(block)
    fn = getattr(interp, method)
    for row in block["derivs"]:
        tolerance.tight(fn(float(row[0]), float(row[1])), float(row[column]))


def test_derivatives_refuse_out_of_range(cpp: dict[str, Any]) -> None:
    """The derivative API range-checks even when extrapolation is enabled.

    C++ ``derivativeX`` etc. build a plain ``CubicInterpolation`` and call
    ``derivative(x)`` with ``allowExtrapolation`` left at ``false``, so the
    fresh interpolation's own check fires regardless of the surface's flag.
    ``value``, by contrast, passes ``true`` to its inner splines.
    """
    block = cpp["bicubic"]
    interp = _make(block)  # extrapolation ENABLED
    x_out = float(block["xs"][-1]) + 1.0
    y_out = float(block["ys"][-1]) + 1.0
    _ = interp(x_out, y_out)  # value: fine
    with pytest.raises(LibraryException, match="extrapolation"):
        interp.derivative_x(x_out, 0.0)
    with pytest.raises(LibraryException, match="extrapolation"):
        interp.derivative_y(1.0, y_out)


def test_factory_matches_direct_construction(cpp: dict[str, Any]) -> None:
    """``Bicubic().interpolate(...)`` is the C++ traits path; same numbers."""
    block = cpp["bicubic"]
    interp = Bicubic().interpolate(
        np.asarray(block["xs"], dtype=np.float64),
        np.asarray(block["ys"], dtype=np.float64),
        np.asarray(block["z"], dtype=np.float64),
    )
    interp.enable_extrapolation()
    for x, y, expected in block["factory_evals"]:
        tolerance.tight(interp(float(x), float(y)), float(expected))


def test_implements_the_derivatives_interface(cpp: dict[str, Any]) -> None:
    """C++ ``BicubicSplineImpl`` inherits ``detail::BicubicSplineDerivatives``."""
    assert isinstance(_make(cpp["bicubic"]), BicubicSplineDerivatives)


def test_accepts_a_two_by_two_grid() -> None:
    """C++ enforces only ``Interpolation2D``'s 2-point minimum on each axis.

    The scipy delegation this replaced required 4 points per axis (``kx+1``),
    so a 2x2 or 3x3 grid raised where C++ happily interpolates.
    """
    xs = np.array([0.0, 1.0], dtype=np.float64)
    ys = np.array([0.0, 2.0], dtype=np.float64)
    z = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float64)
    interp = BicubicSpline(xs, ys, z)
    tolerance.tight(interp(0.0, 0.0), 1.0)
    tolerance.tight(interp(1.0, 2.0), 5.0)


def test_update_rebuilds_from_mutated_z(cpp: dict[str, Any]) -> None:
    """``update()`` re-runs ``calculate()``; scaling z scales the surface."""
    block = cpp["bicubic"]
    interp = _make(block)
    before = interp(1.5, 0.75)
    interp._z[:] = 2.0 * interp._z  # pyright: ignore[reportPrivateUsage]
    interp.update()
    tolerance.tight(interp(1.5, 0.75), 2.0 * before)
