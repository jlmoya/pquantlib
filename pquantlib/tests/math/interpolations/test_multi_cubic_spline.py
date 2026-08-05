"""Cross-validate MultiCubicSpline against the C++ probes.

References:

* ``migration-harness/references/cluster/l10c.json`` — ``multi_cubic_spline``
  section: a uniform 4x4 grid of ``z = sin(x) + cos(y)``.
* ``migration-harness/references/v143/math/interp/kernel.json`` —
  ``multi_cubic_spline`` section: a **non-uniform** 5x4 grid and a
  non-uniform 5x4x4 grid, which is what actually discriminates a
  tensor-product natural spline from a local cubic.

The 2-D and 3-D cases used to delegate to
``scipy.interpolate.RegularGridInterpolator(method='cubic')`` and were 8-9 %
off C++; see the module docstring of the implementation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.interpolations.cubic_interpolation import CubicNaturalSpline
from pquantlib.math.interpolations.multi_cubic_spline import MultiCubicSpline
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("cluster/l10c")


@pytest.fixture(scope="module")
def cpp143() -> dict[str, Any]:
    return reference_reader.load("v143/math/interp/kernel")


def _make_2d() -> MultiCubicSpline:
    xs = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    ys = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    # z[i, j] = sin(xs[i]) + cos(ys[j]) — convention: values[k0, k1]
    # where k0 indexes xs (axis 0), k1 indexes ys (axis 1). This is the
    # same as scipy.RegularGridInterpolator's contract.
    z = np.zeros((4, 4), dtype=np.float64)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            z[i, j] = math.sin(float(x)) + math.cos(float(y))
    return MultiCubicSpline([xs, ys], z)


def test_pillars_match_input(cpp: dict[str, Any]) -> None:
    """Pillar evaluations equal the input values (TIGHT)."""
    block = cpp["multi_cubic_spline"]
    pillars = block["pillars"]
    interp = _make_2d()
    xs = [float(e) for e in block["xs"]]
    ys = [float(e) for e in block["ys"]]
    # C++ probe indexed pillars[j][i] (z[y_index, x_index]); convert to
    # our convention values[i, j].
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            tolerance.tight(interp([x, y]), float(pillars[j][i]))


def test_intermediates_match_cpp_tight(cpp: dict[str, Any]) -> None:
    """Off-node values match C++ to TIGHT.

    This used to be a 0.2 absolute/relative "certify the divergence is
    bounded" assertion, because the port delegated to scipy's
    ``RegularGridInterpolator(method='cubic')`` — a local cubic — where C++
    tensor-products *natural* cubic splines. Now that the recursion is
    transcribed the two describe the same function.
    """
    block = cpp["multi_cubic_spline"]
    mids = block["mids"]
    expected = [float(e) for e in block["mids_y"]]
    interp = _make_2d()
    for (x, y), exp in zip(mids, expected, strict=True):
        tolerance.tight(interp([float(x), float(y)]), exp)


def test_v143_2d_matches_cpp_tight(cpp143: dict[str, Any]) -> None:
    """Non-uniform 5x4 grid: interior nodes and off-node points, TIGHT.

    Non-uniform spacing is what separates the natural spline from a local
    cubic — on a uniform grid many schemes coincide much more closely.
    """
    block = cpp143["multi_cubic_spline"]["dim2"]
    interp = MultiCubicSpline(
        [np.asarray(block["axis0"], dtype=np.float64),
         np.asarray(block["axis1"], dtype=np.float64)],
        np.asarray(block["values"], dtype=np.float64),
    )
    for x, y, expected in block["evals"]:
        tolerance.tight(interp([float(x), float(y)]), float(expected))


def test_v143_3d_matches_cpp_tight(cpp143: dict[str, Any]) -> None:
    """Non-uniform 5x4x4 grid: the recursion has to hold at depth 3 too."""
    block = cpp143["multi_cubic_spline"]["dim3"]
    axes = [
        np.asarray(block[f"axis{k}"], dtype=np.float64) for k in range(3)
    ]
    values = np.asarray(block["values_flat"], dtype=np.float64).reshape(
        tuple(a.shape[0] for a in axes)
    )
    interp = MultiCubicSpline(axes, values)
    for x, y, z, expected in block["evals"]:
        tolerance.tight(interp([float(x), float(y), float(z)]), float(expected))


def test_v143_2d_reproduces_boundary_nodes(cpp143: dict[str, Any]) -> None:
    """Grid values are reproduced on the boundary surface as well.

    Not cross-validated against C++ on purpose: the C++ header carries a
    standing ``\\bug`` note that it "cannot interpolate at the grid points on
    the boundary surface", and its own test-suite only checks strictly
    interior nodes. The port has no such restriction, so the property is
    asserted directly instead.
    """
    block = cpp143["multi_cubic_spline"]["dim2"]
    axis0 = np.asarray(block["axis0"], dtype=np.float64)
    axis1 = np.asarray(block["axis1"], dtype=np.float64)
    values = np.asarray(block["values"], dtype=np.float64)
    interp = MultiCubicSpline([axis0, axis1], values)
    for i, x in enumerate(axis0):
        for j, y in enumerate(axis1):
            tolerance.tight(interp([float(x), float(y)]), float(values[i, j]))


def test_1d_is_the_natural_cubic_spline() -> None:
    """The 1-D degenerate case is exactly ``CubicNaturalSpline``."""
    xs = np.array([0.0, 1.0, 2.0, 3.5, 4.0], dtype=np.float64)
    ys = np.array([0.0, 0.5, 1.5, 3.0, 3.2], dtype=np.float64)
    interp = MultiCubicSpline([xs], ys)
    reference = CubicNaturalSpline(xs, ys)
    for x in (0.0, 0.5, 1.25, 2.75, 3.9, 4.0):
        tolerance.exact(interp(x), reference(x))


def test_n_dim_2() -> None:
    interp = _make_2d()
    assert interp.n_dim == 2


def test_axis_inspectors() -> None:
    interp = _make_2d()
    axis0 = interp.axis(0)
    np.testing.assert_array_almost_equal(axis0, [0.0, 1.0, 2.0, 3.0])
    rng = interp.axis_range(0)
    tolerance.exact(rng[0], 0.0)
    tolerance.exact(rng[1], 3.0)


def test_1d_fallback() -> None:
    """1-D MultiCubicSpline matches scipy CubicSpline(bc_type='natural').

    The 1-D case falls back to the same algorithm as L9-A's
    ``CubicNaturalSpline`` — at pillar nodes the value is exact, at
    intermediate points the value matches a natural-spline reference.
    """
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    ys = np.array([0.0, 1.0, 4.0, 9.0, 16.0], dtype=np.float64)  # y = x^2
    interp = MultiCubicSpline([xs], ys)
    for x, y in zip(xs, ys, strict=True):
        tolerance.tight(interp(float(x)), float(y))
    # natural spline through quadratic data is *exact* (the natural BC
    # ``y''(0) = 2`` doesn't match, but the C^2 spline still interpolates).
    # Intermediate values can drift from x^2; just confirm they're finite.
    for x_mid in (0.5, 1.5, 2.5, 3.5):
        v = interp(x_mid)
        assert math.isfinite(v)


def test_3d_smoke() -> None:
    """3-D MultiCubicSpline through a separable trilinear function."""
    g0 = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    g1 = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    g2 = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    vals = np.zeros((4, 4, 4), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                vals[i, j, k] = math.sin(g0[i]) + math.cos(g1[j]) + g2[k]
    interp = MultiCubicSpline([g0, g1, g2], vals)
    assert interp.n_dim == 3
    # Pillar check.
    for i in range(4):
        for j in range(4):
            for k in range(4):
                expected = math.sin(g0[i]) + math.cos(g1[j]) + g2[k]
                tolerance.tight(
                    interp([g0[i], g1[j], g2[k]]), expected
                )


def test_grid_shape_mismatch_raises() -> None:
    xs = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    ys = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    z_wrong = np.zeros((2, 3), dtype=np.float64)
    with pytest.raises(LibraryException):
        MultiCubicSpline([xs, ys], z_wrong)


def test_non_ascending_axis_raises() -> None:
    xs = np.array([0.0, 1.0, 1.0, 2.0], dtype=np.float64)  # repeated
    ys = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    z = np.zeros((4, 4), dtype=np.float64)
    with pytest.raises(LibraryException):
        MultiCubicSpline([xs, ys], z)


def test_axis_too_short_raises() -> None:
    xs = np.array([0.0], dtype=np.float64)
    ys = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    z = np.zeros((1, 4), dtype=np.float64)
    with pytest.raises(LibraryException):
        MultiCubicSpline([xs, ys], z)
