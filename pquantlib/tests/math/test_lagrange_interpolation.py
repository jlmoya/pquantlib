"""Tests for LagrangeInterpolation (barycentric Lagrange).

C++ parity: ql/math/interpolations/lagrangeinterpolation.hpp @ v1.42.1.

No probe needed — the interpolation is exact for polynomials of degree
< node count, verified analytically.
"""

from __future__ import annotations

import numpy as np

from pquantlib.math.interpolations.lagrange_interpolation import LagrangeInterpolation
from pquantlib.testing.tolerance import loose, tight


def _quad_interp() -> LagrangeInterpolation:
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    f = LagrangeInterpolation(xs, xs**2)
    f.enable_extrapolation()
    return f


def test_exact_for_quadratic() -> None:
    f = _quad_interp()
    for x in (0.5, 1.5, 2.5, 3.7):
        tight(f(x), x * x)


def test_node_snap() -> None:
    f = _quad_interp()
    for i, x in enumerate((0.0, 1.0, 2.0, 3.0, 4.0)):
        tight(f(x), x * x, reason=f"node {i} returns its y-value exactly")


def test_value_with_fresh_y() -> None:
    """``value_with`` reuses cached weights against a new y-vector."""
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    f = LagrangeInterpolation(xs, xs**2)
    f.enable_extrapolation()
    # 5 nodes -> exact for cubic
    tight(f.value_with(xs**3, 2.5), 2.5**3)
    tight(f.value_with(xs**3, 1.5), 1.5**3)


def test_derivative_of_quadratic() -> None:
    f = _quad_interp()
    # d/dx x^2 = 2x
    loose(f.derivative(1.5), 3.0)
    loose(f.derivative(2.5), 5.0)
    # at a node
    loose(f.derivative(2.0), 4.0)


def test_required_points_one() -> None:
    """A single node returns its y-value everywhere."""
    f = LagrangeInterpolation(
        np.array([2.0], dtype=np.float64), np.array([7.0], dtype=np.float64)
    )
    f.enable_extrapolation()
    tight(f(2.0), 7.0)
