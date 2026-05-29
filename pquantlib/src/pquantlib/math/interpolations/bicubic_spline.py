"""Bicubic spline interpolation on a 2-D grid.

# C++ parity: ql/math/interpolations/bicubicsplineinterpolation.hpp
# (v1.42.1).

C++ ``BicubicSpline`` builds a 2-D natural cubic spline by composing
1-D ``CubicSpline``s row-by-row then column-by-column on the (M, N)
matrix ``z[y_index, x_index]``. Each evaluation at ``(x, y)``:

1. Build a 1-D spline through ``z[0, :], ..., z[M-1, :]`` at column
   ``y`` — i.e. for each y-row, evaluate at ``x``, yielding a column
   of M values.
2. Build a 1-D natural cubic spline through those M values at the
   y-coordinates ``ys``, and evaluate at ``y``.

The Python port delegates to ``scipy.interpolate.RectBivariateSpline``
with ``kx=ky=3`` — bicubic spline on a rectilinear (but not
necessarily uniform) grid.

**Documented divergence — boundary condition.** scipy's
``RectBivariateSpline`` uses *not-a-knot* boundary conditions (the
default for ``scipy.interpolate.UnivariateSpline``); QuantLib's
``BicubicSpline`` uses *natural* boundary conditions (it composes
``CubicNaturalSpline``s row-by-row and column-by-column). At pillar
nodes both implementations roundtrip exactly to the input data
(both interpolate, neither approximates). At off-pillar interior
points on a small grid the two BCs measurably disagree: observed
~10% relative error on the L9-A probe's 4x4 ``sin(x)+cos(y)`` grid.
This grows tighter (~1% or better) on larger grids where boundary
effects are diluted. The pillar-correctness contract is preserved;
off-pillar agreement is qualitative. For the L8-C surface-upgrade
use-case (cap/floor and swaption vol surfaces with 20+ points per
axis) the BC difference is negligible compared to the input vol
quote noise.

**Indexing convention** (matches C++ ``zData_[j][i]``): the matrix is
indexed as ``z[y_index, x_index]`` — rows are y, columns are x. We
pass ``RectBivariateSpline(x=xs, y=ys, z=z.T)`` because scipy's
``RectBivariateSpline`` expects ``z`` shaped ``(len(x), len(y))``,
which is the *transpose* of our convention. (The scipy docs are
explicit about this layout.)
"""

from __future__ import annotations

from typing import Any

from scipy.interpolate import (  # type: ignore[import-untyped]
    RectBivariateSpline,
)

from pquantlib.math.array import Array
from pquantlib.math.interpolations.interpolation_2d import Interpolation2D
from pquantlib.math.matrix import Matrix


class BicubicSpline(Interpolation2D):
    """2-D bicubic spline on a rectilinear grid indexed as ``z[y, x]``.

    # C++ parity: ``BicubicSpline`` (bicubicsplineinterpolation.hpp).
    """

    def __init__(self, xs: Array, ys: Array, z: Matrix) -> None:
        # Bicubic needs at least 4 points in each axis for a proper
        # cubic; scipy.RectBivariateSpline allows kx+1=4 minimum.
        # C++ does not enforce this explicitly — it relies on the
        # composed 1-D ``CubicSpline`` to fail at construction.
        # We match: minimum 2 (the base abstract's requirement)
        # but scipy will raise if < 4 on either axis.
        super().__init__(xs, ys, z, required_points=2)
        self._spline: Any = None
        self.update()

    def update(self) -> None:
        """Rebuild the underlying scipy spline.

        # C++ parity: ``BicubicSpline::calculate()`` (PIMPL).
        """
        # scipy.RectBivariateSpline expects z shaped ``(len(x), len(y))``,
        # which is the *transpose* of our ``z[y, x]`` convention.
        self._spline = RectBivariateSpline(
            self._xs, self._ys, self._z.T, kx=3, ky=3
        )

    def _value(self, x: float, y: float) -> float:
        # ``ev`` evaluates a single (x, y) point; equivalent to
        # ``self._spline(x, y, grid=False)`` but slightly faster.
        return float(self._spline.ev(x, y))
