"""Bilinear interpolation over a 2-D grid.

# C++ parity: ql/math/interpolations/bilinearinterpolation.hpp (v1.43)
# + ql/math/interpolations/interpolation2d.hpp ``Interpolation2D``.

Standard textbook bilinear interpolation on a rectilinear (not necessarily
uniform) grid. Given sorted ``xs`` (length ``n``) and ``ys`` (length ``m``)
and a matrix ``z`` shaped ``(m, n)``, the value at ``(x, y)`` is::

    z(x, y) = (1-t)(1-u)*z[j,i] + t(1-u)*z[j,i+1]
            + (1-t)*u*z[j+1,i] + t*u*z[j+1,i+1]

where ``i = locate_x(x)``, ``j = locate_y(y)``,
``t = (x - xs[i]) / (xs[i+1] - xs[i])``,
``u = (y - ys[j]) / (ys[j+1] - ys[j])``.

Because ``locate_x``/``locate_y`` clamp to the last cell, out-of-range
arguments produce a *linear extrapolation* off the edge cell (``t`` or ``u``
outside ``[0, 1]``), not a clamp. That is C++'s behaviour too; wrap in
:class:`~pquantlib.math.interpolations.flat_extrapolation_2d.FlatExtrapolator2D`
for the clamped variant.

**Indexing convention** (matches C++ ``zData_[j][i]``): the matrix is
indexed as ``z[y_index, x_index]`` — rows are y, columns are x.
"""

from __future__ import annotations

from pquantlib.math.array import Array
from pquantlib.math.interpolations.interpolation_2d import Interpolation2D
from pquantlib.math.matrix import Matrix


class BilinearInterpolation(Interpolation2D):
    """Bilinear interpolation on a 2-D grid indexed as ``z[y, x]``.

    # C++ parity: ``class BilinearInterpolation : public Interpolation2D``
    # (bilinearinterpolation.hpp:72-84), Impl at lines 34-63.
    """

    __slots__ = ()

    def __init__(self, xs: Array, ys: Array, z: Matrix) -> None:
        super().__init__(xs, ys, z, required_points=2)

    def _value(self, x: float, y: float) -> float:
        # C++ parity: bilinearinterpolation.hpp:47-61.
        i = self.locate_x(x)
        j = self.locate_y(y)
        z = self._z
        z1 = float(z[j, i])
        z2 = float(z[j, i + 1])
        z3 = float(z[j + 1, i])
        z4 = float(z[j + 1, i + 1])
        t = (x - float(self._xs[i])) / float(self._xs[i + 1] - self._xs[i])
        u = (y - float(self._ys[j])) / float(self._ys[j + 1] - self._ys[j])
        return (1.0 - t) * (1.0 - u) * z1 + t * (1.0 - u) * z2 + (1.0 - t) * u * z3 + t * u * z4


class Bilinear:
    """Bilinear-interpolation factory.

    # C++ parity: ``class Bilinear`` (bilinearinterpolation.hpp:87-95).

    C++ uses this as a template *traits* type: ``Interpolator2D`` template
    parameters are instantiated and their ``interpolate`` member called. The
    Python counterpart is an instance with an :meth:`interpolate` method, so
    ``Bilinear()`` can be passed anywhere a 2-D interpolator factory is
    expected.
    """

    __slots__ = ()

    def interpolate(self, xs: Array, ys: Array, z: Matrix) -> BilinearInterpolation:
        """Build a :class:`BilinearInterpolation` over ``(xs, ys, z)``."""
        return BilinearInterpolation(xs, ys, z)


__all__ = ["Bilinear", "BilinearInterpolation"]
