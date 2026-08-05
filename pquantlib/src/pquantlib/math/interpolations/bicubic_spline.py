"""Bicubic spline interpolation on a 2-D grid.

# C++ parity: ql/math/interpolations/bicubicsplineinterpolation.hpp (v1.43).

C++ ``BicubicSpline`` is **not** a tensor-product B-spline. It is a
composition of 1-D *natural* cubic splines, rebuilt on every evaluation:

1. At construction, one natural cubic spline per grid **row** — spline
   ``i`` runs over ``xs`` through ``z[i, :]``. There are ``len(ys)`` of them.
2. To evaluate at ``(x, y)``: evaluate every row spline at ``x`` (allowing
   extrapolation), giving a column of ``len(ys)`` values, then build a
   *fresh* natural cubic spline over ``ys`` through that column and
   evaluate it at ``y`` (again allowing extrapolation).

Each of those 1-D splines is ``CubicInterpolation(Spline, monotonic=false,
SecondDerivative 0.0 at both ends)`` — i.e. ``CubicNaturalSpline``.

**This module used to delegate to ``scipy.interpolate.RectBivariateSpline``**
with ``kx = ky = 3``. That is a genuinely different function: a tensor-product
B-spline with *not-a-knot* end conditions. It interpolates the same pillars,
so pillar round-trips passed, but off-pillar values were wrong by ~10 %
relative on a 4x4 grid (the old test asserted only a 0.15 relative-error
envelope and called the disagreement "qualitative"). It also could not
produce ``derivativeXY`` at all, and it refused grids with fewer than 4
points per axis, which C++ accepts. The composition above is now transcribed
directly and agrees with C++ to TIGHT.

The derivative API (C++ ``detail::BicubicSplineDerivatives``, reached through
a ``dynamic_pointer_cast`` on the impl) follows the same
build-a-fresh-spline-and-differentiate-it pattern:

- ``derivative_x`` / ``second_derivative_x``: sample ``value(xs[i], y)``
  across the x pillars, spline it over ``xs``, differentiate at ``x``.
- ``derivative_y`` / ``second_derivative_y``: sample the row splines at
  ``x``, spline that column over ``ys``, differentiate at ``y``.
- ``derivative_xy``: sample ``derivative_y(xs[i], y)`` across the x pillars,
  spline it over ``xs``, differentiate at ``x``.

Note the asymmetry C++ has here and which is preserved: ``value`` evaluates
its inner splines with extrapolation enabled, but the *derivative* entry
points build a plain ``CubicInterpolation`` and call
``derivative``/``secondDerivative`` with ``allowExtrapolation`` left at its
``false`` default. Asking for a derivative outside the grid therefore raises,
even on an interpolation with extrapolation enabled.

**Indexing convention** (matches C++ ``zData_``): ``z[y_index, x_index]`` —
rows are y, columns are x, so ``z`` is shaped ``(len(ys), len(xs))`` and
``zData_.rows()`` is ``len(ys)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from pquantlib.math.array import Array
from pquantlib.math.interpolations.cubic_interpolation import CubicNaturalSpline
from pquantlib.math.interpolations.interpolation_2d import Interpolation2D
from pquantlib.math.matrix import Matrix


class BicubicSplineDerivatives(ABC):
    """The partial-derivative interface of a bicubic surface.

    # C++ parity: ``class detail::BicubicSplineDerivatives``
    # (bicubicsplineinterpolation.hpp:35-43).

    In C++ this is a pure-virtual mix-in that ``BicubicSplineImpl`` inherits
    alongside ``Interpolation2D::templateImpl``, so that ``BicubicSpline``
    can ``dynamic_pointer_cast`` its type-erased impl back to something with
    derivatives on it. Python has no type erasure to undo, but the interface
    is still the contract a bicubic surface offers beyond
    ``Interpolation2D``, so it is kept as an ABC.
    """

    __slots__ = ()

    @abstractmethod
    def derivative_x(self, x: float, y: float) -> float:
        """d/dx of the surface at ``(x, y)``. C++ ``derivativeX``."""

    @abstractmethod
    def derivative_y(self, x: float, y: float) -> float:
        """d/dy of the surface at ``(x, y)``. C++ ``derivativeY``."""

    @abstractmethod
    def derivative_xy(self, x: float, y: float) -> float:
        """d2/dxdy of the surface at ``(x, y)``. C++ ``derivativeXY``."""

    @abstractmethod
    def second_derivative_x(self, x: float, y: float) -> float:
        """d2/dx2 of the surface at ``(x, y)``. C++ ``secondDerivativeX``."""

    @abstractmethod
    def second_derivative_y(self, x: float, y: float) -> float:
        """d2/dy2 of the surface at ``(x, y)``. C++ ``secondDerivativeY``."""


class BicubicSpline(Interpolation2D, BicubicSplineDerivatives):
    """2-D bicubic spline on a rectilinear grid indexed as ``z[y, x]``.

    # C++ parity: ``class BicubicSpline : public Interpolation2D``
    # (bicubicsplineinterpolation.hpp:163-196), Impl at lines 45-153.
    """

    __slots__ = ("_splines",)

    def __init__(self, xs: Array, ys: Array, z: Matrix) -> None:
        super().__init__(xs, ys, z, required_points=2)
        self._splines: list[CubicNaturalSpline] = []
        # C++ parity: BicubicSplineImpl's ctor calls calculate().
        self.update()

    def update(self) -> None:
        """Rebuild the per-row (per-y) natural cubic splines.

        # C++ parity: ``BicubicSplineImpl::calculate``
        # (bicubicsplineinterpolation.hpp:58-67) — one CubicInterpolation per
        # ``zData_`` ROW, over the x abscissae.
        """
        self._splines = [
            CubicNaturalSpline(self._xs, self._z[i, :]) for i in range(self._z.shape[0])
        ]

    # ----- value ----------------------------------------------------------

    def _value(self, x: float, y: float) -> float:
        # C++ parity: bicubicsplineinterpolation.hpp:68-79.
        section = self._section_at_x(x)
        return CubicNaturalSpline(self._ys, section)(y, allow_extrapolation=True)

    # ----- BicubicSplineDerivatives --------------------------------------

    def derivative_x(self, x: float, y: float) -> float:
        """d/dx at ``(x, y)``.

        # C++ parity: ``derivativeX`` (bicubicsplineinterpolation.hpp:81-93).
        """
        return CubicNaturalSpline(self._xs, self._x_section(y)).derivative(x)

    def second_derivative_x(self, x: float, y: float) -> float:
        """d2/dx2 at ``(x, y)``.

        # C++ parity: ``secondDerivativeX`` (bicubicsplineinterpolation.hpp:95-108).
        """
        return CubicNaturalSpline(self._xs, self._x_section(y)).second_derivative(x)

    def derivative_y(self, x: float, y: float) -> float:
        """d/dy at ``(x, y)``.

        # C++ parity: ``derivativeY`` (bicubicsplineinterpolation.hpp:110-121).
        """
        return CubicNaturalSpline(self._ys, self._section_at_x(x)).derivative(y)

    def second_derivative_y(self, x: float, y: float) -> float:
        """d2/dy2 at ``(x, y)``.

        # C++ parity: ``secondDerivativeY`` (bicubicsplineinterpolation.hpp:123-135).
        """
        return CubicNaturalSpline(self._ys, self._section_at_x(x)).second_derivative(y)

    def derivative_xy(self, x: float, y: float) -> float:
        """d2/dxdy at ``(x, y)``.

        # C++ parity: ``derivativeXY`` (bicubicsplineinterpolation.hpp:137-149).
        """
        section = np.array(
            [self.derivative_y(float(xi), y) for xi in self._xs], dtype=np.float64
        )
        return CubicNaturalSpline(self._xs, section).derivative(x)

    # ----- helpers --------------------------------------------------------

    def _section_at_x(self, x: float) -> Array:
        """Every row spline evaluated at ``x`` — a column of ``len(ys)`` values.

        # C++ parity: the ``section[i] = splines_[i](x, true)`` loop that
        # appears in ``value``, ``derivativeY`` and ``secondDerivativeY``.
        """
        return np.array(
            [s(x, allow_extrapolation=True) for s in self._splines], dtype=np.float64
        )

    def _x_section(self, y: float) -> Array:
        """``value(xs[i], y)`` across the x pillars — ``len(xs)`` values.

        # C++ parity: the ``section[i] = value(xBegin_[i], y)`` loop in
        # ``derivativeX`` and ``secondDerivativeX``. Note it calls the Impl's
        # ``value`` directly, i.e. WITHOUT the range check.
        """
        return np.array([self._value(float(xi), y) for xi in self._xs], dtype=np.float64)


class Bicubic:
    """Bicubic-spline-interpolation factory.

    # C++ parity: ``class Bicubic`` (bicubicsplineinterpolation.hpp:199-207).

    See :class:`~pquantlib.math.interpolations.bilinear.Bilinear` for why the
    C++ traits struct becomes an instantiable class here.
    """

    __slots__ = ()

    def interpolate(self, xs: Array, ys: Array, z: Matrix) -> BicubicSpline:
        """Build a :class:`BicubicSpline` over ``(xs, ys, z)``."""
        return BicubicSpline(xs, ys, z)


__all__ = ["Bicubic", "BicubicSpline", "BicubicSplineDerivatives"]
