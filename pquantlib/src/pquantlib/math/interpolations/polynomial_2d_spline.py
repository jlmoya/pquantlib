"""Polynomial2DSpline — parabolic in y, cubic-spline in x.

# C++ parity: ql/experimental/inflation/polynomial2Dspline.hpp (v1.42.1) —
   ``Polynomial2DSpline`` + ``detail::Polynomial2DSplineImpl``.

The C++ surface builds, per **column** of the z matrix (i.e. per x value),
a 1-D :cpp:`Parabolic` interpolation in the y direction. ``Parabolic`` is a
``CubicInterpolation`` with the ``Parabolic`` derivative-approximation
(3-point central-difference Hermite slopes) and natural (second-derivative
= 0) boundary conditions. To evaluate ``value(x, y)`` it:

1. samples each column's parabolic interpolation at the query ``y`` (with
   extrapolation enabled) to build a 1-D ``section`` over the x grid, then
2. fits a **natural cubic spline** (``CubicInterpolation::Spline`` +
   ``SecondDerivative = 0`` boundary) through ``(x, section)`` and returns
   its value at the query ``x`` (extrapolation enabled).

**Divergence — not scipy ``RectBivariateSpline``.** A tensor-product
bivariate spline does not reproduce QuantLib's *parabolic-in-y /
cubic-in-x* construction (different basis, different boundary handling),
so we port the algorithm directly:

* the y-direction Parabolic-derivative cubic is implemented inline here
  (PQuantLib's :class:`CubicInterpolation` only supports the ``Spline``
  derivative-approximation, not ``Parabolic``);
* the x-direction natural spline reuses
  :class:`pquantlib.math.interpolations.cubic_interpolation.CubicNaturalSpline`
  (delegates to ``scipy.interpolate.CubicSpline(bc_type='natural')``).

**Indexing convention** matches :class:`Interpolation2D`: ``z[y_index,
x_index]`` — rows are y (strikes), columns are x (maturities). This is the
same orientation as the C++ ``zData_[j][i]`` (``j`` = y row, ``i`` = x
column).
"""

from __future__ import annotations

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.interpolations.cubic_interpolation import CubicNaturalSpline
from pquantlib.math.interpolations.interpolation_2d import Interpolation2D
from pquantlib.math.matrix import Matrix


def _parabolic_cubic_coeffs(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-segment Hermite coefficients ``(a, b, c)`` of a Parabolic cubic.

    # C++ parity: ``detail::CubicInterpolationImpl::calculate`` with
    # ``da_ == Parabolic`` and ``monotonic_ == false`` (cubicinterpolation.hpp
    # :577-583 for the slopes, :754-759 for the coefficients).

    On each interval ``[x[i], x[i+1]]`` the polynomial is
    ``y[i] + dx*(a[i] + dx*(b[i] + dx*c[i]))`` with ``dx = u - x[i]``.
    """
    n = x.shape[0]
    dx = np.diff(x)
    s = np.diff(y) / dx  # secant slopes S_[i]
    tmp = np.empty(n, dtype=np.float64)  # Hermite node slopes

    if n == 2:
        tmp[0] = tmp[1] = s[0]
    else:
        # interior parabolic central-difference slopes
        for i in range(1, n - 1):
            tmp[i] = (dx[i - 1] * s[i] + dx[i] * s[i - 1]) / (dx[i] + dx[i - 1])
        # parabolic end-point slopes
        tmp[0] = ((2.0 * dx[0] + dx[1]) * s[0] - dx[0] * s[1]) / (dx[0] + dx[1])
        tmp[n - 1] = (
            (2.0 * dx[n - 2] + dx[n - 3]) * s[n - 2] - dx[n - 2] * s[n - 3]
        ) / (dx[n - 2] + dx[n - 3])

    a = tmp[:-1].copy()
    b = (3.0 * s - tmp[1:] - 2.0 * tmp[:-1]) / dx
    c = (tmp[1:] + tmp[:-1] - 2.0 * s) / (dx * dx)
    return a, b, c


class _ParabolicColumn:
    """A single-column Parabolic-derivative cubic interpolation in y.

    Evaluates with C++ ``locate``-style clamping so that out-of-range
    queries extrapolate from the boundary segment (the C++ surface always
    calls ``polynomials_[i](y, true)``).
    """

    def __init__(self, x: Array, y: Array) -> None:
        self._x: np.ndarray = np.ascontiguousarray(x, dtype=np.float64)
        self._y: np.ndarray = np.ascontiguousarray(y, dtype=np.float64)
        qassert.require(self._x.shape[0] >= 2, "Parabolic needs at least 2 points")
        self._a, self._b, self._c = _parabolic_cubic_coeffs(self._x, self._y)

    def __call__(self, u: float) -> float:
        # # C++ parity: Interpolation::locate clamps j into [0, n-2].
        x = self._x
        n = x.shape[0]
        if u <= x[0]:
            j = 0
        elif u >= x[-1]:
            j = n - 2
        else:
            j = int(np.searchsorted(x, u, side="right")) - 1
        d = u - x[j]
        return float(self._y[j] + d * (self._a[j] + d * (self._b[j] + d * self._c[j])))


class Polynomial2DSpline(Interpolation2D):
    """Parabolic-in-y, natural-cubic-spline-in-x 2-D interpolation.

    Args mirror :class:`Interpolation2D`: ``xs`` (x grid, the spline
    direction), ``ys`` (y grid, the parabolic direction), ``z`` indexed
    ``[y, x]``.
    """

    def __init__(self, xs: Array, ys: Array, z: Matrix) -> None:
        super().__init__(xs, ys, z, required_points=2)
        # Build one Parabolic column per x value (over the y grid).
        # z is [y, x] so column k (fixed x) is z[:, k].
        self._columns: list[_ParabolicColumn] = [
            _ParabolicColumn(self._ys, self._z[:, k]) for k in range(self._xs.shape[0])
        ]

    def _value(self, x: float, y: float) -> float:
        # # C++ parity: Polynomial2DSplineImpl::value (polynomial2Dspline.hpp:57-73).
        section = np.array([col(y) for col in self._columns], dtype=np.float64)
        spline = CubicNaturalSpline(self._xs, section)
        return spline(x, allow_extrapolation=True)
