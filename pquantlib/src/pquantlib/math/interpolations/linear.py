"""Linear interpolation between discrete points.

# C++ parity: ql/math/interpolations/linearinterpolation.hpp (v1.42.1).

Piecewise linear: between ``xs[i]`` and ``xs[i+1]``, the interpolated
value is ``ys[i] + (x - xs[i]) * s_i`` where ``s_i = (ys[i+1] - ys[i])
/ (xs[i+1] - xs[i])`` is the slope.

The C++ ``LinearInterpolationImpl::update()`` pre-computes the slopes
``s_`` and a primitive (running integral) array; we mirror that in
``update()``. ``primitive(x) = primitive_const[i] + dx*(ys[i] + 0.5*dx*s_i)``.
``derivative(x) = s_[i]``. ``second_derivative(x) = 0`` (inherited).
"""

from __future__ import annotations

import numpy as np

from pquantlib.math.array import Array
from pquantlib.math.interpolations.interpolation import Interpolation


class LinearInterpolation(Interpolation):
    """Piecewise-linear interpolation between sorted ``(x, y)`` knots."""

    def __init__(self, x_seq: Array, y_seq: Array) -> None:
        super().__init__(x_seq, y_seq, required_points=2)
        n = self._xs.shape[0]
        self._slopes: Array = np.zeros(n, dtype=np.float64)
        self._primitive_const: Array = np.zeros(n, dtype=np.float64)
        self.update()

    def update(self) -> None:
        # C++ parity: LinearInterpolationImpl::update — primitiveConst_[0] = 0,
        # primitiveConst_[i] = primitiveConst_[i-1] + dx*(y[i-1] + 0.5*dx*s_[i-1]).
        xs = self._xs
        ys = self._ys
        n = xs.shape[0]
        slopes = self._slopes
        prim = self._primitive_const
        prim[0] = 0.0
        for i in range(1, n):
            dx = float(xs[i] - xs[i - 1])
            slopes[i - 1] = float(ys[i] - ys[i - 1]) / dx
            prim[i] = prim[i - 1] + dx * (float(ys[i - 1]) + 0.5 * dx * float(slopes[i - 1]))

    def _value(self, x: float) -> float:
        i = self._locate(x)
        return float(self._ys[i]) + (x - float(self._xs[i])) * float(self._slopes[i])

    def _primitive(self, x: float) -> float:
        i = self._locate(x)
        dx = x - float(self._xs[i])
        return float(self._primitive_const[i]) + dx * (float(self._ys[i]) + 0.5 * dx * float(self._slopes[i]))

    def _derivative(self, x: float) -> float:
        return float(self._slopes[self._locate(x)])
