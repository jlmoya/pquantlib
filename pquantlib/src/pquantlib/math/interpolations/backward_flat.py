"""Backward-flat (piecewise-constant, right-continuous) interpolation.

# C++ parity: ql/math/interpolations/backwardflatinterpolation.hpp (v1.42.1).

For ``xs[i] < x <= xs[i+1]`` the value is ``ys[i+1]`` — i.e. each step
takes the *right-hand* y value (constant going backward from the next
knot). At the left boundary ``x <= xs[0]`` returns ``ys[0]``.

C++ ``BackwardFlat::requiredPoints = 1`` — a single-knot curve is valid
and returns ``ys[0]`` everywhere. ``derivative`` and ``second_derivative``
are 0 (piecewise constant). ``primitive(x) = primitive_[i] + dx*ys[i+1]``.
"""

from __future__ import annotations

import numpy as np

from pquantlib.math.array import Array
from pquantlib.math.interpolations.interpolation import Interpolation


class BackwardFlatInterpolation(Interpolation):
    """Piecewise-constant interpolation taking the right-hand knot value."""

    def __init__(self, x_seq: Array, y_seq: Array) -> None:
        super().__init__(x_seq, y_seq, required_points=1)
        n = self._xs.shape[0]
        self._primitive_const: Array = np.zeros(n, dtype=np.float64)
        self.update()

    def update(self) -> None:
        # C++ parity: primitive_[0]=0, primitive_[i] = primitive_[i-1] + dx*y[i].
        xs = self._xs
        ys = self._ys
        n = xs.shape[0]
        prim = self._primitive_const
        prim[0] = 0.0
        for i in range(1, n):
            dx = float(xs[i] - xs[i - 1])
            prim[i] = prim[i - 1] + dx * float(ys[i])

    def _value(self, x: float) -> float:
        xs = self._xs
        ys = self._ys
        # C++: if x <= xs[0] or n==1, return ys[0].
        if x <= float(xs[0]) or xs.shape[0] == 1:
            return float(ys[0])
        i = self._locate(x)
        if x == float(xs[i]):
            return float(ys[i])
        return float(ys[i + 1])

    def _primitive(self, x: float) -> float:
        xs = self._xs
        ys = self._ys
        if xs.shape[0] == 1:
            return (x - float(xs[0])) * float(ys[0])
        i = self._locate(x)
        dx = x - float(xs[i])
        return float(self._primitive_const[i]) + dx * float(ys[i + 1])

    def _derivative(self, x: float) -> float:
        # Piecewise-constant — derivative is zero everywhere (C++ parity).
        del x
        return 0.0
