"""Forward-flat (piecewise-constant, left-continuous) interpolation.

# C++ parity: ql/math/interpolations/forwardflatinterpolation.hpp (v1.42.1).

For ``xs[i] <= x < xs[i+1]`` the value is ``ys[i]`` — i.e. each step
holds the *left-hand* y value (constant going forward from the current
knot). At the right boundary ``x >= xs[-1]`` returns ``ys[-1]``.

C++ ``ForwardFlat::requiredPoints = 2`` — a single-knot curve is not
valid here (unlike ``BackwardFlat``). ``derivative`` and
``second_derivative`` are 0. ``primitive(x) = primitive_[i] + dx*ys[i]``.
"""

from __future__ import annotations

import numpy as np

from pquantlib.math.array import Array
from pquantlib.math.interpolations.interpolation import Interpolation


class ForwardFlatInterpolation(Interpolation):
    """Piecewise-constant interpolation taking the left-hand knot value."""

    def __init__(self, x_seq: Array, y_seq: Array) -> None:
        super().__init__(x_seq, y_seq, required_points=2)
        n = self._xs.shape[0]
        self._primitive_const: Array = np.zeros(n, dtype=np.float64)
        self.update()

    def update(self) -> None:
        # C++ parity: primitive_[0]=0, primitive_[i] = primitive_[i-1] + dx*y[i-1].
        xs = self._xs
        ys = self._ys
        n = xs.shape[0]
        prim = self._primitive_const
        prim[0] = 0.0
        for i in range(1, n):
            dx = float(xs[i] - xs[i - 1])
            prim[i] = prim[i - 1] + dx * float(ys[i - 1])

    def _value(self, x: float) -> float:
        xs = self._xs
        ys = self._ys
        n = xs.shape[0]
        if x >= float(xs[n - 1]):
            return float(ys[n - 1])
        i = self._locate(x)
        return float(ys[i])

    def _primitive(self, x: float) -> float:
        i = self._locate(x)
        dx = x - float(self._xs[i])
        return float(self._primitive_const[i]) + dx * float(self._ys[i])

    def _derivative(self, x: float) -> float:
        # Piecewise-constant — derivative is zero everywhere (C++ parity).
        del x
        return 0.0
