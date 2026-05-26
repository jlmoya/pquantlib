"""Log-linear interpolation between discrete points.

# C++ parity: ql/math/interpolations/loginterpolation.hpp (v1.42.1) —
# ``class LogLinearInterpolation`` + ``detail::LogInterpolationImpl``.

LogLinear interpolates linearly in log-space: it builds a
``LinearInterpolation`` over ``(xs, log(ys))`` and exponentiates the
result. All ``ys`` must be strictly positive (otherwise ``log`` is
undefined and ``LibraryException`` is raised on construction).

The C++ ``LogInterpolationImpl`` is templated over the underlying
interpolation (``Linear`` here), but the Python port specializes to
the linear case since LogCubic is deferred per the L1-E carve-outs.
"""

from __future__ import annotations

import math

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.interpolations.interpolation import Interpolation
from pquantlib.math.interpolations.linear import LinearInterpolation


class LogLinearInterpolation(Interpolation):
    """Log-linear interpolation: linear in ``log(y)`` between knots."""

    def __init__(self, x_seq: Array, y_seq: Array) -> None:
        super().__init__(x_seq, y_seq, required_points=2)
        # C++ ``LogInterpolationImpl::update`` requires all y > 0.
        qassert.require(
            bool(np.all(self._ys > 0.0)),
            "LogLinearInterpolation requires strictly positive y values",
        )
        log_ys: Array = np.log(self._ys)
        self._inner: LinearInterpolation = LinearInterpolation(self._xs, log_ys)

    def update(self) -> None:
        # Rebuild the inner linear interpolation if the underlying data
        # changed. The Python port owns its data so this is rarely needed.
        qassert.require(
            bool(np.all(self._ys > 0.0)),
            "LogLinearInterpolation requires strictly positive y values",
        )
        log_ys: Array = np.log(self._ys)
        self._inner = LinearInterpolation(self._xs, log_ys)

    def _value(self, x: float) -> float:
        return math.exp(self._inner(x, allow_extrapolation=True))

    def _derivative(self, x: float) -> float:
        # d/dx exp(L(x)) = exp(L(x)) * L'(x)
        return self._value(x) * self._inner.derivative(x, allow_extrapolation=True)

    def _second_derivative(self, x: float) -> float:
        # d²/dx² exp(L(x)) = exp(L(x)) * L'(x)² (since L is piecewise linear,
        # L'' = 0). The piecewise-linear second derivative vanishes between
        # knots; at knots the value is undefined and we return the right-side
        # limit by convention (C++ does not expose this either).
        slope = self._inner.derivative(x, allow_extrapolation=True)
        return self._value(x) * slope * slope
