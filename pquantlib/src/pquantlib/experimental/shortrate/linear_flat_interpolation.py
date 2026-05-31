"""LinearFlatInterpolation — linear interpolation with flat extrapolation.

# C++ parity: ql/experimental/shortrate/generalizedhullwhite.hpp:310-382
# (v1.42.1) — ``class LinearFlatInterpolation`` + ``LinearFlat`` factory +
# ``detail::LinearFlatInterpolationImpl``.

Behaves exactly like ordinary linear interpolation strictly inside the
node range, but clamps to the boundary y-values for x outside
``[x_min, x_max]`` (flat extrapolation) — including the ``value`` itself,
not just the derivative. This is the interpolator ``GeneralizedHullWhite``
uses for its piecewise reversion / volatility structures (with
``enable_extrapolation()`` always on).

The ``LinearFlat`` companion class is the C++ traits/factory object
(``global = False``, ``requiredPoints = 1``); the Python port exposes a
single ``interpolate(x, y)`` classmethod-style builder.
"""

from __future__ import annotations

from typing import final

import numpy as np

from pquantlib.math.array import Array
from pquantlib.math.interpolations.interpolation import Interpolation


@final
class LinearFlatInterpolation(Interpolation):
    """Linear interpolation with flat extrapolation.

    # C++ parity: ``class LinearFlatInterpolation : public Interpolation``
    # in generalizedhullwhite.hpp:312-323 (Impl at 338-382).

    The constructor requires only one point (C++ ``requiredPoints = 1``);
    with a single node every query returns that node's y-value.
    """

    __slots__ = ("_primitive_const", "_slopes")

    def __init__(self, x_seq: Array, y_seq: Array) -> None:
        super().__init__(x_seq, y_seq, required_points=1)
        n = self._xs.shape[0]
        self._slopes: Array = np.zeros(n, dtype=np.float64)
        self._primitive_const: Array = np.zeros(n, dtype=np.float64)
        self.update()

    def update(self) -> None:
        # C++ parity: LinearFlatInterpolationImpl::update (lines 348-356).
        xs = self._xs
        ys = self._ys
        n = xs.shape[0]
        self._primitive_const[0] = 0.0
        for i in range(1, n):
            dx = float(xs[i] - xs[i - 1])
            self._slopes[i - 1] = (float(ys[i]) - float(ys[i - 1])) / dx
            self._primitive_const[i] = self._primitive_const[i - 1] + dx * (
                float(ys[i - 1]) + 0.5 * dx * self._slopes[i - 1]
            )

    def _value(self, x: float) -> float:
        # C++ parity: LinearFlatInterpolationImpl::value (lines 357-364) —
        # flat outside the range, linear inside.
        if x <= self.x_min:
            return float(self._ys[0])
        if x >= self.x_max:
            return float(self._ys[-1])
        i = self._locate(x)
        return float(self._ys[i]) + (x - float(self._xs[i])) * self._slopes[i]

    def _primitive(self, x: float) -> float:
        # C++ parity: LinearFlatInterpolationImpl::primitive (lines 365-370).
        i = self._locate(x)
        dx = x - float(self._xs[i])
        return self._primitive_const[i] + dx * (
            float(self._ys[i]) + 0.5 * dx * self._slopes[i]
        )

    def _derivative(self, x: float) -> float:
        # C++ parity: LinearFlatInterpolationImpl::derivative (lines 371-376)
        # — zero outside the range, slope of the bracketing segment inside.
        if not self.is_in_range(x):
            return 0.0
        i = self._locate(x)
        return float(self._slopes[i])

    def _second_derivative(self, x: float) -> float:
        # C++ parity: LinearFlatInterpolationImpl::secondDerivative -> 0.0.
        del x
        return 0.0


@final
class LinearFlat:
    """Linear-flat interpolation factory / traits object.

    # C++ parity: ``class LinearFlat`` in generalizedhullwhite.hpp:327-336.
    """

    global_ = False  # C++ ``static const bool global = false``.
    required_points = 1  # C++ ``static const Size requiredPoints = 1``.

    @staticmethod
    def interpolate(x_seq: Array, y_seq: Array) -> LinearFlatInterpolation:
        """Build a ``LinearFlatInterpolation`` over ``(x, y)``."""
        return LinearFlatInterpolation(x_seq, y_seq)


__all__ = ["LinearFlat", "LinearFlatInterpolation"]
