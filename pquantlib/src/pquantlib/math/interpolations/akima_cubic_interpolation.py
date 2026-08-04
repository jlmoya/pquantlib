"""Akima (1970) cubic spline interpolation.

# C++ parity: ql/math/interpolations/cubicinterpolation.hpp
#             ``class AkimaCubicInterpolation`` (v1.42.1).

Akima's method is a piecewise cubic spline that is *local* and
*non-monotonic*. The slope at each knot is a weighted average of the
slopes of the four neighboring segments; the weights are designed to
suppress wiggles near sharp transitions. Compared to a natural cubic
spline, Akima trades smoothness (C^2 -> C^1) for monotonicity
preservation in flat regions.

The C++ implementation derives Akima as a special case of the generic
``CubicInterpolation`` (the full DA / SecondDerivative / FirstDerivative
boundary-condition matrix). Since pquantlib has not yet ported the full
``CubicInterpolation`` family (deferred), this stub uses
``scipy.interpolate.Akima1DInterpolator`` — scipy provides Akima 1970
with the same *interior* per-knot slope weighting formula.

**Documented divergence — boundary slopes.** C++ QuantLib uses a
nonlinear endpoint slope formula (cubicinterpolation.hpp:606-621)
that is *not* the standard Akima 1970 reflection rule. scipy uses
the standard rule (and the SciPy ``method='akima'`` documentation
cites Akima 1970). On C^2 inputs (e.g. y = x^2 on a uniform grid),
scipy recovers x^2 exactly while C++ produces visible boundary
artefacts. We accept scipy's behaviour as the more faithful Akima
1970 port and tier the cross-validation against the C++ probe at
LOOSE on the boundary cells.

Interior cells (after the second knot from each boundary) agree
to TIGHT tier; the endpoint cubic differs algorithmically. The full
``CubicInterpolation`` family port is carved out for a later stage.
"""

from __future__ import annotations

from typing import Any

from scipy.interpolate import Akima1DInterpolator  # type: ignore[import-untyped]

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.interpolations.interpolation import Interpolation


class AkimaCubicInterpolation(Interpolation):
    """Akima 1970 cubic spline.

    # C++ parity: ``AkimaCubicInterpolation`` (cubicinterpolation.hpp:258).
    """

    def __init__(self, x_seq: Array, y_seq: Array) -> None:
        super().__init__(x_seq, y_seq, required_points=2)
        # C++ parity: cubicinterpolation.hpp:402-406 (v1.43), checked inside
        # ``CubicInterpolationImpl::update()`` — i.e. after the structural
        # length checks, which is why this sits below ``super().__init__``.
        # The Akima slope estimates read S_[2] and S_[n-4]; with three points
        # S_ has size two and both indices are out of bounds. v1.42.1 had no
        # guard and read past the end via unsigned wrap-around; this port used
        # to record the absence of a guard as deliberate C++ parity, which it
        # never was.
        n = int(self._xs.shape[0])
        qassert.require(
            n >= 4,
            f"Akima approximation requires at least 4 points ({n} are given)",
        )
        self._spline: Any = Akima1DInterpolator(self._xs, self._ys)
        # Cache first and second derivative splines (scipy provides
        # them via .derivative(1) and .derivative(2)).
        self._d1: Any = self._spline.derivative(1)
        self._d2: Any = self._spline.derivative(2)

    def update(self) -> None:
        # C++ parity: ``CubicInterpolation::update()`` reconstructs the
        # tridiagonal system. scipy does the same internally; rebuild
        # the cached splines so callers can mutate the underlying
        # arrays and force a refresh.
        self._spline = Akima1DInterpolator(self._xs, self._ys)
        self._d1 = self._spline.derivative(1)
        self._d2 = self._spline.derivative(2)

    def _value(self, x: float) -> float:
        return float(self._spline(x))

    def _derivative(self, x: float) -> float:
        return float(self._d1(x))

    def _second_derivative(self, x: float) -> float:
        return float(self._d2(x))
