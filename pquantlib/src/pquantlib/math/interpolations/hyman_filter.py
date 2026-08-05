"""Hyman-1983 monotonicity-filtered natural cubic spline.

# C++ parity: ql/math/interpolations/cubicinterpolation.hpp — the
#             ``Spline + monotonic=true + SecondDerivative=0`` arm of
#             ``CubicInterpolation`` (v1.43).

QuantLib's ``CubicInterpolation`` with ``DerivativeApprox::Spline``,
``BoundaryCondition::SecondDerivative = 0`` (natural BC) and
``monotonic = true`` is *not* the Fritsch-Carlson PCHIP. It:

1. solves the natural-cubic-spline tridiagonal system for the first
   derivatives at the knots (so ``y''`` vanishes at both ends);
2. applies Hyman's (1983) monotonicity filter to those derivatives,
   clipping each slope to a bound built from the surrounding chord
   slopes;
3. re-emits cubic-Hermite coefficients from the *filtered* derivatives.

``scipy.interpolate.PchipInterpolator`` derives its slopes from a
three-point stencil instead of filtering a C² spline. Both are
monotonicity-preserving cubics through the same knots, which is why a
pillar-only test cannot tell them apart; between the pillars they are
different functions.

Since the whole ``CubicInterpolation`` matrix is now transcribed in
:mod:`pquantlib.math.interpolations.cubic_interpolation`, this module is
the named entry point for that one cell rather than a second copy of the
arithmetic: :class:`HymanFilteredCubic` is
``CubicInterpolation(Spline, monotonic=True, natural, natural)``, i.e.
exactly what :class:`~pquantlib.math.interpolations.cubic_interpolation.MonotonicCubicNaturalSpline`
is. Keeping the name keeps the existing call sites and the regression
test that pins the PCHIP defect from coming back.

The Hyman criterion is described in:

    R. L. Dougherty, A. Edelman, J. M. Hyman.
    *Nonnegativity-, Monotonicity-, or Convexity-preserving Cubic and
    Quintic Hermite Interpolation.*
    Mathematics of Computation, 52(186):471-494, 1989.
"""

from __future__ import annotations

import math
from typing import final

import numpy as np

from pquantlib.math.array import Array
from pquantlib.math.interpolations.cubic_interpolation import (
    BoundaryCondition,
    CubicInterpolation,
    DerivativeApprox,
)


@final
class HymanFilteredCubic(CubicInterpolation):
    """Natural cubic spline with the Hyman-1983 monotonicity filter.

    # C++ parity: ``CubicInterpolation(Spline, monotonic=true,
    #             SecondDerivative 0.0, SecondDerivative 0.0)``.
    """

    def __init__(self, x_seq: Array, y_seq: Array) -> None:
        super().__init__(
            x_seq,
            y_seq,
            DerivativeApprox.Spline,
            True,
            BoundaryCondition.SecondDerivative,
            0.0,
            BoundaryCondition.SecondDerivative,
            0.0,
        )


def is_monotone(values: Array, tol: float = 0.0) -> bool:
    """Check whether ``values`` is weakly monotone (up or down).

    Used by tests to verify the Hyman filter's monotonicity-preservation
    contract: interpolated values on a monotone input stay monotone.
    """
    diff = np.diff(values)
    if diff.size == 0:
        return True
    up = bool(np.all(diff >= -tol))
    down = bool(np.all(diff <= tol))
    return (up or down) and math.isfinite(float(values[0]))


__all__ = ["HymanFilteredCubic", "is_monotone"]
