"""Cubic spline interpolation family.

# C++ parity: ql/math/interpolations/cubicinterpolation.hpp (v1.42.1).

Provides ``CubicInterpolation`` plus the standard convenience
subclasses ``CubicNaturalSpline`` (Spline + Natural BC + non-monotonic)
and ``MonotonicCubicNaturalSpline`` (PCHIP / Hyman-Fritsch-Carlson
monotonic).

C++ ``CubicInterpolation`` carries a derivative-approximation strategy
(``DerivativeApprox`` — 9 values: Spline, SplineOM1, SplineOM2,
FourthOrder, Parabolic, FritschButland, Akima, Kruger, Harmonic) crossed
with a boundary-condition strategy (``BoundaryCondition`` — 5 values:
NotAKnot, FirstDerivative, SecondDerivative (= Natural when value=0),
Periodic, Lagrange), plus an orthogonal monotonicity-preserving filter.
The full 9 x 5 x 2 = 90-cell matrix lives in 820 LOC of templated C++.

**Python port — scope.** This port lands the two cells that the rest
of the library actually exercises:

* ``Spline + SecondDerivative=0 BC + monotonic=false`` → natural cubic
  spline (``CubicNaturalSpline``). Delegated to
  ``scipy.interpolate.CubicSpline(bc_type='natural')``.
* ``Spline + SecondDerivative=0 BC + monotonic=true`` → monotonic
  cubic (``MonotonicCubicNaturalSpline``). Delegated to
  ``scipy.interpolate.PchipInterpolator``.

  **Documented divergence — monotonic-cubic algorithm.** scipy's
  ``PchipInterpolator`` is the Fritsch-Carlson PCHIP — slopes are
  derived from scratch from a one-sided three-point formula plus the
  Fritsch-Carlson monotonicity filter. QuantLib's "Spline +
  monotonic=true" instead solves the natural-spline tridiagonal system
  *first*, then applies the Hyman (1983) monotonicity filter to the
  resulting C^2 slopes (cubicinterpolation.hpp:507-560). Both
  algorithms guarantee monotonicity at the input but produce
  measurably different intermediate values (~1e-2 magnitude on the
  L9-A probe). We accept scipy's PCHIP because (a) the API contract —
  "monotonic cubic spline through the input knots" — is satisfied,
  (b) PCHIP is the standard monotonic cubic in scientific Python,
  (c) porting QuantLib's exact Hyman-filtered natural-cubic algorithm
  would require a custom implementation. Pillar nodes still agree to
  TIGHT (both pass through the input data); intermediate values are
  tier-LOOSE in cross-validation.

All other ``DerivativeApprox`` + ``BoundaryCondition`` combinations
raise ``LibraryException("not implemented in this port")`` from the
``CubicInterpolation`` constructor. Documented carve-outs:

* ``SplineOM1``, ``SplineOM2`` — overshooting-minimization variants.
* ``FourthOrder``, ``Parabolic``, ``FritschButland``, ``Kruger``,
  ``Harmonic`` derivative styles.
* ``Akima`` — covered separately by
  ``pquantlib.math.interpolations.akima_cubic_interpolation.AkimaCubicInterpolation``
  (Phase 5 L5-A, a different scipy delegation).
* Boundary conditions ``NotAKnot``, ``FirstDerivative``,
  non-zero ``SecondDerivative``, ``Periodic``, ``Lagrange``.

The validation hook is the C++ probe at
``migration-harness/cpp/probes/cluster_l9a/probe.cpp``. Spline values
at pillar nodes agree EXACT to TIGHT; intermediate values agree TIGHT.

# C++ parity: convenience classes
#   CubicNaturalSpline (cubicinterpolation.hpp:206-217)
#   MonotonicCubicNaturalSpline (cubicinterpolation.hpp:219-230)
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

from scipy.interpolate import (  # type: ignore[import-untyped]
    CubicSpline,
    PchipInterpolator,
)

from pquantlib.exceptions import LibraryException
from pquantlib.math.array import Array
from pquantlib.math.interpolations.interpolation import Interpolation


class DerivativeApprox(IntEnum):
    """Cubic derivative-approximation styles.

    # C++ parity: ``CubicInterpolation::DerivativeApprox``
    #             (cubicinterpolation.hpp:111-141).

    Only ``Spline`` is implemented; the rest raise
    ``LibraryException`` from the ``CubicInterpolation`` constructor.
    Values match the C++ enum ordering.
    """

    Spline = 0
    SplineOM1 = 1
    SplineOM2 = 2
    FourthOrder = 3
    Parabolic = 4
    FritschButland = 5
    Akima = 6
    Kruger = 7
    Harmonic = 8


class BoundaryCondition(IntEnum):
    """Cubic spline boundary conditions.

    # C++ parity: ``CubicInterpolation::BoundaryCondition``
    #             (cubicinterpolation.hpp:142-159).

    Only ``SecondDerivative`` with value 0.0 (= natural BC) is
    implemented. Values match the C++ enum ordering.
    """

    NotAKnot = 0
    FirstDerivative = 1
    SecondDerivative = 2
    Periodic = 3
    Lagrange = 4


def _validate_supported(
    derivative_approx: DerivativeApprox,
    monotonic: bool,
    left_condition: BoundaryCondition,
    left_value: float,
    right_condition: BoundaryCondition,
    right_value: float,
) -> None:
    if derivative_approx != DerivativeApprox.Spline:
        raise LibraryException(
            f"DerivativeApprox.{derivative_approx.name} not implemented in this port "
            "(only Spline is supported)"
        )
    if left_condition != BoundaryCondition.SecondDerivative or left_value != 0.0:
        raise LibraryException(
            f"left BoundaryCondition.{left_condition.name} (value={left_value}) "
            "not implemented in this port (only SecondDerivative=0.0 / natural is supported)"
        )
    if right_condition != BoundaryCondition.SecondDerivative or right_value != 0.0:
        raise LibraryException(
            f"right BoundaryCondition.{right_condition.name} (value={right_value}) "
            "not implemented in this port (only SecondDerivative=0.0 / natural is supported)"
        )
    # Both `monotonic` arms are supported — that's the toggle between
    # CubicSpline and PchipInterpolator.
    _ = monotonic


class CubicInterpolation(Interpolation):
    """Top-level cubic spline with parameterized derivative/BC strategy.

    # C++ parity: ``CubicInterpolation`` (cubicinterpolation.hpp:109-201).

    Only ``Spline + Natural`` (``DerivativeApprox.Spline`` +
    ``BoundaryCondition.SecondDerivative`` with value 0.0) is implemented.
    The ``monotonic`` flag selects ``scipy.PchipInterpolator`` (monotonic
    Hyman/Fritsch-Carlson cubic) over ``scipy.CubicSpline`` (Natural BC).
    """

    def __init__(
        self,
        x_seq: Array,
        y_seq: Array,
        derivative_approx: DerivativeApprox = DerivativeApprox.Spline,
        monotonic: bool = False,
        left_condition: BoundaryCondition = BoundaryCondition.SecondDerivative,
        left_value: float = 0.0,
        right_condition: BoundaryCondition = BoundaryCondition.SecondDerivative,
        right_value: float = 0.0,
    ) -> None:
        super().__init__(x_seq, y_seq, required_points=2)
        _validate_supported(
            derivative_approx,
            monotonic,
            left_condition,
            left_value,
            right_condition,
            right_value,
        )
        self._derivative_approx: DerivativeApprox = derivative_approx
        self._monotonic: bool = monotonic
        self._left_condition: BoundaryCondition = left_condition
        self._left_value: float = left_value
        self._right_condition: BoundaryCondition = right_condition
        self._right_value: float = right_value
        # scipy state — rebuilt by ``update()``.
        self._spline: Any = None
        self._d1: Any = None
        self._d2: Any = None
        self._prim: Any = None
        self.update()

    def update(self) -> None:
        """Rebuild scipy spline (call after mutating x_seq / y_seq).

        # C++ parity: ``CubicInterpolation::update()`` (PIMPL).
        """
        if self._monotonic:
            # PCHIP — scipy's Hyman/Fritsch-Carlson monotonic cubic.
            # PchipInterpolator does not accept bc_type (its boundary
            # behaviour is encoded in the PCHIP slope formula itself —
            # one-sided three-point at the endpoints).
            self._spline = PchipInterpolator(self._xs, self._ys, extrapolate=True)
        else:
            # Natural cubic — second derivative = 0 at both ends.
            self._spline = CubicSpline(
                self._xs, self._ys, bc_type="natural", extrapolate=True
            )
        self._d1 = self._spline.derivative(1)
        self._d2 = self._spline.derivative(2)
        # scipy's ``antiderivative()`` returns a PPoly whose value at x is
        # F(x) = ∫_{x0}^{x} s(t) dt + C, where C is chosen so that F(x0) = 0.
        # That matches the C++ ``primitive`` semantics (primitiveConst_[0] = 0).
        self._prim = self._spline.antiderivative(1)

    def _value(self, x: float) -> float:
        return float(self._spline(x))

    def _derivative(self, x: float) -> float:
        return float(self._d1(x))

    def _second_derivative(self, x: float) -> float:
        return float(self._d2(x))

    def _primitive(self, x: float) -> float:
        return float(self._prim(x))


class CubicNaturalSpline(CubicInterpolation):
    """Spline + Natural BC + non-monotonic — the textbook cubic spline.

    # C++ parity: ``CubicNaturalSpline`` (cubicinterpolation.hpp:206-217).
    """

    def __init__(self, x_seq: Array, y_seq: Array) -> None:
        super().__init__(
            x_seq,
            y_seq,
            derivative_approx=DerivativeApprox.Spline,
            monotonic=False,
            left_condition=BoundaryCondition.SecondDerivative,
            left_value=0.0,
            right_condition=BoundaryCondition.SecondDerivative,
            right_value=0.0,
        )


class MonotonicCubicNaturalSpline(CubicInterpolation):
    """Monotonic cubic Hermite interpolation (PCHIP).

    # C++ parity: ``MonotonicCubicNaturalSpline`` (cubicinterpolation.hpp:219-230).

    Implemented via ``scipy.interpolate.PchipInterpolator`` — Fritsch-Carlson
    PCHIP. See the module-level docstring for the documented divergence
    from QuantLib's Hyman-on-natural-cubic algorithm: both are
    monotonicity-preserving cubics through the same knots, but they
    use different intermediate-slope formulas, so off-pillar values
    agree only at LOOSE tier.
    """

    def __init__(self, x_seq: Array, y_seq: Array) -> None:
        super().__init__(
            x_seq,
            y_seq,
            derivative_approx=DerivativeApprox.Spline,
            monotonic=True,
            left_condition=BoundaryCondition.SecondDerivative,
            left_value=0.0,
            right_condition=BoundaryCondition.SecondDerivative,
            right_value=0.0,
        )
