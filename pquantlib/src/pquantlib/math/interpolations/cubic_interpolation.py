"""Cubic interpolation between discrete points.

# C++ parity: ql/math/interpolations/cubicinterpolation.hpp (v1.43).

A full transcription of ``CubicInterpolation`` — all nine
``DerivativeApprox`` schemes crossed with all five ``BoundaryCondition``
choices, plus the orthogonal Hyman-1983 monotonicity filter — together
with the ``Cubic`` factory/traits object and every convenience preset.

Structure mirrors the C++ layering:

* :class:`CubicInterpolationBaseImpl` holds the per-interval coefficient
  storage (``a``, ``b``, ``c``), the primitive constants, the
  monotonicity-adjustment flags and the two boundary-condition values.
  In C++ this exists so ``MixedLinearCubicInterpolation`` can reach
  through the type-erased ``Interpolation::Impl`` handle and overwrite
  ``leftValue_``; here it plays the same role for
  :mod:`pquantlib.math.interpolations.mixed_interpolation`.
* :class:`CubicInterpolation` implements ``update()`` and the four
  evaluators on top of it.

The interpolating cubic on ``[x_i, x_{i+1}]`` is

.. math::

   P_i(x) = y_i + a_i (x-x_i) + b_i (x-x_i)^2 + c_i (x-x_i)^3

with the pillar first derivatives produced by the chosen
``DerivativeApprox``. Non-local (``Spline``, ``SplineOM1``, ``SplineOM2``)
schemes solve a linear system for them; local schemes (``Parabolic``,
``FritschButland``, ``Akima``, ``Kruger``, ``Harmonic``) use a stencil.

Reference for the monotonicity filter:

    R. L. Dougherty, A. Edelman, J. M. Hyman. *Nonnegativity-,
    Monotonicity-, or Convexity-Preserving Cubic and Quintic Hermite
    Interpolation.* Mathematics of Computation, 52(186):471-494, 1989.

Documented divergences from the C++ source:

* ``NotAKnot`` at either end reads ``dx_[1]`` / ``dx_[n-3]``, which is
  out of bounds for fewer than three points; C++ reads past the end of a
  ``std::vector`` (undefined behaviour). This port raises instead —
  see :func:`_check_not_a_knot_size`.
* C++ stores iterators into caller-owned arrays; per the
  :class:`~pquantlib.math.interpolations.interpolation.Interpolation`
  divergence note, this port copies.
* ``Cubic.global`` is spelled ``Cubic.global_`` — ``global`` is a Python
  keyword. Same convention as
  :class:`pquantlib.experimental.shortrate.linear_flat_interpolation.LinearFlat`.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import final

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.closeness import close
from pquantlib.math.interpolations.interpolation import Interpolation
from pquantlib.math.matrix import Matrix

# C++ ``QL_MAX_REAL`` / ``QL_MIN_REAL`` (ql/qldefines.hpp): the largest finite
# double and its negation. The FritschButland arm assigns them verbatim when
# ``Smax + 2*Smin`` cancels exactly, so the port needs the same constants and
# not ``inf``.
_QL_MAX_REAL: float = 1.7976931348623157e308
_QL_MIN_REAL: float = -_QL_MAX_REAL


class DerivativeApprox(IntEnum):
    """Pillar first-derivative approximation schemes.

    # C++ parity: ``CubicInterpolation::DerivativeApprox``
    #             (cubicinterpolation.hpp:111-141). Values match the C++
    #             enum ordering.
    """

    Spline = 0
    """Spline approximation (non-local, non-monotonic)."""
    SplineOM1 = 1
    """Overshooting minimization, 1st derivative."""
    SplineOM2 = 2
    """Overshooting minimization, 2nd derivative."""
    FourthOrder = 3
    """Fourth-order approximation — ``QL_FAIL`` in C++, unimplemented."""
    Parabolic = 4
    """Parabolic approximation (local, non-monotonic, linear)."""
    FritschButland = 5
    """Fritsch-Butland approximation (local, monotonic, non-linear)."""
    Akima = 6
    """Akima approximation (local, non-monotonic, non-linear)."""
    Kruger = 7
    """Kruger approximation (local, monotonic, non-linear)."""
    Harmonic = 8
    """Weighted harmonic mean (local, monotonic, non-linear)."""


class BoundaryCondition(IntEnum):
    """End conditions for the spline schemes.

    # C++ parity: ``CubicInterpolation::BoundaryCondition``
    #             (cubicinterpolation.hpp:142-159). Values match the C++
    #             enum ordering.
    """

    NotAKnot = 0
    """Make the second(-last) point an inactive knot."""
    FirstDerivative = 1
    """Match the value of the end slope."""
    SecondDerivative = 2
    """Match the second derivative at the end (0.0 = natural spline)."""
    Periodic = 3
    """Match first and second derivative at either end — ``QL_FAIL`` in C++."""
    Lagrange = 4
    """Match the end slope to the cubic through the four nearest data."""


def _solve_tridiagonal(
    lower: list[float],
    diagonal: list[float],
    upper: list[float],
    rhs: list[float],
) -> list[float]:
    """Solve a tridiagonal system by the C++ elimination, step for step.

    # C++ parity: ``TridiagonalOperator::solveFor``
    #             (ql/methods/finitedifferences/tridiagonaloperator.cpp:86-110).

    ``lower`` and ``upper`` have ``n-1`` entries, ``diagonal`` and ``rhs``
    have ``n``. Row ``j`` of the matrix is
    ``lower[j-1], diagonal[j], upper[j]``.

    The two zero-pivot guards are C++'s own (``QL_REQUIRE`` on the first
    diagonal element, ``QL_ENSURE`` inside the sweep); they are not
    defensive additions, and they fire for genuinely singular set-ups —
    e.g. ``NotAKnot`` at both ends with exactly three points.
    """
    n = len(diagonal)
    result = [0.0] * n
    temp = [0.0] * n
    bet = diagonal[0]
    qassert.require(
        not close(bet, 0.0),
        f"diagonal's first element ({bet}) cannot be close to zero",
    )
    result[0] = rhs[0] / bet
    for j in range(1, n):
        temp[j] = upper[j - 1] / bet
        bet = diagonal[j] - lower[j - 1] * temp[j]
        qassert.require(not close(bet, 0.0), "division by zero")
        result[j] = (rhs[j] - lower[j - 1] * result[j - 1]) / bet
    for j in range(n - 2, 0, -1):
        result[j] -= temp[j + 1] * result[j + 1]
    result[0] -= temp[1] * result[1]
    return result


def _cubic_interpolating_polynomial_derivative(
    a: float,
    b: float,
    c: float,
    d: float,
    u: float,
    v: float,
    w: float,
    z: float,
    x: float,
) -> float:
    """Derivative at ``x`` of the cubic through ``(a,u) (b,v) (c,w) (d,z)``.

    # C++ parity: ``CubicInterpolationImpl::cubicInterpolatingPolynomialDerivative``
    #             (cubicinterpolation.hpp:812-821). Transcribed verbatim,
    #             including the grouping — the expression is a difference of
    #             large nearly-equal products and re-associating it moves the
    #             result.
    """
    return (
        -(
            (
                (
                    (a - c) * (b - c) * (c - x) * z - (a - d) * (b - d) * (d - x) * w
                )
                * (a - x + b - x)
                + ((a - c) * (b - c) * z - (a - d) * (b - d) * w) * (a - x) * (b - x)
            )
            * (a - b)
            + ((a - c) * (a - d) * v - (b - c) * (b - d) * u) * (c - d) * (c - x) * (d - x)
            + ((a - c) * (a - d) * (a - x) * v - (b - c) * (b - d) * (b - x) * u)
            * (c - x + d - x)
            * (c - d)
        )
    ) / ((a - b) * (a - c) * (a - d) * (b - c) * (b - d) * (c - d))


class CubicInterpolationBaseImpl(Interpolation):
    """Coefficient storage shared by every cubic interpolation.

    # C++ parity: ``detail::CubicInterpolationBaseImpl``
    #             (cubicinterpolation.hpp:42-58).

    Exists for the same reason as in C++: it is the slice of state that a
    *different* interpolation needs to reach into. ``MixedLinearCubic``
    with ``FirstDerivative`` + a null condition value overwrites
    ``left_value`` between the two sub-updates, and the only thing it can
    see through the handle is this base.
    """

    def __init__(
        self,
        x_seq: Array,
        y_seq: Array,
        left_condition_value: float,
        right_condition_value: float,
    ) -> None:
        # ``Cubic::requiredPoints`` == 2 (cubicinterpolation.hpp:365).
        super().__init__(x_seq, y_seq, required_points=2)
        n = int(self._xs.shape[0])
        self._primitive_const: list[float] = [0.0] * (n - 1)
        self._a: list[float] = [0.0] * (n - 1)
        self._b: list[float] = [0.0] * (n - 1)
        self._c: list[float] = [0.0] * (n - 1)
        self._monotonicity_adjustments: list[bool] = [False] * n
        self._left_value: float = left_condition_value
        self._right_value: float = right_condition_value

    # ----- inspectors (C++ CubicInterpolation accessors) ------------------

    def primitive_constants(self) -> list[float]:
        """# C++ parity: ``CubicInterpolation::primitiveConstants()``."""
        return list(self._primitive_const)

    def a_coefficients(self) -> list[float]:
        """# C++ parity: ``CubicInterpolation::aCoefficients()``."""
        return list(self._a)

    def b_coefficients(self) -> list[float]:
        """# C++ parity: ``CubicInterpolation::bCoefficients()``."""
        return list(self._b)

    def c_coefficients(self) -> list[float]:
        """# C++ parity: ``CubicInterpolation::cCoefficients()``."""
        return list(self._c)

    def monotonicity_adjustments(self) -> list[bool]:
        """# C++ parity: ``CubicInterpolation::monotonicityAdjustments()``."""
        return list(self._monotonicity_adjustments)

    @property
    def left_value(self) -> float:
        """The left end-condition value (C++ ``leftValue_``)."""
        return self._left_value

    @property
    def right_value(self) -> float:
        """The right end-condition value (C++ ``rightValue_``)."""
        return self._right_value

    def update_left_condition_value(self, value: float) -> None:
        """# C++ parity: ``CubicInterpolation::updateLeftConditionValue``."""
        self._left_value = value

    def update_right_condition_value(self, value: float) -> None:
        """# C++ parity: ``CubicInterpolation::updateRightConditionValue``."""
        self._right_value = value


def _check_not_a_knot_size(n: int) -> None:
    """Guard the out-of-range read C++ performs for tiny ``NotAKnot`` inputs.

    C++ indexes ``dx_[1]`` (left arm) and ``dx_[n-3]`` (right arm) of a
    vector of length ``n-1``. For ``n == 2`` both are out of bounds and C++
    reads uninitialised memory. Python's negative indexing would silently
    wrap instead, which is worse, so the port refuses.
    """
    qassert.require(
        n >= 3,
        f"NotAKnot boundary condition requires at least 3 points ({n} are given)",
    )


class CubicInterpolation(CubicInterpolationBaseImpl):
    """Cubic interpolation with a selectable derivative scheme and end conditions.

    # C++ parity: ``CubicInterpolation`` (cubicinterpolation.hpp:109-203)
    #             + ``detail::CubicInterpolationImpl`` (376-822).

    Args:
        x_seq: pillar abscissae, sorted ascending.
        y_seq: pillar ordinates.
        derivative_approx: which pillar-slope scheme to use.
        monotonic: apply the Hyman-1983 filter to the slopes afterwards.
        left_condition: end condition at ``x[0]``.
        left_value: value that goes with ``left_condition``.
        right_condition: end condition at ``x[-1]``.
        right_value: value that goes with ``right_condition``.
        update: run :meth:`update` at construction (C++ ctor flag).
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
        update: bool = True,
    ) -> None:
        super().__init__(x_seq, y_seq, left_value, right_value)
        self._n: int = int(self._xs.shape[0])
        self._da: DerivativeApprox = derivative_approx
        self._monotonic: bool = monotonic
        self._left_type: BoundaryCondition = left_condition
        self._right_type: BoundaryCondition = right_condition
        # C++ parity: cubicinterpolation.hpp:397-407 — both checks live in the
        # constructor, after the base-class point-count check.
        if BoundaryCondition.Lagrange in (self._left_type, self._right_type):
            qassert.require(
                self._n >= 4,
                "Lagrange boundary condition requires at least "
                f"4 points ({self._n} are given)",
            )
        if self._da == DerivativeApprox.Akima:
            qassert.require(
                self._n >= 4,
                f"Akima approximation requires at least 4 points ({self._n} are given)",
            )
        if BoundaryCondition.NotAKnot in (self._left_type, self._right_type):
            _check_not_a_knot_size(self._n)
        if update:
            self.update()

    # ------------------------------------------------------------------
    # update — port of ``CubicInterpolationImpl::update``
    # (cubicinterpolation.hpp:410-779)
    # ------------------------------------------------------------------

    def update(self) -> None:
        """Recompute the pillar slopes and the per-interval cubic coefficients."""
        n = self._n
        xs = self._xs
        ys = self._ys
        dx: list[float] = [0.0] * (n - 1)
        s: list[float] = [0.0] * (n - 1)
        for i in range(n - 1):
            dx[i] = float(xs[i + 1]) - float(xs[i])
            s[i] = (float(ys[i + 1]) - float(ys[i])) / dx[i]

        if self._da == DerivativeApprox.Spline:
            tmp = self._spline_derivatives(dx, s)
        elif self._da in (DerivativeApprox.SplineOM1, DerivativeApprox.SplineOM2):
            tmp = self._overshooting_minimization_derivatives(dx)
        else:
            tmp = self._local_derivatives(dx, s)

        self._monotonicity_adjustments = [False] * n
        if self._monotonic:
            self._apply_hyman_filter(tmp, dx, s)

        # cubic coefficients (cubicinterpolation.hpp:764-768)
        for i in range(n - 1):
            self._a[i] = tmp[i]
            self._b[i] = (3.0 * s[i] - tmp[i + 1] - 2.0 * tmp[i]) / dx[i]
            self._c[i] = (tmp[i + 1] + tmp[i] - 2.0 * s[i]) / (dx[i] * dx[i])

        # primitive constants (cubicinterpolation.hpp:770-778)
        self._primitive_const[0] = 0.0
        for i in range(1, n - 1):
            self._primitive_const[i] = self._primitive_const[i - 1] + dx[i - 1] * (
                float(ys[i - 1])
                + dx[i - 1]
                * (
                    self._a[i - 1] / 2.0
                    + dx[i - 1] * (self._b[i - 1] / 3.0 + dx[i - 1] * self._c[i - 1] / 4.0)
                )
            )

    def _spline_derivatives(  # noqa: PLR0915 — one-for-one with the C++ body
        self, dx: list[float], s: list[float]
    ) -> list[float]:
        """Solve the tridiagonal system for the pillar slopes.

        # C++ parity: cubicinterpolation.hpp:418-489 (the ``da == Spline`` arm).
        """
        n = self._n
        xs = self._xs
        ys = self._ys
        lower: list[float] = [0.0] * (n - 1)
        diag: list[float] = [0.0] * n
        upper: list[float] = [0.0] * (n - 1)
        rhs: list[float] = [0.0] * n

        # setMidRow(i, dx[i], 2*(dx[i]+dx[i-1]), dx[i-1]) writes
        # lowerDiagonal_[i-1], diagonal_[i], upperDiagonal_[i].
        for i in range(1, n - 1):
            lower[i - 1] = dx[i]
            diag[i] = 2.0 * (dx[i] + dx[i - 1])
            upper[i] = dx[i - 1]
            rhs[i] = 3.0 * (dx[i] * s[i - 1] + dx[i - 1] * s[i])

        # left boundary condition
        if self._left_type == BoundaryCondition.NotAKnot:
            # C++ ignores the end-condition value for this arm.
            diag[0] = dx[1] * (dx[1] + dx[0])
            upper[0] = (dx[0] + dx[1]) * (dx[0] + dx[1])
            rhs[0] = s[0] * dx[1] * (2.0 * dx[1] + 3.0 * dx[0]) + s[1] * dx[0] * dx[0]
        elif self._left_type == BoundaryCondition.FirstDerivative:
            diag[0] = 1.0
            upper[0] = 0.0
            rhs[0] = self._left_value
        elif self._left_type == BoundaryCondition.SecondDerivative:
            diag[0] = 2.0
            upper[0] = 1.0
            rhs[0] = 3.0 * s[0] - self._left_value * dx[0] / 2.0
        elif self._left_type == BoundaryCondition.Periodic:
            qassert.fail("this end condition is not implemented yet")
        elif self._left_type == BoundaryCondition.Lagrange:
            diag[0] = 1.0
            upper[0] = 0.0
            rhs[0] = _cubic_interpolating_polynomial_derivative(
                float(xs[0]),
                float(xs[1]),
                float(xs[2]),
                float(xs[3]),
                float(ys[0]),
                float(ys[1]),
                float(ys[2]),
                float(ys[3]),
                float(xs[0]),
            )
        else:
            qassert.fail("unknown end condition")

        # right boundary condition
        if self._right_type == BoundaryCondition.NotAKnot:
            lower[n - 2] = -(dx[n - 2] + dx[n - 3]) * (dx[n - 2] + dx[n - 3])
            diag[n - 1] = -dx[n - 3] * (dx[n - 3] + dx[n - 2])
            rhs[n - 1] = -s[n - 3] * dx[n - 2] * dx[n - 2] - s[n - 2] * dx[n - 3] * (
                3.0 * dx[n - 2] + 2.0 * dx[n - 3]
            )
        elif self._right_type == BoundaryCondition.FirstDerivative:
            lower[n - 2] = 0.0
            diag[n - 1] = 1.0
            rhs[n - 1] = self._right_value
        elif self._right_type == BoundaryCondition.SecondDerivative:
            lower[n - 2] = 1.0
            diag[n - 1] = 2.0
            rhs[n - 1] = 3.0 * s[n - 2] + self._right_value * dx[n - 2] / 2.0
        elif self._right_type == BoundaryCondition.Periodic:
            qassert.fail("this end condition is not implemented yet")
        elif self._right_type == BoundaryCondition.Lagrange:
            lower[n - 2] = 0.0
            diag[n - 1] = 1.0
            rhs[n - 1] = _cubic_interpolating_polynomial_derivative(
                float(xs[n - 4]),
                float(xs[n - 3]),
                float(xs[n - 2]),
                float(xs[n - 1]),
                float(ys[n - 4]),
                float(ys[n - 3]),
                float(ys[n - 2]),
                float(ys[n - 1]),
                float(xs[n - 1]),
            )
        else:
            qassert.fail("unknown end condition")

        return _solve_tridiagonal(lower, diag, upper, rhs)

    def _overshooting_minimization_derivatives(self, dx: list[float]) -> list[float]:
        """Least-overshoot spline slopes (``SplineOM1`` / ``SplineOM2``).

        # C++ parity: cubicinterpolation.hpp:490-576. The two arms differ
        # only in the weighting matrix ``Q``: OM1 weights by ``dx^3``
        # (minimising the first derivative's jumps), OM2 by ``dx``.
        """
        n = self._n
        ys = self._ys
        t: Matrix = np.zeros((n - 2, n), dtype=np.float64)
        s_mat: Matrix = np.zeros((n - 2, n), dtype=np.float64)
        for i in range(n - 2):
            t[i, i] = dx[i] / 6.0
            t[i, i + 1] = (dx[i + 1] + dx[i]) / 3.0
            t[i, i + 2] = dx[i + 1] / 6.0
            s_mat[i, i] = 1.0 / dx[i]
            s_mat[i, i + 1] = -(1.0 / dx[i + 1] + 1.0 / dx[i])
            s_mat[i, i + 2] = 1.0 / dx[i + 1]
        up: Matrix = np.zeros((n, 2), dtype=np.float64)
        up[0, 0] = 1.0
        up[n - 1, 1] = 1.0
        us: Matrix = np.zeros((n, n - 2), dtype=np.float64)
        for i in range(n - 2):
            us[i + 1, i] = 1.0
        z: Matrix = us @ np.linalg.inv(t @ us)
        identity: Matrix = np.eye(n, dtype=np.float64)
        v: Matrix = (identity - z @ t) @ up
        w: Matrix = z @ s_mat
        q: Matrix = np.zeros((n, n), dtype=np.float64)
        if self._da == DerivativeApprox.SplineOM1:
            q[0, 0] = 1.0 / (n - 1) * dx[0] * dx[0] * dx[0]
            q[0, 1] = 7.0 / 8 * 1.0 / (n - 1) * dx[0] * dx[0] * dx[0]
            for i in range(1, n - 1):
                q[i, i - 1] = 7.0 / 8 * 1.0 / (n - 1) * dx[i - 1] * dx[i - 1] * dx[i - 1]
                q[i, i] = 1.0 / (n - 1) * dx[i] * dx[i] * dx[i] + 1.0 / (n - 1) * dx[
                    i - 1
                ] * dx[i - 1] * dx[i - 1]
                q[i, i + 1] = 7.0 / 8 * 1.0 / (n - 1) * dx[i] * dx[i] * dx[i]
            q[n - 1, n - 2] = 7.0 / 8 * 1.0 / (n - 1) * dx[n - 2] * dx[n - 2] * dx[n - 2]
            q[n - 1, n - 1] = 1.0 / (n - 1) * dx[n - 2] * dx[n - 2] * dx[n - 2]
        else:
            q[0, 0] = 1.0 / (n - 1) * dx[0]
            q[0, 1] = 1.0 / 2 * 1.0 / (n - 1) * dx[0]
            for i in range(1, n - 1):
                q[i, i - 1] = 1.0 / 2 * 1.0 / (n - 1) * dx[i - 1]
                q[i, i] = 1.0 / (n - 1) * dx[i] + 1.0 / (n - 1) * dx[i - 1]
                q[i, i + 1] = 1.0 / 2 * 1.0 / (n - 1) * dx[i]
            q[n - 1, n - 2] = 1.0 / 2 * 1.0 / (n - 1) * dx[n - 2]
            q[n - 1, n - 1] = 1.0 / (n - 1) * dx[n - 2]
        j: Matrix = (
            identity - v @ np.linalg.inv(v.T @ q @ v) @ v.T @ q
        ) @ w
        y_vec: Array = np.asarray(ys, dtype=np.float64)
        d_vec: Array = j @ y_vec
        tmp: list[float] = [0.0] * n
        for i in range(n - 1):
            tmp[i] = (float(y_vec[i + 1]) - float(y_vec[i])) / dx[i] - (
                2.0 * float(d_vec[i]) + float(d_vec[i + 1])
            ) * dx[i] / 6.0
        tmp[n - 1] = (
            tmp[n - 2]
            + float(d_vec[n - 2]) * dx[n - 2]
            + (float(d_vec[n - 1]) - float(d_vec[n - 2])) * dx[n - 2] / 2.0
        )
        return tmp

    def _local_derivatives(  # noqa: PLR0915 — one-for-one with the C++ body
        self, dx: list[float], s: list[float]
    ) -> list[float]:
        """Stencil-based pillar slopes.

        # C++ parity: cubicinterpolation.hpp:577-685 (the ``else`` arm).
        """
        n = self._n
        tmp: list[float] = [0.0] * n
        if n == 2:
            tmp[0] = tmp[1] = s[0]
            return tmp

        if self._da == DerivativeApprox.FourthOrder:
            qassert.fail("FourthOrder not implemented yet")
        elif self._da == DerivativeApprox.Parabolic:
            for i in range(1, n - 1):
                tmp[i] = (dx[i - 1] * s[i] + dx[i] * s[i - 1]) / (dx[i] + dx[i - 1])
            tmp[0] = ((2.0 * dx[0] + dx[1]) * s[0] - dx[0] * s[1]) / (dx[0] + dx[1])
            tmp[n - 1] = ((2.0 * dx[n - 2] + dx[n - 3]) * s[n - 2] - dx[n - 2] * s[n - 3]) / (
                dx[n - 2] + dx[n - 3]
            )
        elif self._da == DerivativeApprox.FritschButland:
            for i in range(1, n - 1):
                s_min = min(s[i - 1], s[i])
                s_max = max(s[i - 1], s[i])
                if s_max + 2.0 * s_min == 0:
                    # C++ hands back QL_MIN_REAL / QL_MAX_REAL here; the cubic
                    # coefficients then overflow to +-inf. Reproduced, not
                    # "fixed" — it is what the reference implementation does.
                    if s_min * s_max < 0:
                        tmp[i] = _QL_MIN_REAL
                    elif s_min * s_max == 0:
                        tmp[i] = 0.0
                    else:
                        tmp[i] = _QL_MAX_REAL
                else:
                    tmp[i] = 3.0 * s_min * s_max / (s_max + 2.0 * s_min)
            tmp[0] = ((2.0 * dx[0] + dx[1]) * s[0] - dx[0] * s[1]) / (dx[0] + dx[1])
            tmp[n - 1] = ((2.0 * dx[n - 2] + dx[n - 3]) * s[n - 2] - dx[n - 2] * s[n - 3]) / (
                dx[n - 2] + dx[n - 3]
            )
        elif self._da == DerivativeApprox.Akima:
            tmp[0] = (
                abs(s[1] - s[0]) * 2 * s[0] * s[1]
                + abs(2 * s[0] * s[1] - 4 * s[0] * s[0] * s[1]) * s[0]
            ) / (abs(s[1] - s[0]) + abs(2 * s[0] * s[1] - 4 * s[0] * s[0] * s[1]))
            tmp[1] = (
                abs(s[2] - s[1]) * s[0] + abs(s[0] - 2 * s[0] * s[1]) * s[1]
            ) / (abs(s[2] - s[1]) + abs(s[0] - 2 * s[0] * s[1]))
            for i in range(2, n - 2):
                if s[i - 2] == s[i - 1] and s[i] != s[i + 1]:
                    tmp[i] = s[i - 1]
                # C++ keeps these two tests separate
                # (cubicinterpolation.hpp:619-622); merging them would reorder
                # the evaluation of the equality tests.
                elif s[i - 2] != s[i - 1] and s[i] == s[i + 1]:  # noqa: SIM114
                    tmp[i] = s[i]
                elif s[i] == s[i - 1]:
                    tmp[i] = s[i]
                elif s[i - 2] == s[i - 1] and s[i - 1] != s[i] and s[i] == s[i + 1]:
                    # Unreachable given the two branches above — kept because
                    # C++ has it and a port that drops "dead" arms is a port
                    # that has started paraphrasing.
                    tmp[i] = (s[i - 1] + s[i]) / 2.0
                else:
                    tmp[i] = (
                        abs(s[i + 1] - s[i]) * s[i - 1] + abs(s[i - 1] - s[i - 2]) * s[i]
                    ) / (abs(s[i + 1] - s[i]) + abs(s[i - 1] - s[i - 2]))
            tmp[n - 2] = (
                abs(2 * s[n - 2] * s[n - 3] - s[n - 2]) * s[n - 3]
                + abs(s[n - 3] - s[n - 4]) * s[n - 2]
            ) / (abs(2 * s[n - 2] * s[n - 3] - s[n - 2]) + abs(s[n - 3] - s[n - 4]))
            tmp[n - 1] = (
                abs(4 * s[n - 2] * s[n - 2] * s[n - 3] - 2 * s[n - 2] * s[n - 3]) * s[n - 2]
                + abs(s[n - 2] - s[n - 3]) * 2 * s[n - 2] * s[n - 3]
            ) / (
                abs(4 * s[n - 2] * s[n - 2] * s[n - 3] - 2 * s[n - 2] * s[n - 3])
                + abs(s[n - 2] - s[n - 3])
            )
        elif self._da == DerivativeApprox.Kruger:
            for i in range(1, n - 1):
                if s[i - 1] * s[i] < 0.0:
                    tmp[i] = 0.0
                else:
                    tmp[i] = 2.0 / (1.0 / s[i - 1] + 1.0 / s[i])
            tmp[0] = (3.0 * s[0] - tmp[1]) / 2.0
            tmp[n - 1] = (3.0 * s[n - 2] - tmp[n - 2]) / 2.0
        elif self._da == DerivativeApprox.Harmonic:
            for i in range(1, n - 1):
                w1 = 2 * dx[i] + dx[i - 1]
                w2 = dx[i] + 2 * dx[i - 1]
                if s[i - 1] * s[i] <= 0.0:
                    tmp[i] = 0.0
                else:
                    tmp[i] = (w1 + w2) / (w1 / s[i - 1] + w2 / s[i])
            tmp[0] = ((2 * dx[0] + dx[1]) * s[0] - dx[0] * s[1]) / (dx[1] + dx[0])
            if tmp[0] * s[0] < 0.0:
                tmp[0] = 0.0
            elif s[0] * s[1] < 0 and math.fabs(tmp[0]) > math.fabs(3 * s[0]):
                tmp[0] = 3 * s[0]
            tmp[n - 1] = ((2 * dx[n - 2] + dx[n - 3]) * s[n - 2] - dx[n - 2] * s[n - 3]) / (
                dx[n - 3] + dx[n - 2]
            )
            if tmp[n - 1] * s[n - 2] < 0.0:
                tmp[n - 1] = 0.0
            elif s[n - 2] * s[n - 3] < 0 and math.fabs(tmp[n - 1]) > math.fabs(
                3 * s[n - 2]
            ):
                tmp[n - 1] = 3 * s[n - 2]
        else:
            qassert.fail("unknown scheme")
        return tmp

    def _apply_hyman_filter(
        self, tmp: list[float], dx: list[float], s: list[float]
    ) -> None:
        """Clip the pillar slopes to preserve local monotonicity, in place.

        # C++ parity: cubicinterpolation.hpp:690-760 (the ``if (monotonic_)``
        # block). Sets ``monotonicity_adjustments[i]`` wherever a slope moved.
        """
        n = self._n
        adjustments = self._monotonicity_adjustments
        for i in range(n):
            if i == 0:
                if tmp[i] * s[0] > 0.0:
                    correction = (
                        tmp[i] / math.fabs(tmp[i]) * min(math.fabs(tmp[i]), math.fabs(3.0 * s[0]))
                    )
                else:
                    correction = 0.0
                if correction != tmp[i]:
                    tmp[i] = correction
                    adjustments[i] = True
            elif i == n - 1:
                if tmp[i] * s[n - 2] > 0.0:
                    correction = (
                        tmp[i]
                        / math.fabs(tmp[i])
                        * min(math.fabs(tmp[i]), math.fabs(3.0 * s[n - 2]))
                    )
                else:
                    correction = 0.0
                if correction != tmp[i]:
                    tmp[i] = correction
                    adjustments[i] = True
            else:
                pm = (s[i - 1] * dx[i] + s[i] * dx[i - 1]) / (dx[i - 1] + dx[i])
                m_bound = 3.0 * min(math.fabs(s[i - 1]), math.fabs(s[i]), math.fabs(pm))
                if i > 1 and (s[i - 1] - s[i - 2]) * (s[i] - s[i - 1]) > 0.0:
                    pd = (s[i - 1] * (2.0 * dx[i - 1] + dx[i - 2]) - s[i - 2] * dx[i - 1]) / (
                        dx[i - 2] + dx[i - 1]
                    )
                    if pm * pd > 0.0 and pm * (s[i - 1] - s[i - 2]) > 0.0:
                        m_bound = max(m_bound, 1.5 * min(math.fabs(pm), math.fabs(pd)))
                if i < n - 2 and (s[i] - s[i - 1]) * (s[i + 1] - s[i]) > 0.0:
                    pu = (s[i] * (2.0 * dx[i] + dx[i + 1]) - s[i + 1] * dx[i]) / (
                        dx[i] + dx[i + 1]
                    )
                    if pm * pu > 0.0 and -pm * (s[i] - s[i - 1]) > 0.0:
                        m_bound = max(m_bound, 1.5 * min(math.fabs(pm), math.fabs(pu)))
                if tmp[i] * pm > 0.0:
                    correction = tmp[i] / math.fabs(tmp[i]) * min(math.fabs(tmp[i]), m_bound)
                else:
                    correction = 0.0
                if correction != tmp[i]:
                    tmp[i] = correction
                    adjustments[i] = True

    # ----- evaluators (cubicinterpolation.hpp:780-801) --------------------

    def _value(self, x: float) -> float:
        j = self._locate(x)
        h = x - float(self._xs[j])
        return float(self._ys[j]) + h * (self._a[j] + h * (self._b[j] + h * self._c[j]))

    def _primitive(self, x: float) -> float:
        j = self._locate(x)
        h = x - float(self._xs[j])
        return self._primitive_const[j] + h * (
            float(self._ys[j])
            + h * (self._a[j] / 2.0 + h * (self._b[j] / 3.0 + h * self._c[j] / 4.0))
        )

    def _derivative(self, x: float) -> float:
        j = self._locate(x)
        h = x - float(self._xs[j])
        return self._a[j] + (2.0 * self._b[j] + 3.0 * self._c[j] * h) * h

    def _second_derivative(self, x: float) -> float:
        j = self._locate(x)
        h = x - float(self._xs[j])
        return 2.0 * self._b[j] + 6.0 * self._c[j] * h


# ---------------------------------------------------------------------------
# factory / traits
# ---------------------------------------------------------------------------


@final
class Cubic:
    """Cubic-interpolation factory and traits.

    # C++ parity: ``class Cubic`` (cubicinterpolation.hpp:340-371).

    ``global_`` spells C++'s ``static const bool global`` (``global`` is a
    Python keyword). The C++ defaults are Kruger, non-monotonic, natural at
    both ends — note that they are *not* the defaults of
    :class:`CubicInterpolation`, whose own C++ constructor has none.
    """

    global_ = True  # C++ ``static const bool global = true``.
    required_points = 2  # C++ ``static const Size requiredPoints = 2``.

    def __init__(
        self,
        derivative_approx: DerivativeApprox = DerivativeApprox.Kruger,
        monotonic: bool = False,
        left_condition: BoundaryCondition = BoundaryCondition.SecondDerivative,
        left_value: float = 0.0,
        right_condition: BoundaryCondition = BoundaryCondition.SecondDerivative,
        right_value: float = 0.0,
    ) -> None:
        self._da: DerivativeApprox = derivative_approx
        self._monotonic: bool = monotonic
        self._left_type: BoundaryCondition = left_condition
        self._right_type: BoundaryCondition = right_condition
        self._left_value: float = left_value
        self._right_value: float = right_value

    def interpolate(
        self, x_seq: Array, y_seq: Array, update: bool = True
    ) -> CubicInterpolation:
        """Build a :class:`CubicInterpolation` over ``(x, y)``."""
        return CubicInterpolation(
            x_seq,
            y_seq,
            self._da,
            self._monotonic,
            self._left_type,
            self._left_value,
            self._right_type,
            self._right_value,
            update,
        )


# ---------------------------------------------------------------------------
# convenience classes (cubicinterpolation.hpp:206-336)
#
# Each preset IS a specific (DerivativeApprox, monotonic, leftCondition,
# rightCondition) tuple — that tuple is the class's entire semantics, so the
# constructors below are transcribed one for one.
# ---------------------------------------------------------------------------


@final
class CubicNaturalSpline(CubicInterpolation):
    """Spline, natural BC at both ends, unfiltered.

    # C++ parity: ``CubicNaturalSpline`` (cubicinterpolation.hpp:208-219).
    """

    def __init__(self, x_seq: Array, y_seq: Array) -> None:
        super().__init__(
            x_seq,
            y_seq,
            DerivativeApprox.Spline,
            False,
            BoundaryCondition.SecondDerivative,
            0.0,
            BoundaryCondition.SecondDerivative,
            0.0,
        )


@final
class MonotonicCubicNaturalSpline(CubicInterpolation):
    """Natural cubic spline with the Hyman-1983 monotonicity filter.

    # C++ parity: ``MonotonicCubicNaturalSpline`` (cubicinterpolation.hpp:221-232).

    Not the Fritsch-Carlson PCHIP: that derives its slopes from a
    three-point stencil, whereas this solves the natural-BC tridiagonal
    system first and *then* filters. Same knots, different function
    between them.
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


@final
class CubicSplineOvershootingMinimization1(CubicInterpolation):
    """``SplineOM1`` with natural BCs.

    # C++ parity: ``CubicSplineOvershootingMinimization1``
    #             (cubicinterpolation.hpp:234-245).
    """

    def __init__(self, x_seq: Array, y_seq: Array) -> None:
        super().__init__(
            x_seq,
            y_seq,
            DerivativeApprox.SplineOM1,
            False,
            BoundaryCondition.SecondDerivative,
            0.0,
            BoundaryCondition.SecondDerivative,
            0.0,
        )


@final
class CubicSplineOvershootingMinimization2(CubicInterpolation):
    """``SplineOM2`` with natural BCs.

    # C++ parity: ``CubicSplineOvershootingMinimization2``
    #             (cubicinterpolation.hpp:247-258).
    """

    def __init__(self, x_seq: Array, y_seq: Array) -> None:
        super().__init__(
            x_seq,
            y_seq,
            DerivativeApprox.SplineOM2,
            False,
            BoundaryCondition.SecondDerivative,
            0.0,
            BoundaryCondition.SecondDerivative,
            0.0,
        )


@final
class KrugerCubic(CubicInterpolation):
    """Kruger slopes, natural BCs, unfiltered.

    # C++ parity: ``KrugerCubic`` (cubicinterpolation.hpp:273-284).
    """

    def __init__(self, x_seq: Array, y_seq: Array) -> None:
        super().__init__(
            x_seq,
            y_seq,
            DerivativeApprox.Kruger,
            False,
            BoundaryCondition.SecondDerivative,
            0.0,
            BoundaryCondition.SecondDerivative,
            0.0,
        )


@final
class HarmonicCubic(CubicInterpolation):
    """Weighted-harmonic-mean slopes, natural BCs.

    # C++ parity: ``HarmonicCubic`` (cubicinterpolation.hpp:286-297).
    """

    def __init__(self, x_seq: Array, y_seq: Array) -> None:
        super().__init__(
            x_seq,
            y_seq,
            DerivativeApprox.Harmonic,
            False,
            BoundaryCondition.SecondDerivative,
            0.0,
            BoundaryCondition.SecondDerivative,
            0.0,
        )


@final
class FritschButlandCubic(CubicInterpolation):
    """Fritsch-Butland slopes, natural BCs, Hyman-filtered.

    # C++ parity: ``FritschButlandCubic`` (cubicinterpolation.hpp:299-310).

    Note the ``monotonic=True``: unlike the other local-scheme presets,
    this one *is* filtered.
    """

    def __init__(self, x_seq: Array, y_seq: Array) -> None:
        super().__init__(
            x_seq,
            y_seq,
            DerivativeApprox.FritschButland,
            True,
            BoundaryCondition.SecondDerivative,
            0.0,
            BoundaryCondition.SecondDerivative,
            0.0,
        )


@final
class Parabolic(CubicInterpolation):
    """Parabolic slopes, natural BCs, unfiltered.

    # C++ parity: ``Parabolic`` (cubicinterpolation.hpp:312-323).
    """

    def __init__(self, x_seq: Array, y_seq: Array) -> None:
        super().__init__(
            x_seq,
            y_seq,
            DerivativeApprox.Parabolic,
            False,
            BoundaryCondition.SecondDerivative,
            0.0,
            BoundaryCondition.SecondDerivative,
            0.0,
        )


@final
class MonotonicParabolic(CubicInterpolation):
    """Parabolic slopes, natural BCs, Hyman-filtered.

    # C++ parity: ``MonotonicParabolic`` (cubicinterpolation.hpp:325-336).
    """

    def __init__(self, x_seq: Array, y_seq: Array) -> None:
        super().__init__(
            x_seq,
            y_seq,
            DerivativeApprox.Parabolic,
            True,
            BoundaryCondition.SecondDerivative,
            0.0,
            BoundaryCondition.SecondDerivative,
            0.0,
        )


__all__ = [
    "BoundaryCondition",
    "Cubic",
    "CubicInterpolation",
    "CubicInterpolationBaseImpl",
    "CubicNaturalSpline",
    "CubicSplineOvershootingMinimization1",
    "CubicSplineOvershootingMinimization2",
    "DerivativeApprox",
    "FritschButlandCubic",
    "HarmonicCubic",
    "KrugerCubic",
    "MonotonicCubicNaturalSpline",
    "MonotonicParabolic",
    "Parabolic",
]
