"""Hyman-1983 monotonicity-filtered natural cubic spline.

# C++ parity: ql/math/interpolations/cubicinterpolation.hpp
#             ``Spline + monotonic=true + SecondDerivative=0`` arm
#             of ``CubicInterpolation`` (cubicinterpolation.hpp:402-770)
#             (v1.42.1).

QuantLib's ``CubicInterpolation`` with ``DerivativeApprox::Spline``,
``BoundaryCondition::SecondDerivative=0`` (natural BC), and
``monotonic=true`` is *not* the Fritsch-Carlson PCHIP. Instead:

1. Build a natural cubic spline by solving the tridiagonal system for
   the first derivatives ``tmp_[i]`` at each knot (so ``y''(x0) = 0``
   and ``y''(x_{n-1}) = 0``).
2. Apply Hyman's (1983) monotonicity filter to those derivatives —
   clipping each interior slope to a monotonicity-preserving bound
   built from the surrounding chord slopes ``S_[i-1]``, ``S_[i]``,
   ``S_[i+1]`` (with edge cases at i=0 and i=n-1).
3. Re-emit cubic-Hermite coefficients from the *filtered* derivatives,
   piecewise on each interval ``[x_i, x_{i+1}]``.

The PCHIP-based ``MonotonicCubicNaturalSpline`` (delegated to
``scipy.interpolate.PchipInterpolator``) lives in
:mod:`pquantlib.math.interpolations.cubic_interpolation`; that one
derives slopes from scratch via Fritsch-Carlson. This module ports the
*natural-spline-then-filter* algorithm used by QuantLib's C++ source.
On C^2-smooth inputs the two algorithms agree at pillars (exact) but
disagree at intermediate points by ~1e-2 magnitude.

The Hyman criterion is described in:

    R. L. Dougherty, A. Edelman, J. M. Hyman.
    *Nonnegativity-, Monotonicity-, or Convexity-preserving Cubic and
    Quintic Hermite Interpolation.*
    Mathematics of Computation, 52(186):471-494, 1989.

Cross-validated against the C++ probe in
``migration-harness/cpp/probes/cluster_l10c/probe.cpp`` —
``HymanFilteredCubic`` pillar values match EXACT, intermediate values
match TIGHT (we share the same algorithm).
"""

from __future__ import annotations

import math

import numpy as np

from pquantlib.math.array import Array
from pquantlib.math.interpolations.interpolation import Interpolation


def _solve_natural_spline_derivatives(
    xs: np.ndarray,
    ys: np.ndarray,
) -> np.ndarray:
    """Solve the tridiagonal natural-cubic-spline first-derivative system.

    # C++ parity: cubicinterpolation.hpp:402-481 — the ``da==Spline``
    # arm of ``CubicInterpolationImpl::update`` with
    # ``leftType == rightType == SecondDerivative == 0.0``.

    The interior rows of the tridiagonal system are
    ``L[i] = (dx[i], 2*(dx[i]+dx[i-1]), dx[i-1])`` with RHS
    ``3*(dx[i]*S[i-1] + dx[i-1]*S[i])`` for ``i in 1..n-2``. The natural
    BC gives row 0 = ``(2, 1)`` with RHS ``3*S[0]``, and row n-1 =
    ``(1, 2)`` with RHS ``3*S[n-2]``.
    """
    n = xs.shape[0]
    dx = np.diff(xs)
    s = np.diff(ys) / dx
    # Allocate the tridiagonal coefficients (lower, diagonal, upper) and
    # RHS. We use a dense form; SciPy has no first-class tridiagonal
    # solver that beats a small numpy ``solve`` at this size.
    a = np.zeros(n, dtype=np.float64)  # sub-diagonal (row i, col i-1)
    b = np.zeros(n, dtype=np.float64)  # diagonal
    c = np.zeros(n, dtype=np.float64)  # super-diagonal
    rhs = np.zeros(n, dtype=np.float64)
    # Left BC: SecondDerivative=0 → b[0]=2, c[0]=1, rhs[0]=3*S[0]
    b[0] = 2.0
    c[0] = 1.0
    rhs[0] = 3.0 * s[0]
    # Interior rows
    for i in range(1, n - 1):
        a[i] = dx[i]
        b[i] = 2.0 * (dx[i] + dx[i - 1])
        c[i] = dx[i - 1]
        rhs[i] = 3.0 * (dx[i] * s[i - 1] + dx[i - 1] * s[i])
    # Right BC: SecondDerivative=0 → a[n-1]=1, b[n-1]=2, rhs[n-1]=3*S[n-2]
    a[n - 1] = 1.0
    b[n - 1] = 2.0
    rhs[n - 1] = 3.0 * s[n - 2]
    # Thomas (forward sweep, back substitution).
    for i in range(1, n):
        m = a[i] / b[i - 1]
        b[i] -= m * c[i - 1]
        rhs[i] -= m * rhs[i - 1]
    tmp = np.zeros(n, dtype=np.float64)
    tmp[n - 1] = rhs[n - 1] / b[n - 1]
    for i in range(n - 2, -1, -1):
        tmp[i] = (rhs[i] - c[i] * tmp[i + 1]) / b[i]
    return tmp


def _apply_hyman_filter(
    tmp: np.ndarray,
    s: np.ndarray,
    dx: np.ndarray,
) -> np.ndarray:
    """Apply the Hyman-1983 monotonicity filter to first derivatives.

    # C++ parity: cubicinterpolation.hpp:682-752 — the
    # ``if (monotonic_)`` block inside ``CubicInterpolationImpl::update``.

    Returns a *new* array of filtered derivatives; ``tmp`` is left
    untouched. Edge derivatives (i=0, i=n-1) are clipped to
    ``min(|tmp|, |3 * S_endpoint|)`` if they share sign with the
    adjacent chord slope, otherwise zeroed. Interior derivatives are
    bounded by ``M = 3 * min(|S[i-1]|, |S[i]|, |pm|)``, possibly
    enlarged via the second-neighbor "pu"/"pd" rules when the local
    curvature is monotone.
    """
    n = tmp.shape[0]
    out = tmp.copy()
    for i in range(n):
        if i == 0:
            if tmp[i] * s[0] > 0.0:
                correction = (tmp[i] / abs(tmp[i])) * min(abs(tmp[i]), abs(3.0 * s[0]))
            else:
                correction = 0.0
            if correction != tmp[i]:
                out[i] = correction
        elif i == n - 1:
            if tmp[i] * s[n - 2] > 0.0:
                correction = (tmp[i] / abs(tmp[i])) * min(
                    abs(tmp[i]), abs(3.0 * s[n - 2])
                )
            else:
                correction = 0.0
            if correction != tmp[i]:
                out[i] = correction
        else:
            pm = (s[i - 1] * dx[i] + s[i] * dx[i - 1]) / (dx[i - 1] + dx[i])
            m_val = 3.0 * min(abs(s[i - 1]), abs(s[i]), abs(pm))
            if i > 1 and (s[i - 1] - s[i - 2]) * (s[i] - s[i - 1]) > 0.0:
                pd = (
                    s[i - 1] * (2.0 * dx[i - 1] + dx[i - 2])
                    - s[i - 2] * dx[i - 1]
                ) / (dx[i - 2] + dx[i - 1])
                if pm * pd > 0.0 and pm * (s[i - 1] - s[i - 2]) > 0.0:
                    m_val = max(m_val, 1.5 * min(abs(pm), abs(pd)))
            if i < n - 2 and (s[i] - s[i - 1]) * (s[i + 1] - s[i]) > 0.0:
                pu = (s[i] * (2.0 * dx[i] + dx[i + 1]) - s[i + 1] * dx[i]) / (
                    dx[i] + dx[i + 1]
                )
                if pm * pu > 0.0 and -pm * (s[i] - s[i - 1]) > 0.0:
                    m_val = max(m_val, 1.5 * min(abs(pm), abs(pu)))
            correction = (
                (tmp[i] / abs(tmp[i])) * min(abs(tmp[i]), m_val)
                if tmp[i] * pm > 0.0
                else 0.0
            )
            if correction != tmp[i]:
                out[i] = correction
    return out


class HymanFilteredCubic(Interpolation):
    """Natural cubic spline with Hyman-1983 monotonicity filter.

    # C++ parity: ``CubicInterpolation(Spline, monotonic=true,
    #             SecondDerivative=0, SecondDerivative=0)``
    #             (cubicinterpolation.hpp:402-770).

    Algorithm:

    1. Solve the natural-cubic-spline tridiagonal system for the
       first derivatives at the pillars.
    2. Clip those derivatives via the Hyman-1983 monotonicity filter.
    3. Build cubic-Hermite coefficients ``(a, b, c)`` per interval
       from the filtered derivatives, where the cubic on
       ``[x_i, x_{i+1}]`` is

       .. math::

          y(x) = y_i + a_i (x - x_i) + b_i (x - x_i)^2 + c_i (x - x_i)^3

       with ``a_i = tmp[i]``, ``b_i = (3 S[i] - tmp[i+1] - 2 tmp[i]) / dx[i]``
       and ``c_i = (tmp[i+1] + tmp[i] - 2 S[i]) / dx[i]^2``
       (cubicinterpolation.hpp:756-760).

    Pillar values match the input data exactly (the natural-spline +
    Hyman filter does not perturb knot heights). Intermediate values
    match the C++ implementation to TIGHT tolerance.
    """

    def __init__(self, x_seq: Array, y_seq: Array) -> None:
        super().__init__(x_seq, y_seq, required_points=2)
        n = self._xs.shape[0]
        # Coefficients per interval — populated by ``update``.
        self._a_coef: np.ndarray = np.zeros(max(n - 1, 1), dtype=np.float64)
        self._b_coef: np.ndarray = np.zeros(max(n - 1, 1), dtype=np.float64)
        self._c_coef: np.ndarray = np.zeros(max(n - 1, 1), dtype=np.float64)
        # Primitive constants per interval — cumulative integral
        # F(x_i) = sum_{k<i} ∫_{x_k}^{x_{k+1}} y(t) dt, so that the
        # primitive at x is F(x_i) + ∫_{x_i}^{x} y(t) dt with
        # F(x_0) = 0. Mirrors C++ ``primitiveConst_``.
        self._prim_const: np.ndarray = np.zeros(max(n, 1), dtype=np.float64)
        self.update()

    def update(self) -> None:
        n = self._xs.shape[0]
        if n < 2:
            return
        dx = np.diff(self._xs)
        s = np.diff(self._ys) / dx
        # 1) Natural-spline first derivatives at pillars.
        tmp = _solve_natural_spline_derivatives(self._xs, self._ys)
        # 2) Hyman filter.
        tmp = _apply_hyman_filter(tmp, s, dx)
        # 3) Cubic-Hermite per-interval coefficients.
        a = tmp[:-1].copy()
        b = (3.0 * s - tmp[1:] - 2.0 * tmp[:-1]) / dx
        c = (tmp[1:] + tmp[:-1] - 2.0 * s) / (dx * dx)
        self._a_coef = a
        self._b_coef = b
        self._c_coef = c
        # Primitive constants. The integral of
        # y(x_i + h) = y_i + a h + b h^2 + c h^3 over h ∈ [0, dx_i] is
        #   y_i dx_i + a dx_i^2/2 + b dx_i^3/3 + c dx_i^4/4.
        # We accumulate to get F(x_i) for i ≥ 1, with F(x_0) = 0.
        n_int = a.shape[0]
        prim = np.zeros(n, dtype=np.float64)
        for i in range(n_int):
            h = dx[i]
            seg = (
                self._ys[i] * h
                + 0.5 * a[i] * h * h
                + (b[i] * h * h * h) / 3.0
                + 0.25 * c[i] * h * h * h * h
            )
            prim[i + 1] = prim[i] + seg
        self._prim_const = prim

    def _value(self, x: float) -> float:
        i = self._locate(x)
        h = x - float(self._xs[i])
        return float(
            self._ys[i] + self._a_coef[i] * h + self._b_coef[i] * h * h
            + self._c_coef[i] * h * h * h
        )

    def _derivative(self, x: float) -> float:
        i = self._locate(x)
        h = x - float(self._xs[i])
        return float(
            self._a_coef[i] + 2.0 * self._b_coef[i] * h + 3.0 * self._c_coef[i] * h * h
        )

    def _second_derivative(self, x: float) -> float:
        i = self._locate(x)
        h = x - float(self._xs[i])
        return float(2.0 * self._b_coef[i] + 6.0 * self._c_coef[i] * h)

    def _primitive(self, x: float) -> float:
        i = self._locate(x)
        h = x - float(self._xs[i])
        seg = (
            float(self._ys[i]) * h
            + 0.5 * float(self._a_coef[i]) * h * h
            + (float(self._b_coef[i]) * h * h * h) / 3.0
            + 0.25 * float(self._c_coef[i]) * h * h * h * h
        )
        return float(self._prim_const[i]) + seg

    # Convenience inspector — useful for downstream debugging /
    # cross-validation against C++ ``CubicInterpolation::aCoefficients``
    # (cubicinterpolation.hpp:198-200).
    def a_coefficients(self) -> np.ndarray:
        return self._a_coef.copy()

    def b_coefficients(self) -> np.ndarray:
        return self._b_coef.copy()

    def c_coefficients(self) -> np.ndarray:
        return self._c_coef.copy()


# Public helper — used by tests and downstream callers wanting to verify
# the Hyman filter's monotonicity-preservation contract.
def is_monotone(values: np.ndarray, tol: float = 0.0) -> bool:
    """Check whether ``values`` is weakly monotone (up or down).

    Used by tests to verify the Hyman filter's monotonicity-preservation
    contract (interpolated values on a monotone input remain monotone).
    """
    diff = np.diff(values)
    if diff.size == 0:
        return True
    up = np.all(diff >= -tol)
    down = np.all(diff <= tol)
    return bool(up or down) and bool(math.isfinite(float(values[0])))
