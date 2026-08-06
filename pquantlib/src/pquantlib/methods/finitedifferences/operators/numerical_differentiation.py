"""NumericalDifferentiation — finite-difference weights on arbitrary grids.

# C++ parity: ql/methods/finitedifferences/operators/numericaldifferentiation.{hpp,cpp}
# (v1.43).

References:

* B. Fornberg, 1988. *Generation of Finite Difference Formulas on
  Arbitrarily Spaced Grids*.
* B. Fornberg, 1998. *Calculation of Weights in Finite Difference
  Formulas* — the recurrence actually implemented here.

Given ``N`` sample offsets ``x_0 ... x_{N-1}`` (relative to the
evaluation point) and a derivative order ``M < N``, the class produces
the ``N`` weights ``w_i`` such that

.. math::

    f^{(M)}(x) \\approx \\sum_i w_i f(x + x_i)

**Deliberately not delegated to numpy/scipy.** The C++ implementation
runs Fornberg's *recurrence* — not a Vandermonde linear solve — so the
weights carry a very specific round-off signature (e.g. the 3-point
central first-derivative weights come out as
``[-5, 5.55e-16, 4.999999999999999]`` for ``h = 0.1``, not
``[-5, 0, 5]``). ``numpy.linalg.solve`` on the equivalent Vandermonde
system agrees only to a few ULP and would silently drift from C++.
The recurrence is therefore ported statement-for-statement, in plain
Python floats (IEEE-754 binary64, identical to C++ ``Real``), and
cross-validated against the C++ probe.

**Fused multiply-add.** The two ``a*b - c`` sub-expressions of the
recurrence are contracted into a single ``fma`` by the C++ compiler
(Clang defaults to ``-ffp-contract=on``, GCC to ``fast``; both x86-64
FMA3 and AArch64 have the instruction). Measured against the v1.43
probe: the unfused Python form drifts from the C++ reference by up to
1.4e-15 *relative to the largest weight in the stencil*, whereas using
:func:`math.fma` in exactly those two places reproduces all 63 probed
weights **bit-for-bit**. The fused form is also strictly more accurate
(one rounding instead of two), so it is what this port uses; the
``# C++ parity`` comments below mark the two sites.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Sequence
from enum import IntEnum
from typing import final

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array

# C++ parity: ``QL_EPSILON`` == ``std::numeric_limits<Real>::epsilon()``.
_QL_EPSILON: float = sys.float_info.epsilon


class Scheme(IntEnum):
    """Offset layout for the step-size constructor.

    # C++ parity: ``NumericalDifferentiation::Scheme`` — a nested enum
    # in C++; PQuantLib spells nested C++ enums as module-level
    # ``IntEnum`` (same convention as ``FdmSchemeType`` /
    # ``BoundaryConditionSide``).
    """

    Central = 0
    Backward = 1
    Forward = 2


def _calc_offsets(h: float, n: int, scheme: Scheme) -> Array:
    """Build the offset grid for ``(step_size, steps, scheme)``.

    # C++ parity: anonymous-namespace ``calcOffsets`` in
    # numericaldifferentiation.cpp.
    """
    qassert.require(n > 1, "number of steps must be greater than one")

    if scheme == Scheme.Central:
        qassert.require(
            n > 2 and (n % 2) != 0,
            "number of steps must be an odd number greater than two",
        )
        # C++: retVal[i] = (i - Integer(n/2)) * h  (integer division).
        return np.array([(i - n // 2) * h for i in range(n)], dtype=np.float64)
    if scheme == Scheme.Backward:
        # C++: retVal[i] = -(i*h) — note the negation of the product, so
        # the first entry is -0.0, not +0.0.
        return np.array([-(i * h) for i in range(n)], dtype=np.float64)
    if scheme == Scheme.Forward:
        return np.array([i * h for i in range(n)], dtype=np.float64)
    qassert.fail("unknown numerical differentiation scheme")


def _calc_weights(x: Sequence[float], m_order: int) -> Array:
    """Fornberg (1998) recurrence for the finite-difference weights.

    # C++ parity: anonymous-namespace ``calcWeights`` in
    # numericaldifferentiation.cpp — ported statement-for-statement,
    # including the ``c1/c2*(...)`` association and the ``m > 0`` guards.
    """
    n_pts = len(x)
    qassert.require(
        n_pts > m_order,
        "number of points must be greater than the order of the derivative",
    )

    # C++ uses boost::multi_array<Real, 3> d(extents[M+1][N][N]), whose
    # elements are value-initialised to 0.0.
    d: list[list[list[float]]] = [
        [[0.0] * n_pts for _ in range(n_pts)] for _ in range(m_order + 1)
    ]
    d[0][0][0] = 1.0
    c1 = 1.0

    for n in range(1, n_pts):
        c2 = 1.0
        for nu in range(n):
            c3 = x[n] - x[nu]
            c2 *= c3

            for m in range(min(n, m_order) + 1):
                # C++: (x[n]*d[m][n-1][nu] - m*d[m-1][n-1][nu]) / c3,
                # contracted by the compiler into fma(x[n], d, -term).
                term = m * d[m - 1][n - 1][nu] if m > 0 else 0.0
                d[m][n][nu] = math.fma(x[n], d[m][n - 1][nu], -term) / c3

        for m in range(m_order + 1):
            # C++: c1/c2 * (m*d[m-1][n-1][n-1] - x[n-1]*d[m][n-1][n-1]),
            # the parenthesised difference contracted into
            # fma(-x[n-1], d, term).
            term = m * d[m - 1][n - 1][n - 1] if m > 0 else 0.0
            d[m][n][n] = c1 / c2 * math.fma(-x[n - 1], d[m][n - 1][n - 1], term)
        c1 = c2

    return np.array([d[m_order][n_pts - 1][i] for i in range(n_pts)], dtype=np.float64)


@final
class NumericalDifferentiation:
    """Numerical differentiation on arbitrarily spaced grids.

    # C++ parity: ``class NumericalDifferentiation``.

    The primary constructor takes explicit offsets (the C++
    ``(f, orderOfDerivative, Array x_offsets)`` overload);
    :meth:`from_scheme` is the ``(f, orderOfDerivative, stepSize, steps,
    scheme)`` overload, which Python cannot express as a second
    ``__init__``.
    """

    __slots__ = ("_f", "_offsets", "_w")

    def __init__(
        self,
        f: Callable[[float], float] | None,
        order_of_derivative: int,
        x_offsets: Array | Sequence[float],
    ) -> None:
        """Build from explicit offsets.

        # C++ parity: ``NumericalDifferentiation(std::function<Real(Real)>,
        # Size, Array)``. ``f`` may be ``None`` — C++ callers that only
        # want the weights (``NthOrderDerivativeOp``) pass an empty
        # ``std::function``.
        """
        offsets = np.asarray(x_offsets, dtype=np.float64)
        self._offsets: Array = offsets
        self._w: Array = _calc_weights([float(v) for v in offsets], order_of_derivative)
        self._f: Callable[[float], float] | None = f

    @classmethod
    def from_scheme(
        cls,
        f: Callable[[float], float] | None,
        order_of_derivative: int,
        step_size: float,
        steps: int,
        scheme: Scheme,
    ) -> NumericalDifferentiation:
        """Build from ``(step_size, steps, scheme)``.

        # C++ parity: ``NumericalDifferentiation(std::function<Real(Real)>,
        # Size, Real, Size, Scheme)``.
        """
        return cls(f, order_of_derivative, _calc_offsets(step_size, steps, scheme))

    def offsets(self) -> Array:
        """# C++ parity: ``const Array& offsets() const``."""
        return self._offsets

    def weights(self) -> Array:
        """# C++ parity: ``const Array& weights() const``."""
        return self._w

    def __call__(self, x: float) -> float:
        """Approximate the ``order``-th derivative of ``f`` at ``x``.

        # C++ parity: ``Real NumericalDifferentiation::operator()(Real)``
        # — weights whose magnitude does not exceed ``QL_EPSILON^2`` are
        # skipped, so ``f`` is never evaluated at those offsets.
        """
        f = self._f
        qassert.require(f is not None, "no function given to NumericalDifferentiation")
        assert f is not None
        s = 0.0
        cutoff = _QL_EPSILON * _QL_EPSILON
        w = self._w
        offsets = self._offsets
        for i in range(w.shape[0]):
            wi = float(w[i])
            if abs(wi) > cutoff:
                s += wi * f(x + float(offsets[i]))
        return s


__all__ = ["NumericalDifferentiation", "Scheme"]
