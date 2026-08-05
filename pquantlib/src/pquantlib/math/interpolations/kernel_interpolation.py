"""Kernel interpolation between discrete points.

# C++ parity: ql/math/interpolations/kernelinterpolation.hpp (v1.43).

Hakala/Wystup, *Foreign Exchange Risk*, p. 256. The interpolant is a
gamma-normalised kernel expansion::

    f(x) = sum_i alpha_i K(|x - x_i|) / gamma(x)
    gamma(x) = sum_i K(|x - x_i|)

with the weights ``alpha`` fixed by requiring ``f(x_r) == y_r`` at every
node, i.e. by solving the dense linear system ``M alpha = y`` where::

    M[r][c] = K(|x_r - x_c|) / gamma(x_r)

The kernel is deliberately generic — C++ takes it as a template parameter
requiring only ``Real operator()(Real)``, so a plain function pointer is
legal (the C++ test-suite passes ``&epanechnikovKernel``). This port
accepts any ``Callable[[float], float]``, including a
:class:`~pquantlib.math.kernel_functions.KernelFunction`.

Two details are load-bearing and reproduced exactly:

- The kernel is always called on the **absolute** difference,
  ``K(|x1 - x2|)`` (``kernelAbs``), never on the signed one. For a
  symmetric kernel that is a no-op; for an asymmetric callable it is not,
  and C++ commits to the absolute value.
- After the solve, C++ recomputes ``M alpha`` and fails if any component of
  ``|M alpha - y|`` reaches ``epsilon`` (default ``1e-7``). It deliberately
  does *not* pre-check ``det(M) != 0``; the residual check is the whole
  singularity guard. Kept, including the strict ``<``.

``primitive``, ``derivative`` and ``second_derivative`` all ``QL_FAIL`` in
C++ and raise here.

**Linear solve.** C++ calls ``qrSolve(M_, yVec_)`` — MINPACK ``qrfac`` +
``qrsolv``, i.e. Householder QR with column pivoting. This port calls
``numpy.linalg.solve`` (LAPACK ``gesv``, LU with partial pivoting), which
is the repo's existing precedent for a square dense solve
(``vanna_volga_barrier_engine``). For a square non-singular ``M`` the two
compute the same mathematical solution; they differ only by rounding, and
the difference is bounded by ``cond(M) * eps``. The cross-validation test
derives its tolerance from ``cond(M)`` for exactly that reason rather than
picking a round number. If/when ``qr_solve`` lands under
``pquantlib.math.matrixutilities`` this should switch to it and the
tolerance should tighten.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.interpolations.interpolation import Interpolation
from pquantlib.math.matrix import Matrix

_DEFAULT_EPSILON: float = 1.0e-7


def _ieee_divide(numerator: float, denominator: float) -> float:
    """``numerator / denominator`` with C++/IEEE-754 semantics at zero.

    C++ evaluates ``res / gammaFunc(x)`` as a plain double division, so a
    compactly-supported kernel evaluated where every node is out of support
    gives ``0.0 / 0.0 == nan`` (and a non-zero numerator a signed infinity).
    Python raises ``ZeroDivisionError`` instead, so the IEEE result is spelled
    out rather than left to blow up.
    """
    if denominator == 0.0:
        if numerator == 0.0:
            return math.nan
        return math.copysign(math.inf, numerator)
    return numerator / denominator


class KernelInterpolation(Interpolation):
    """Kernel interpolation through ``(x, y)`` with a caller-supplied kernel.

    # C++ parity: ``class KernelInterpolation : public Interpolation``
    # (kernelinterpolation.hpp:146-170), Impl at lines 36-129.

    Args:
        x_seq: abscissae; must be sorted ascending.
        y_seq: ordinates, same length as ``x_seq``.
        kernel: any callable taking one float and returning a float.
        epsilon: the inversion-precision guard. After solving
            ``M alpha = y`` the residual ``|M alpha - y|`` must be
            strictly below this in every component, else construction
            fails. C++ default ``1.0e-7``.
    """

    __slots__ = ("_alpha", "_inv_prec", "_kernel", "_m", "_x_size")

    def __init__(
        self,
        x_seq: Array,
        y_seq: Array,
        kernel: Callable[[float], float],
        epsilon: float = _DEFAULT_EPSILON,
    ) -> None:
        super().__init__(x_seq, y_seq, required_points=1)
        self._x_size: int = self._xs.shape[0]
        self._inv_prec: float = epsilon
        self._kernel: Callable[[float], float] = kernel
        self._m: Matrix = np.zeros((self._x_size, self._x_size), dtype=np.float64)
        self._alpha: Array = np.zeros(self._x_size, dtype=np.float64)
        # C++ parity: the KernelInterpolation ctor calls impl_->update()
        # immediately (kernelinterpolation.hpp:167).
        self.update()

    # ----- construction ---------------------------------------------------

    def update(self) -> None:
        """Recompute the kernel matrix and the weight vector.

        # C++ parity: ``KernelInterpolationImpl::update`` ->
        # ``updateAlphaVec`` (kernelinterpolation.hpp:49, 93-122).
        """
        n = self._x_size
        xs = self._xs
        m = np.zeros((n, n), dtype=np.float64)
        y_vec = np.zeros(n, dtype=np.float64)
        for row in range(n):
            y_vec[row] = float(self._ys[row])
            # C++ computes ``1.0 / gammaFunc(...)`` as a raw double division;
            # a kernel with no support at the node makes that an infinity
            # rather than an exception, and the resulting nan row is what the
            # post-solve residual guard is there to catch.
            tmp = _ieee_divide(1.0, self._gamma(float(xs[row])))
            for col in range(n):
                m[row, col] = self._kernel_abs(float(xs[row]), float(xs[col])) * tmp
        self._m = m

        # Solve y = M alpha for alpha.
        alpha = np.linalg.solve(m, y_vec)
        self._alpha = np.ascontiguousarray(alpha, dtype=np.float64)

        # C++ parity: kernelinterpolation.hpp:114-121 — no det(M) pre-check;
        # the residual IS the singularity guard, and the comparison is strict.
        diff = np.abs(m @ self._alpha - y_vec)
        for value in diff:
            qassert.require(
                float(value) < self._inv_prec,
                "Inversion failed in 1d kernel interpolation",
            )

    # ----- evaluation -----------------------------------------------------

    def _value(self, x: float) -> float:
        # C++ parity: kernelinterpolation.hpp:51-60.
        res = 0.0
        for i in range(self._x_size):
            res += float(self._alpha[i]) * self._kernel_abs(x, float(self._xs[i]))
        return _ieee_divide(res, self._gamma(x))

    def _primitive(self, x: float) -> float:
        del x
        qassert.fail("Primitive calculation not implemented for kernel interpolation")

    def _derivative(self, x: float) -> float:
        del x
        qassert.fail("First derivative calculation not implemented for kernel interpolation")

    def _second_derivative(self, x: float) -> float:
        del x
        qassert.fail("Second derivative calculation not implemented for kernel interpolation")

    # ----- inspectors (not in C++: the Impl members are private there) ----

    @property
    def alpha(self) -> Array:
        """The solved kernel weights, as a copy.

        Not exposed by C++ (``alphaVec_`` is a private Impl member); surfaced
        here because it is the whole content of the fit and makes the
        construction testable without reaching into privates.
        """
        return self._alpha.copy()

    @property
    def kernel_matrix(self) -> Matrix:
        """The gamma-normalised kernel matrix ``M``, as a copy.

        Not exposed by C++ (``M_`` is a private Impl member). See
        :attr:`alpha`.
        """
        return self._m.copy()

    # ----- helpers --------------------------------------------------------

    def _kernel_abs(self, x1: float, x2: float) -> float:
        # C++ parity: kernelinterpolation.hpp:79-81 — ALWAYS the absolute
        # difference, never the signed one.
        return self._kernel(abs(x1 - x2))

    def _gamma(self, x: float) -> float:
        # C++ parity: kernelinterpolation.hpp:83-91.
        res = 0.0
        for i in range(self._x_size):
            res += self._kernel_abs(x, float(self._xs[i]))
        return res


__all__ = ["KernelInterpolation"]
