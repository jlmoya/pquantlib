"""BiCGstab — biconjugate gradient stabilized iterative solver.

# C++ parity: ql/math/matrixutilities/bicgstab.{hpp,cpp} (v1.43).

Matrix-free Krylov solver: the operator and the preconditioner are
supplied as callables ``Array -> Array``, so the system matrix is never
formed. Used by :class:`ImplicitEulerScheme` for multi-dimensional
operators, where the implicit step cannot be reduced to a single
tridiagonal solve.

The iteration stops when ``||b - A x|| / ||b|| < rel_tol``; the solution
therefore agrees with the exact one only to ``rel_tol``, and any test
that compares against a C++ run through this solver inherits that
bound.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array

#: Matrix-free operator application.
type MatrixMult = Callable[[Array], Array]


@dataclass(frozen=True, slots=True)
class BiCGStabResult:
    """Iteration count, final relative residual and solution.

    # C++ parity: ``struct BiCGStabResult``.
    """

    iterations: int
    error: float
    x: Array


def _norm2(v: Array) -> float:
    """``sqrt(DotProduct(v, v))``.

    # C++ parity: ``Norm2`` (ql/math/array.hpp), built on
    # ``std::inner_product``. numpy dispatches the dot product to BLAS,
    # whose accumulation order differs from a sequential sum; the two
    # agree to a few ULP, which perturbs the reported residual but not
    # the ``rel_tol`` stopping decision except in a measure-zero tie.
    """
    return math.sqrt(float(np.dot(v, v)))


class BiCGstab:
    """Preconditioned BiCGstab.

    # C++ parity: ``class BiCGstab``.

    Args:
        a: the operator ``x -> A x``.
        max_iter: iteration cap; exceeding it raises.
        rel_tol: relative-residual stopping threshold.
        pre_conditioner: optional ``x -> M^-1 x``.
    """

    def __init__(
        self,
        a: MatrixMult,
        max_iter: int,
        rel_tol: float,
        pre_conditioner: MatrixMult | None = None,
    ) -> None:
        self._a: MatrixMult = a
        self._m: MatrixMult | None = pre_conditioner
        self._max_iter: int = max_iter
        self._rel_tol: float = rel_tol

    def solve(self, b: Array, x0: Array | None = None) -> BiCGStabResult:
        """Solve ``A x = b``.

        # C++ parity: ``BiCGstab::solve``.
        """
        bnorm2 = _norm2(b)
        if bnorm2 == 0.0:
            return BiCGStabResult(0, 0.0, b)

        x: Array = np.array(x0, dtype=np.float64) if x0 is not None else np.zeros(b.size)
        r: Array = b - self._a(x)

        r_tld: Array = r.copy()
        p: Array = np.zeros(b.size)
        v: Array = np.zeros(b.size)
        omega = 1.0
        rho_tld = 1.0
        alpha = 0.0
        error = _norm2(r) / bnorm2

        i = 0
        while i < self._max_iter and error >= self._rel_tol:
            rho = float(np.dot(r_tld, r))
            if rho == 0.0 or omega == 0.0:
                break

            if i != 0:
                beta = (rho / rho_tld) * (alpha / omega)
                p = r + beta * (p - omega * v)
            else:
                p = r

            p_tld: Array = p if self._m is None else self._m(p)
            v = self._a(p_tld)

            alpha = rho / float(np.dot(r_tld, v))
            s: Array = r - alpha * v
            if _norm2(s) < self._rel_tol * bnorm2:
                x = x + alpha * p_tld
                error = _norm2(s) / bnorm2
                break

            s_tld: Array = s if self._m is None else self._m(s)
            t: Array = self._a(s_tld)
            omega = float(np.dot(t, s)) / float(np.dot(t, t))
            x = x + alpha * p_tld + omega * s_tld
            r = s - omega * t
            error = _norm2(r) / bnorm2
            rho_tld = rho
            i += 1

        qassert.require(i < self._max_iter, "max number of iterations exceeded")
        qassert.require(error < self._rel_tol, "could not converge")

        return BiCGStabResult(i, error, x)


__all__ = ["BiCGStabResult", "BiCGstab", "MatrixMult"]
