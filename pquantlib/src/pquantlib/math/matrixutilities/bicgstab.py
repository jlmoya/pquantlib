"""BiCGstab — biconjugate gradient stabilized method.

# C++ parity: ql/math/matrixutilities/bicgstab.{hpp,cpp} (v1.43).

**This is not ``scipy.sparse.linalg.bicgstab``.** The two share a name and a
textbook, not an implementation:

* the operator is a *function object* ``Array -> Array`` (``MatrixMult``), and
  so is the preconditioner — there is no matrix object at all;
* the convergence test is ``||r||_2 / ||b||_2 < relTol``, checked at the *top*
  of each sweep, with no absolute-tolerance term (scipy tests
  ``||r|| <= max(rtol*||b||, atol)``);
* the breakdown test is the exact equality ``rho == 0.0 or omega == 0.0`` and
  it simply leaves the loop, letting the trailing ``QL_REQUIRE`` turn it into
  an error;
* there is an early exit when ``||s|| < relTol * ||b||``, taken *before*
  ``omega`` is formed, which updates ``x`` with the ``alpha`` step only;
* the result carries the **iteration count** and the final relative error, and
  non-convergence raises rather than returning an info flag.

Every one of those changes where the iteration stops and what ``x`` it stops
at, so the loop is transcribed.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array

#: ``Array -> Array`` operator; C++ ``BiCGstab::MatrixMult``.
type MatrixMult = Callable[[Array], Array]


@dataclass(frozen=True, slots=True)
class BiCGStabResult:
    """Outcome of a :meth:`BiCGstab.solve` call.

    # C++ parity: ``struct BiCGStabResult`` (bicgstab.hpp:33-37).
    """

    iterations: int
    error: float
    x: Array


def _dot(v1: Array, v2: Array) -> float:
    """``DotProduct`` — sequential accumulation, as ``std::inner_product``.

    # C++ parity: ql/math/array.hpp:541-546.
    """
    return float(np.dot(v1, v2))


def _norm2(v: Array) -> float:
    """``Norm2`` — ``sqrt(DotProduct(v, v))``.

    # C++ parity: ql/math/array.hpp:548-550. Deliberately not
    # ``numpy.linalg.norm``, which rescales to avoid overflow and therefore
    # rounds differently.
    """
    return math.sqrt(_dot(v, v))


class BiCGstab:
    """Bi-conjugate gradient stabilized solver.

    # C++ parity: ``class BiCGstab`` (bicgstab.hpp:39-51,
    # bicgstab.cpp:31-92).

    Args:
        a: the operator ``x -> A x``.
        max_iter: iteration cap; exceeding it raises.
        rel_tol: relative residual tolerance.
        pre_conditioner: optional operator ``x -> M^-1 x``.
    """

    __slots__ = ("_a", "_m", "_max_iter", "_rel_tol")

    def __init__(
        self,
        a: MatrixMult,
        max_iter: int,
        rel_tol: float,
        pre_conditioner: MatrixMult | None = None,
    ) -> None:
        # C++ parity: bicgstab.cpp:31-35.
        self._a: MatrixMult = a
        self._m: MatrixMult | None = pre_conditioner
        self._max_iter: int = max_iter
        self._rel_tol: float = rel_tol

    def solve(self, b: Array, x0: Array | None = None) -> BiCGStabResult:
        """Solve ``A x = b``.

        # C++ parity: ``BiCGstab::solve`` (bicgstab.cpp:37-92).

        Raises:
            LibraryException: if the iteration cap is hit or the residual never
                falls below ``rel_tol`` (mirroring the two trailing
                ``QL_REQUIRE``\\ s).
        """
        b_arr = np.asarray(b, dtype=np.float64)
        bnorm2 = _norm2(b_arr)
        if bnorm2 == 0.0:
            return BiCGStabResult(iterations=0, error=0.0, x=b_arr.astype(np.float64, copy=True))

        x: Array = (
            np.asarray(x0, dtype=np.float64).astype(np.float64, copy=True)
            if x0 is not None and x0.size != 0
            else np.zeros(int(b_arr.shape[0]), dtype=np.float64)
        )
        r: Array = b_arr - self._a(x)

        r_tld: Array = r.astype(np.float64, copy=True)
        p: Array = np.zeros(0, dtype=np.float64)
        v: Array = np.zeros(0, dtype=np.float64)
        omega = 1.0
        rho_tld = 1.0
        alpha = 0.0
        error = _norm2(r) / bnorm2

        i = 0
        while i < self._max_iter and error >= self._rel_tol:
            rho = _dot(r_tld, r)
            if rho == 0.0 or omega == 0.0:
                break

            if i != 0:
                beta = (rho / rho_tld) * (alpha / omega)
                p = r + beta * (p - omega * v)
            else:
                p = r

            p_tld = p if self._m is None else self._m(p)
            v = self._a(p_tld)

            alpha = rho / _dot(r_tld, v)
            s = r - alpha * v
            if _norm2(s) < self._rel_tol * bnorm2:
                x = x + alpha * p_tld
                error = _norm2(s) / bnorm2
                break

            s_tld = s if self._m is None else self._m(s)
            t = self._a(s_tld)
            omega = _dot(t, s) / _dot(t, t)
            # C++ ``x += alpha*pTld + omega*sTld`` forms the bracketed sum
            # first; the parentheses are load-bearing for the rounding.
            x = x + (alpha * p_tld + omega * s_tld)
            r = s - omega * t
            error = _norm2(r) / bnorm2
            rho_tld = rho
            i += 1

        qassert.require(i < self._max_iter, "max number of iterations exceeded")
        qassert.require(error < self._rel_tol, "could not converge")

        return BiCGStabResult(iterations=i, error=error, x=x)
