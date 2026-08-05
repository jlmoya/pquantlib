"""GMRES — generalized minimal residual method.

# C++ parity: ql/math/matrixutilities/gmres.{hpp,cpp} (v1.43).

**This is not ``scipy.sparse.linalg.gmres``.** The C++ is the plain
Arnoldi + Givens-rotation formulation of Saad (1996) / Kanzow, and it differs
from scipy's in what it returns and when it stops:

* :class:`GMRESResult` carries the **full per-iteration error history**
  (``||z[j+1]|| / ||b||`` after each Givens rotation), not an info flag; the
  history is what callers plot and what the restart loop concatenates;
* restarting is an explicit outer loop (:meth:`solve_with_restart`) that feeds
  the previous ``x`` back in and glues the error lists together, rather than a
  ``restart=`` keyword;
* the Arnoldi breakdown test is ``h[j+1][j] < QL_EPSILON**2`` (~4.9e-32), an
  unusually tight threshold that leaves the loop *before* the corresponding
  error is recorded;
* the preconditioner is applied on the **right** (``A M^-1 y``, with
  ``x = x0 + M^-1 y``), which changes both the Krylov space and the meaning of
  the recorded errors;
* non-convergence raises rather than returning a code.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array

#: ``Array -> Array`` operator; C++ ``GMRES::MatrixMult``.
type MatrixMult = Callable[[Array], Array]

_QL_EPSILON: float = float(np.finfo(np.float64).eps)


@dataclass(frozen=True, slots=True)
class GMRESResult:
    """Outcome of a GMRES solve.

    # C++ parity: ``struct GMRESResult`` (gmres.hpp:48-51) — ``errors`` is the
    # C++ ``std::list<Real>``.
    """

    errors: list[float]
    x: Array


def _dot(v1: Array, v2: Array) -> float:
    """``DotProduct`` — C++ parity: ql/math/array.hpp:541-546."""
    return float(np.dot(v1, v2))


def _norm2(v: Array) -> float:
    """``Norm2`` — C++ parity: ql/math/array.hpp:548-550."""
    return math.sqrt(_dot(v, v))


class GMRES:
    """Generalized minimal residual solver.

    # C++ parity: ``class GMRES`` (gmres.hpp:53-69, gmres.cpp:33-145).

    Args:
        a: the operator ``x -> A x``.
        max_iter: Arnoldi steps per cycle; must be positive.
        rel_tol: relative residual tolerance.
        pre_conditioner: optional right preconditioner ``x -> M^-1 x``.
    """

    __slots__ = ("_a", "_m", "_max_iter", "_rel_tol")

    def __init__(
        self,
        a: MatrixMult,
        max_iter: int,
        rel_tol: float,
        pre_conditioner: MatrixMult | None = None,
    ) -> None:
        # C++ parity: gmres.cpp:33-37.
        self._a: MatrixMult = a
        self._m: MatrixMult | None = pre_conditioner
        self._max_iter: int = max_iter
        self._rel_tol: float = rel_tol

        qassert.require(self._max_iter > 0, "maxIter must be greater than zero")

    def solve(self, b: Array, x0: Array | None = None) -> GMRESResult:
        """One GMRES cycle of at most ``max_iter`` steps.

        # C++ parity: ``GMRES::solve`` (gmres.cpp:39-45).
        """
        result = self._solve_impl(b, x0)
        qassert.require(result.errors[-1] < self._rel_tol, "could not converge")
        return result

    def solve_with_restart(
        self, restart: int, b: Array, x0: Array | None = None
    ) -> GMRESResult:
        """Up to ``restart`` GMRES cycles, each restarted from the previous ``x``.

        # C++ parity: ``GMRES::solveWithRestart`` (gmres.cpp:47-64).

        Note: C++'s loop bound is ``restart - 1`` on an unsigned ``Size``, so
        ``restart == 0`` wraps to a near-infinite loop there. Python's
        ``range(restart - 1)`` is empty instead — the divergence only bites a
        nonsensical argument.
        """
        result = self._solve_impl(b, x0)

        errors = list(result.errors)

        for _ in range(restart - 1):
            if result.errors[-1] < self._rel_tol:
                break
            result = self._solve_impl(b, result.x)
            errors.extend(result.errors)

        qassert.require(errors[-1] < self._rel_tol, "could not converge")

        return GMRESResult(errors=errors, x=result.x)

    def _solve_impl(self, b: Array, x0: Array | None) -> GMRESResult:
        # C++ parity: ``GMRES::solveImpl`` (gmres.cpp:66-145).
        b_arr = np.asarray(b, dtype=np.float64)
        bn = _norm2(b_arr)
        if bn == 0.0:
            return GMRESResult(errors=[0.0], x=b_arr.astype(np.float64, copy=True))

        x: Array = (
            np.asarray(x0, dtype=np.float64).astype(np.float64, copy=True)
            if x0 is not None and x0.size != 0
            else np.zeros(int(b_arr.shape[0]), dtype=np.float64)
        )
        r: Array = b_arr - self._a(x)

        g = _norm2(r)
        if g / bn < self._rel_tol:
            return GMRESResult(errors=[g / bn], x=x)

        v: list[Array] = [r / g]
        h: list[Array] = [np.zeros(self._max_iter, dtype=np.float64)]
        c: list[float] = [0.0] * (self._max_iter + 1)
        s: list[float] = [0.0] * (self._max_iter + 1)
        z: list[float] = [0.0] * (self._max_iter + 1)

        z[0] = g

        errors: list[float] = [g / bn]

        for j in range(self._max_iter):
            if errors[-1] < self._rel_tol:
                break
            h.append(np.zeros(self._max_iter, dtype=np.float64))
            w: Array = self._a(v[j] if self._m is None else self._m(v[j]))

            for i in range(j + 1):
                h[i][j] = _dot(w, v[i])
                w = w - float(h[i][j]) * v[i]

            h[j + 1][j] = _norm2(w)

            if float(h[j + 1][j]) < _QL_EPSILON * _QL_EPSILON:
                break

            v.append(w / float(h[j + 1][j]))

            for i in range(j):
                h0 = c[i] * float(h[i][j]) + s[i] * float(h[i + 1][j])
                h1 = -s[i] * float(h[i][j]) + c[i] * float(h[i + 1][j])

                h[i][j] = h0
                h[i + 1][j] = h1

            nu = math.sqrt(float(h[j][j]) * float(h[j][j]) + float(h[j + 1][j]) * float(h[j + 1][j]))

            c[j] = float(h[j][j]) / nu
            s[j] = float(h[j + 1][j]) / nu

            h[j][j] = nu
            h[j + 1][j] = 0.0

            z[j + 1] = -s[j] * z[j]
            z[j] = c[j] * z[j]

            errors.append(abs(z[j + 1] / bn))

        k = len(v) - 1

        y: Array = np.zeros(k, dtype=np.float64)
        y[k - 1] = z[k - 1] / float(h[k - 1][k - 1])

        for i in range(k - 2, -1, -1):
            # C++ parity: std::inner_product over h[i][i+1 .. k-1] and
            # y[i+1 .. k-1], accumulated left to right from 0.0.
            acc = 0.0
            for jj in range(i + 1, k):
                acc += float(h[i][jj]) * float(y[jj])
            y[i] = (z[i] - acc) / float(h[i][i])

        # C++ parity: std::inner_product over v[0 .. k-1] and y[0 .. k-1],
        # accumulated into a zero Array.
        xm: Array = np.zeros(int(x.shape[0]), dtype=np.float64)
        for i in range(k):
            xm = xm + v[i] * float(y[i])

        xm = x + (xm if self._m is None else self._m(xm))

        return GMRESResult(errors=errors, x=xm)
