"""Shared operator plumbing for the BiCGstab / GMRES cross-validation tests.

The C++ probe drives both solvers through a *recording* ``MatrixMult``: it logs
``Norm2`` of every argument handed to the operator (and to the preconditioner)
before delegating. Replaying that instrumentation on the Python side pins the
iteration's whole trajectory — which vectors it visits, in which order — and
not merely the point it converges to. A different Bi-CG-Stab or GMRES (scipy's,
say) lands on the same solution through a different sequence of operator
applications, and the trace is what catches that.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from pquantlib.math.array import Array
from pquantlib.math.matrix import Matrix
from pquantlib.math.matrixutilities.sparse_ilu_preconditioner import SparseILUPreconditioner


def matvec(a: Matrix, x: Array) -> Array:
    """``A @ x`` accumulated exactly as C++ ``operator*(Matrix, Array)`` does.

    C++ uses ``std::inner_product`` per row, i.e. a left-to-right scalar
    accumulation. numpy's ``@`` routes through BLAS ``dgemv``, whose blocking
    is free to reassociate; the explicit loop keeps the summation order the
    same on both sides so the pinned traces mean what they say.
    """
    out: Array = np.zeros(int(a.shape[0]), dtype=np.float64)
    for i in range(a.shape[0]):
        acc = 0.0
        for j in range(a.shape[1]):
            acc += float(x[j]) * float(a[i, j])
        out[i] = acc
    return out


def norm2(v: Array) -> float:
    """``Norm2`` — ``sqrt(DotProduct(v, v))``, as ql/math/array.hpp."""
    return math.sqrt(float(np.dot(v, v)))


class Recorded:
    """Operator pair for one probe case, with the argument-norm logs."""

    __slots__ = ("a_norms", "apply_a", "apply_m", "m_norms")

    def __init__(self, case: dict[str, Any]) -> None:
        a: Matrix = np.asarray(case["matrix"], dtype=np.float64)
        self.a_norms: list[float] = []
        self.m_norms: list[float] = []

        def apply_a(x: Array) -> Array:
            self.a_norms.append(norm2(x))
            return matvec(a, x)

        self.apply_a: Callable[[Array], Array] = apply_a
        self.apply_m: Callable[[Array], Array] | None = None

        if case["preconditioner"] == "ilu":
            ilu = SparseILUPreconditioner(a)

            def apply_ilu(x: Array) -> Array:
                self.m_norms.append(norm2(x))
                return ilu.apply(x)

            self.apply_m = apply_ilu
        elif case["preconditioner"] == "jacobi":
            diag: Array = np.array([a[i, i] for i in range(a.shape[0])], dtype=np.float64)

            def apply_jacobi(x: Array) -> Array:
                self.m_norms.append(norm2(x))
                out: Array = np.zeros(int(x.shape[0]), dtype=np.float64)
                for i in range(x.shape[0]):
                    out[i] = float(x[i]) / float(diag[i])
                return out

            self.apply_m = apply_jacobi


def x0_of(case: dict[str, Any]) -> Array | None:
    """The probe's ``x0``, or ``None`` where C++ passed an empty ``Array``."""
    if not case["x0"]:
        return None
    return np.asarray(case["x0"], dtype=np.float64)
