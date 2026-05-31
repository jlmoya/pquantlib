"""GaussianQuadrature — Gauss quadrature from an orthogonal polynomial.

# C++ parity: ql/math/integrals/gaussianquadratures.hpp (v1.42.1) —
# ``class GaussianQuadrature``.

Given a ``GaussianOrthogonalPolynomial`` (here any object exposing the
3-term-recurrence ``alpha(i)`` / ``beta(i)``, the zeroth moment
``mu_0()`` and the weight ``w(x)``), build the ``n``-point Gauss
quadrature rule via the Golub-Welsch algorithm:

* The nodes are the eigenvalues of the symmetric tridiagonal Jacobi
  matrix with diagonal ``alpha(i)`` and off-diagonal ``sqrt(beta(i))``.
* The weights are ``mu_0 * v0_i^2 / w(x_i)`` where ``v0_i`` is the first
  component of the ``i``-th normalised eigenvector.

(G.H. Golub & J.H. Welsch, "Calculation of Gauss quadrature rules",
Math. Comput. 23 (1969), 221-230.)

The C++ class uses its own ``TqrEigenDecomposition`` for the symmetric
tridiagonal eigenproblem; the Python port delegates to
``scipy.linalg.eigh_tridiagonal`` (consistent with the project's
numerical-tooling policy — eigen-decomposition is exactly the kind of
routine we delegate to scipy). The node/weight *formulae* are ported
line-for-line.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
from scipy.linalg import (  # pyright: ignore[reportMissingTypeStubs]
    eigh_tridiagonal,  # pyright: ignore[reportUnknownVariableType]
)

from pquantlib.math.array import Array


class GaussianOrthogonalPolynomial(Protocol):
    """Structural interface for a Gaussian orthogonal polynomial.

    # C++ parity: ``class GaussianOrthogonalPolynomial`` — the abstract
    # base in gaussianorthogonalpolynomial.hpp. The Python port uses a
    # Protocol so both the moment-based polynomials (e.g. the non-central
    # chi-squared) and any classical (Laguerre/Hermite/...) polynomial
    # can drive the quadrature.
    """

    def alpha(self, u: int) -> float: ...
    def beta(self, u: int) -> float: ...
    def mu_0(self) -> float: ...
    def w(self, x: float) -> float: ...


class GaussianQuadrature:
    """``n``-point Gauss quadrature for an orthogonal polynomial.

    # C++ parity: ``class GaussianQuadrature`` in
    # gaussianquadratures.hpp:47-77 (ctor at gaussianquadratures.cpp:33-61).
    """

    __slots__ = ("_w", "_x")

    def __init__(self, n: int, orth_poly: GaussianOrthogonalPolynomial) -> None:
        # C++ parity: gaussianquadratures.cpp:33-61 — Golub-Welsch.
        diag = np.empty(n, dtype=np.float64)
        off = np.empty(n - 1, dtype=np.float64)
        diag[0] = orth_poly.alpha(0)
        for i in range(1, n):
            diag[i] = orth_poly.alpha(i)
            off[i - 1] = np.sqrt(orth_poly.beta(i))

        # scipy.linalg.eigh_tridiagonal is untyped; the ndarray results are
        # immediately normalised into owned float64 arrays.
        eigenvalues, eigenvectors = eigh_tridiagonal(diag, off)  # pyright: ignore[reportUnknownVariableType]
        self._x: Array = np.ascontiguousarray(eigenvalues, dtype=np.float64)

        mu_0 = orth_poly.mu_0()
        w = np.empty(n, dtype=np.float64)
        first_row = np.ascontiguousarray(eigenvectors[0, :], dtype=np.float64)  # pyright: ignore[reportUnknownArgumentType]
        for i in range(n):
            w[i] = mu_0 * first_row[i] * first_row[i] / orth_poly.w(float(self._x[i]))
        self._w: Array = w

    def order(self) -> int:
        # C++ parity: gaussianquadratures.hpp:70.
        return self._x.shape[0]

    def x(self) -> Array:
        # C++ parity: gaussianquadratures.hpp:72.
        return self._x

    def weights(self) -> Array:
        # C++ parity: gaussianquadratures.hpp:71.
        return self._w

    def __call__(self, f: Callable[[float], float]) -> float:
        # C++ parity: gaussianquadratures.hpp:58-65 — sum w_i f(x_i),
        # accumulated from the largest index down (matching C++ exactly).
        total = 0.0
        for i in range(self.order() - 1, -1, -1):
            total += float(self._w[i]) * float(f(float(self._x[i])))
        return total


__all__ = ["GaussianOrthogonalPolynomial", "GaussianQuadrature"]
