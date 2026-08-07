"""Gauss quadratures built from an orthogonal polynomial.

# C++ parity: ql/math/integrals/gaussianquadratures.{hpp,cpp} (v1.43).

Given a :class:`GaussianOrthogonalPolynomial` (anything exposing the
three-term-recurrence ``alpha(i)`` / ``beta(i)``, the zeroth moment ``mu_0()``
and the weight ``w(x)``), build the ``n``-point Gauss quadrature rule via the
Golub-Welsch algorithm:

* the nodes are the eigenvalues of the symmetric tridiagonal Jacobi matrix
  with diagonal ``alpha(i)`` and off-diagonal ``sqrt(beta(i))``;
* the weights are ``mu_0 * v0_i^2 / w(x_i)`` where ``v0_i`` is the first
  component of the ``i``-th normalised eigenvector.

(G.H. Golub & J.H. Welsch, "Calculation of Gauss quadrature rules",
Math. Comput. 23 (1969), 221-230.)

Why the eigen-decomposition is NOT delegated to LAPACK
-----------------------------------------------------

C++ runs its own ``TqrEigenDecomposition`` (implicit-shift QL with the
"over-relaxation" Wilkinson shift) in ``OnlyFirstRowEigenVector`` mode, and
this port does the same. An earlier revision delegated the symmetric-
tridiagonal eigenproblem to ``scipy.linalg.eigh_tridiagonal`` on the reasoning
that both are backward-stable solvers for the same matrix. **They are, and
that is not sufficient here.**

A norm-wise backward-stable eigensolver guarantees each eigenvector to a small
error *relative to the vector's norm*. This algorithm needs something much
stronger: the weights are ``mu_0 * v0_i**2 / w(x_i)``, and for Gauss-Laguerre
``w(x) = x**s * exp(-x)``, so dividing by ``w(x_i)`` multiplies by ``exp(x_i)``.
At order 144 the largest node is ``x ~ 547``, and the first eigenvector
component there is ``v0 ~ 5e-119`` — 119 decades below the unit norm. LAPACK
returns noise (often an exact zero) in that position; squaring the noise and
multiplying by ``exp(547) ~ 1e237`` yields weights that are either 0 or ~1e126
where C++ has ``O(10)``. The quadrature then returns values like 1e72 for a
Heston call worth ~12.

The QL iteration keeps those components because it builds the first row by
*multiplying* Givens rotations into it and never forms a difference that can
cancel, so tiny entries retain full relative accuracy. That property, not
backward stability, is what this formula depends on. The failure is invisible
below order ~20 (``exp(-x_max/2)`` is still above 1e-16 there), which is why
an 8- and a 16-point test can both pass over a broken implementation.

Ordering follows from using the C++ algorithm: ``TqrEigenDecomposition`` sorts
``(eigenvalue, eigenvector)`` pairs with ``std::greater<>``, so nodes come out
descending, which is the order ``x()`` / ``weights()`` expose and the order
``__call__`` accumulates in.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

from pquantlib.math.array import Array
from pquantlib.math.integrals.gaussian_orthogonal_polynomial import (
    GaussHermitePolynomial,
    GaussHyperbolicPolynomial,
    GaussianOrthogonalPolynomial,
    GaussJacobiPolynomial,
    GaussLaguerrePolynomial,
)
from pquantlib.math.integrals.integrator import Integrator, RealFunction
from pquantlib.math.matrixutilities.tqr_eigen_decomposition import (
    EigenVectorCalculation,
    ShiftStrategy,
    TqrEigenDecomposition,
)

# C++ ``Null<Real>()`` — ql/utilities/null.hpp returns
# ``std::numeric_limits<float>::max()`` for any floating-point type. Several
# integrators in this file pass it as the absolute accuracy, which the
# ``Integrator`` base only checks against ``QL_EPSILON``.
_NULL_REAL: float = 3.4028234663852886e38


class GaussianQuadrature:
    """``n``-point Gauss quadrature for an orthogonal polynomial.

    # C++ parity: gaussianquadratures.hpp:48-77 (ctor at
    # gaussianquadratures.cpp:34-61).
    """

    __slots__ = ("_w", "_x")

    def __init__(self, n: int, orth_poly: GaussianOrthogonalPolynomial) -> None:
        # C++ parity: gaussianquadratures.cpp:34-61 — Golub-Welsch, with the
        # eigenproblem solved by TqrEigenDecomposition and NOT by LAPACK. See
        # the module docstring for why the substitution is not admissible.
        diag = np.empty(n, dtype=np.float64)
        off = np.empty(n - 1, dtype=np.float64)
        diag[0] = orth_poly.alpha(0)
        for i in range(1, n):
            diag[i] = orth_poly.alpha(i)
            off[i - 1] = math.sqrt(orth_poly.beta(i))

        tqr = TqrEigenDecomposition(
            diag,
            off,
            EigenVectorCalculation.ONLY_FIRST_ROW_EIGEN_VECTOR,
            ShiftStrategy.OVERRELAXATION,
        )
        self._x: Array = np.ascontiguousarray(tqr.eigenvalues(), dtype=np.float64)

        mu_0 = orth_poly.mu_0()
        w = np.empty(n, dtype=np.float64)
        first_row = tqr.eigenvectors()[0]
        for i in range(n):
            w[i] = mu_0 * first_row[i] * first_row[i] / orth_poly.w(float(self._x[i]))
        self._w: Array = w

    def order(self) -> int:
        # C++ parity: gaussianquadratures.hpp:71.
        return self._x.shape[0]

    def x(self) -> Array:
        # C++ parity: gaussianquadratures.hpp:73.
        return self._x

    def weights(self) -> Array:
        # C++ parity: gaussianquadratures.hpp:72.
        return self._w

    def __call__(self, f: Callable[[float], float]) -> float:
        # C++ parity: gaussianquadratures.hpp:58-65 — sum w_i f(x_i),
        # accumulated from the largest index down (matching C++ exactly).
        total = 0.0
        for i in range(self.order() - 1, -1, -1):
            total += float(self._w[i]) * float(f(float(self._x[i])))
        return total


class MultiDimGaussianIntegration:
    """Tensor-product Gauss quadrature over ``len(ns)`` dimensions.

    # C++ parity: gaussianquadratures.hpp:79-93, gaussianquadratures.cpp:64-104.
    """

    __slots__ = ("_weights", "_x")

    def __init__(
        self, ns: Sequence[int], gen_quad: Callable[[int], GaussianQuadrature]
    ) -> None:
        # C++ parity: gaussianquadratures.cpp:64-94.
        total = 1
        for e in ns:
            total *= e
        weights = np.ones(total, dtype=np.float64)
        m = len(ns)
        n = total
        x: list[Array] = [np.zeros(m, dtype=np.float64) for _ in range(n)]

        # C++ std::partial_sum(ns.begin(), ns.end()-1, spacing.begin()+1, *)
        spacing = [1] * m
        for j in range(1, m):
            spacing[j] = spacing[j - 1] * ns[j - 1]

        n2weights: dict[int, Array] = {}
        n2x: dict[int, Array] = {}
        for quad_order in ns:
            if quad_order not in n2x:
                quad = gen_quad(quad_order)
                n2x[quad_order] = quad.x()
                n2weights[quad_order] = quad.weights()

        for i in range(n):
            for j in range(m):
                quad_order = ns[j]
                nx = (i // spacing[j]) % ns[j]
                weights[i] *= float(n2weights[quad_order][nx])
                x[i][j] = n2x[quad_order][nx]

        self._weights: Array = weights
        self._x: list[Array] = x

    def weights(self) -> Array:
        # C++ parity: gaussianquadratures.hpp:87.
        return self._weights

    def x(self) -> list[Array]:
        # C++ parity: gaussianquadratures.hpp:88.
        return self._x

    def __call__(self, f: Callable[[Array], float]) -> float:
        # C++ parity: gaussianquadratures.cpp:96-104.
        s = 0.0
        for i in range(len(self._x)):
            s += float(self._weights[i]) * float(f(self._x[i]))
        return s


class GaussLaguerreIntegration(GaussianQuadrature):
    """``n``-point generalized Gauss-Laguerre rule for ``int_0^inf f(x) dx``.

    # C++ parity: gaussianquadratures.hpp:107-111.
    """

    def __init__(self, n: int, s: float = 0.0) -> None:
        super().__init__(n, GaussLaguerrePolynomial(s))


class GaussHermiteIntegration(GaussianQuadrature):
    """``n``-point generalized Gauss-Hermite rule for ``int_-inf^inf f(x) dx``.

    # C++ parity: gaussianquadratures.hpp:124-128.
    """

    def __init__(self, n: int, mu: float = 0.0) -> None:
        super().__init__(n, GaussHermitePolynomial(mu))


class GaussJacobiIntegration(GaussianQuadrature):
    """``n``-point Gauss-Jacobi rule for ``int_-1^1 f(x) dx``.

    # C++ parity: gaussianquadratures.hpp:140-144.
    """

    def __init__(self, n: int, alpha: float, beta: float) -> None:
        super().__init__(n, GaussJacobiPolynomial(alpha, beta))


class GaussHyperbolicIntegration(GaussianQuadrature):
    """``n``-point Gauss-hyperbolic rule for ``int_-inf^inf f(x) dx``.

    # C++ parity: gaussianquadratures.hpp:156-160.
    """

    def __init__(self, n: int) -> None:
        super().__init__(n, GaussHyperbolicPolynomial())


class GaussLegendreIntegration(GaussianQuadrature):
    """``n``-point Gauss-Legendre rule for ``int_-1^1 f(x) dx``.

    # C++ parity: gaussianquadratures.hpp:172-176 — built from
    # ``GaussJacobiPolynomial(0, 0)``, not from ``GaussLegendrePolynomial``.
    """

    def __init__(self, n: int) -> None:
        super().__init__(n, GaussJacobiPolynomial(0.0, 0.0))


class GaussChebyshevIntegration(GaussianQuadrature):
    """``n``-point Gauss-Chebyshev rule for ``int_-1^1 f(x) dx``.

    # C++ parity: gaussianquadratures.hpp:188-192.
    """

    def __init__(self, n: int) -> None:
        super().__init__(n, GaussJacobiPolynomial(-0.5, -0.5))


class GaussChebyshev2ndIntegration(GaussianQuadrature):
    """``n``-point Gauss-Chebyshev (2nd kind) rule for ``int_-1^1 f(x) dx``.

    # C++ parity: gaussianquadratures.hpp:204-208.
    """

    def __init__(self, n: int) -> None:
        super().__init__(n, GaussJacobiPolynomial(0.5, 0.5))


class GaussGegenbauerIntegration(GaussianQuadrature):
    """``n``-point Gauss-Gegenbauer rule for ``int_-1^1 f(x) dx``.

    # C++ parity: gaussianquadratures.hpp:220-225.
    """

    def __init__(self, n: int, lambda_: float) -> None:
        super().__init__(n, GaussJacobiPolynomial(lambda_ - 0.5, lambda_ - 0.5))


class GaussianQuadratureIntegrator(Integrator):
    """Adapt a fixed Gauss rule on ``[-1, 1]`` to an :class:`Integrator`.

    # C++ parity: ``detail::GaussianQuadratureIntegrator<Integration>``
    # (gaussianquadratures.hpp:228-243, gaussianquadratures.cpp:108-130). The
    # C++ template parameter selects the rule at compile time; the Python port
    # takes the rule factory as a constructor argument and the three C++
    # ``typedef``\\ s become the subclasses below.
    """

    __slots__ = ("_integration",)

    def __init__(
        self, n: int, integration: Callable[[int], GaussianQuadrature]
    ) -> None:
        # C++ parity: Integrator(Null<Real>(), n).
        super().__init__(_NULL_REAL, n)
        self._integration: GaussianQuadrature = integration(n)

    def get_integration(self) -> GaussianQuadrature:
        # C++ parity: gaussianquadratures.hpp:234.
        return self._integration

    def _integrate(self, f: RealFunction, a: float, b: float) -> float:
        # C++ parity: gaussianquadratures.cpp:116-125.
        c1 = 0.5 * (b - a)
        c2 = 0.5 * (a + b)
        return c1 * self._integration(lambda x: f(c1 * x + c2))


class GaussLegendreIntegrator(GaussianQuadratureIntegrator):
    """# C++ parity: ``typedef ... GaussLegendreIntegrator`` (hpp:245-246)."""

    def __init__(self, n: int) -> None:
        super().__init__(n, GaussLegendreIntegration)


class GaussChebyshevIntegrator(GaussianQuadratureIntegrator):
    """# C++ parity: ``typedef ... GaussChebyshevIntegrator`` (hpp:248-249)."""

    def __init__(self, n: int) -> None:
        super().__init__(n, GaussChebyshevIntegration)


class GaussChebyshev2ndIntegrator(GaussianQuadratureIntegrator):
    """# C++ parity: ``typedef ... GaussChebyshev2ndIntegrator`` (hpp:251-252)."""

    def __init__(self, n: int) -> None:
        super().__init__(n, GaussChebyshev2ndIntegration)


__all__ = [
    "GaussChebyshev2ndIntegration",
    "GaussChebyshev2ndIntegrator",
    "GaussChebyshevIntegration",
    "GaussChebyshevIntegrator",
    "GaussGegenbauerIntegration",
    "GaussHermiteIntegration",
    "GaussHermitePolynomial",
    "GaussHyperbolicIntegration",
    "GaussJacobiIntegration",
    "GaussLaguerreIntegration",
    "GaussLegendreIntegration",
    "GaussLegendreIntegrator",
    "GaussianOrthogonalPolynomial",
    "GaussianQuadrature",
    "GaussianQuadratureIntegrator",
    "MultiDimGaussianIntegration",
]
