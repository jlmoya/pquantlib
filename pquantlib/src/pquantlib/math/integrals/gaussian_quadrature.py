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

Eigen-decomposition delegation
------------------------------

The C++ class runs its own ``TqrEigenDecomposition`` (implicit-shift QL with
the "over-relaxation" Wilkinson shift) on the Jacobi matrix; the Python port
delegates the symmetric-tridiagonal eigenproblem to
``scipy.linalg.eigh_tridiagonal``. That is a genuine linear-algebra
delegation, not an algorithm substitution: both are backward-stable solvers
for the *same* matrix, and the node/weight arrays are cross-validated
element-by-element against the C++ probe.

What is **not** free is the ordering. ``TqrEigenDecomposition`` sorts
``(eigenvalue, eigenvector)`` pairs with ``std::greater<>`` — nodes come out
**descending** — whereas LAPACK returns them ascending. ``x()``/``weights()``
are public API and ``operator()`` accumulates from the last index down, so the
port reverses LAPACK's output to restore the C++ order.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from scipy.linalg import (  # pyright: ignore[reportMissingTypeStubs]
    eigh_tridiagonal,  # pyright: ignore[reportUnknownVariableType]
)

from pquantlib.math.array import Array
from pquantlib.math.integrals.gaussian_orthogonal_polynomial import (
    GaussHermitePolynomial,
    GaussHyperbolicPolynomial,
    GaussianOrthogonalPolynomial,
    GaussJacobiPolynomial,
    GaussLaguerrePolynomial,
)
from pquantlib.math.integrals.integrator import Integrator, RealFunction

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
        # C++ parity: gaussianquadratures.cpp:34-61 — Golub-Welsch.
        diag = np.empty(n, dtype=np.float64)
        off = np.empty(n - 1, dtype=np.float64)
        diag[0] = orth_poly.alpha(0)
        for i in range(1, n):
            diag[i] = orth_poly.alpha(i)
            off[i - 1] = math.sqrt(orth_poly.beta(i))

        # scipy.linalg.eigh_tridiagonal is untyped; the ndarray results are
        # immediately normalised into owned float64 arrays.
        eigenvalues, eigenvectors = eigh_tridiagonal(diag, off)  # pyright: ignore[reportUnknownVariableType]
        # C++ parity: tqreigendecomposition.cpp:124 sorts the (eigenvalue,
        # eigenvector) pairs with std::greater<> — descending. LAPACK hands
        # them back ascending, so undo that here; the sign normalisation the
        # C++ applies afterwards is irrelevant because the weight squares the
        # eigenvector component.
        order = np.argsort(eigenvalues, kind="stable")[::-1]  # pyright: ignore[reportUnknownArgumentType]
        self._x: Array = np.ascontiguousarray(eigenvalues[order], dtype=np.float64)  # pyright: ignore[reportUnknownArgumentType]

        mu_0 = orth_poly.mu_0()
        w = np.empty(n, dtype=np.float64)
        first_row = np.ascontiguousarray(eigenvectors[0, :][order], dtype=np.float64)  # pyright: ignore[reportUnknownArgumentType]
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
