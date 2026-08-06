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

Eigen-decomposition — QuantLib's own TQR, not scipy
--------------------------------------------------

The C++ class runs ``TqrEigenDecomposition`` (implicit-shift QL with the
"over-relaxation" Wilkinson shift, tracking only the first eigenvector row) on
the Jacobi matrix. This module used to delegate that to
``scipy.linalg.eigh_tridiagonal`` on the argument that "both are
backward-stable solvers for the same matrix". They are — but backward
stability is not the property the Golub-Welsch weight needs.

The weight is ``mu_0 * ev[0][i]^2 / w(x_i)``. For Laguerre that divides by
``exp(-x_i)``, and at ``n = 128`` the largest node is ``x ~ 484``, so the first
eigenvector component has to be accurate down to about ``1e-105``. LAPACK
delivers eigenvectors to *absolute* accuracy ``eps * ||T||``: those components
are noise. QuantLib's TQR accumulates the Givens rotations into the first row
alone, starting from the identity, and preserves their relative accuracy.

The port therefore uses the ported ``TqrEigenDecomposition``, which also fixes
the ordering for free (it sorts descending, as C++ does).
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
        """Golub-Welsch, via QuantLib's own TQR — not ``scipy.eigh_tridiagonal``.

        # C++ parity: gaussianquadratures.cpp:34-61.

        **Divergence fixed here.** This constructor used to call
        ``scipy.linalg.eigh_tridiagonal``. That is not C++'s
        ``TqrEigenDecomposition(..., OnlyFirstRowEigenVector, Overrelaxation)``,
        and the difference is not cosmetic. The Golub-Welsch weight is
        ``mu_0 * ev[0][i]^2 / w(x_i)``; for Laguerre that divides by
        ``exp(-x_i)``, so at ``n = 128`` the largest node is ``x ~ 484`` and the
        first eigenvector component must be accurate down to ``1e-105``. LAPACK
        computes eigenvectors to *absolute* accuracy ``eps * ||T||``, so those
        components are pure noise; QuantLib's TQR accumulates the Givens
        rotations into the first row only, starting from the identity, and keeps
        their *relative* accuracy.

        Measured at ``GaussLaguerreIntegration(128)``: C++ weights span
        ``[0.0289, 25.26]`` and sum to 498.1; the scipy-based version produced
        weights up to ``2.8e109`` summing to ``2.8e109``. The error was invisible
        for integrands that decay like the weight function — ``int exp(-x)``
        returned 1.0 either way — and catastrophic otherwise:
        ``int 1/(1+x^2)`` returned ``1.85e104`` against the true ``pi/2``.
        It was found because ``HestonProcess.pdf`` integrates a
        polynomially-decaying characteristic function with exactly this rule.
        """
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
        first_row = np.ascontiguousarray(tqr.eigenvectors()[0, :], dtype=np.float64)

        mu_0 = orth_poly.mu_0()
        w = np.empty(n, dtype=np.float64)
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
