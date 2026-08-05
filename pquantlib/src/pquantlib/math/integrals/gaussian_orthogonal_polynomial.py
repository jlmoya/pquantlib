"""Orthogonal polynomials for the Gauss quadratures.

# C++ parity: ql/math/integrals/gaussianorthogonalpolynomial.{hpp,cpp} (v1.43).

Every polynomial here is defined by the three-term recurrence

    P_{k+1}(x) = (x - alpha_k) P_k(x) - beta_k P_{k-1}(x),   mu_0 = int w(x) dx

which is all :class:`~pquantlib.math.integrals.gaussian_quadrature.GaussianQuadrature`
needs in order to build the nodes and weights (Golub-Welsch).

References:

* G.H. Golub and J.H. Welsch, "Calculation of Gauss quadrature rules",
  Math. Comput. 23 (1969), 221-230.
* "Numerical Recipes in C", 2nd edition, Press/Teukolsky/Vetterling/Flannery.

The C++ abstract base ``GaussianOrthogonalPolynomial`` carries two concrete
methods (``value`` / ``weightedValue``) on top of the four pure-virtual ones.
The Python port splits that in two:

* :class:`GaussianOrthogonalPolynomial` stays a ``Protocol`` — the *structural*
  contract that ``GaussianQuadrature`` consumes, so that moment-based
  polynomials which are not in this class hierarchy still satisfy it;
* :class:`OrthogonalPolynomialBase` is the concrete implementation base that
  supplies ``value`` / ``weighted_value``. It has no C++ counterpart of its
  own — it exists only because a ``Protocol`` cannot both stay structural and
  hand out inherited implementations.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Protocol

from pquantlib import qassert
from pquantlib.math.closeness import close_enough
from pquantlib.math.constants import M_PI, M_PI_2
from pquantlib.math.distributions.gamma_function import GammaFunction


class GaussianOrthogonalPolynomial(Protocol):
    """Structural interface for a Gaussian orthogonal polynomial.

    # C++ parity: ``class GaussianOrthogonalPolynomial`` — the pure-virtual
    # part of the abstract base in gaussianorthogonalpolynomial.hpp:50-60.
    """

    def alpha(self, u: int) -> float: ...
    def beta(self, u: int) -> float: ...
    def mu_0(self) -> float: ...
    def w(self, x: float) -> float: ...


class OrthogonalPolynomialBase(ABC):
    """Concrete base carrying ``value`` / ``weighted_value``.

    # C++ parity: the non-virtual half of ``GaussianOrthogonalPolynomial``
    # (gaussianorthogonalpolynomial.cpp:32-46).
    """

    __slots__ = ()

    @abstractmethod
    def mu_0(self) -> float:
        """Zeroth moment ``int w(x) dx``."""

    @abstractmethod
    def alpha(self, u: int) -> float:
        """Recurrence coefficient ``alpha_u``."""

    @abstractmethod
    def beta(self, u: int) -> float:
        """Recurrence coefficient ``beta_u``."""

    @abstractmethod
    def w(self, x: float) -> float:
        """Weight function ``w(x)``."""

    def value(self, n: int, x: float) -> float:
        """``P_n(x)`` from the three-term recurrence.

        # C++ parity: gaussianorthogonalpolynomial.cpp:32-42. The C++ body is
        # doubly recursive (it does not memoise); kept as-is so the evaluation
        # order — and therefore the rounding — matches.
        """
        if n > 1:
            return (x - self.alpha(n - 1)) * self.value(n - 1, x) - self.beta(n - 1) * self.value(
                n - 2, x
            )
        if n == 1:
            return x - self.alpha(0)
        return 1.0

    def weighted_value(self, n: int, x: float) -> float:
        """``sqrt(w(x)) * P_n(x)``.

        # C++ parity: gaussianorthogonalpolynomial.cpp:44-46 (``weightedValue``).
        """
        return math.sqrt(self.w(x)) * self.value(n, x)


class GaussLaguerrePolynomial(OrthogonalPolynomialBase):
    """Gauss-Laguerre polynomial, weight ``x^s exp(-x)`` on ``[0, inf)``.

    # C++ parity: gaussianorthogonalpolynomial.cpp:49-68.
    """

    __slots__ = ("_s",)

    def __init__(self, s: float = 0.0) -> None:
        qassert.require(s > -1.0, "s must be bigger than -1")
        self._s: float = s

    def mu_0(self) -> float:
        return math.exp(GammaFunction.log_value(self._s + 1))

    def alpha(self, u: int) -> float:
        return 2 * u + 1 + self._s

    def beta(self, u: int) -> float:
        return u * (u + self._s)

    def w(self, x: float) -> float:
        return math.pow(x, self._s) * math.exp(-x)


class GaussHermitePolynomial(OrthogonalPolynomialBase):
    """Gauss-Hermite polynomial, weight ``|x|^{2 mu} exp(-x^2)`` on ``R``.

    # C++ parity: gaussianorthogonalpolynomial.cpp:71-90.
    """

    __slots__ = ("_mu",)

    def __init__(self, mu: float = 0.0) -> None:
        qassert.require(mu > -0.5, "mu must be bigger than -0.5")
        self._mu: float = mu

    def mu_0(self) -> float:
        return math.exp(GammaFunction.log_value(self._mu + 0.5))

    def alpha(self, u: int) -> float:
        del u
        return 0.0

    def beta(self, u: int) -> float:
        # C++ parity: (i % 2) != 0U ? i/2.0 + mu_ : i/2.0.
        if u % 2 != 0:
            return u / 2.0 + self._mu
        return u / 2.0

    def w(self, x: float) -> float:
        return math.pow(abs(x), 2 * self._mu) * math.exp(-x * x)


class GaussJacobiPolynomial(OrthogonalPolynomialBase):
    """Gauss-Jacobi polynomial, weight ``(1-x)^a (1+x)^b`` on ``(-1, 1)``.

    # C++ parity: gaussianorthogonalpolynomial.cpp:92-147.
    """

    __slots__ = ("_alpha", "_beta")

    def __init__(self, alpha: float, beta: float) -> None:
        qassert.require(alpha + beta > -2.0, "alpha+beta must be bigger than -2")
        qassert.require(alpha > -1.0, "alpha must be bigger than -1")
        qassert.require(beta > -1.0, "beta  must be bigger than -1")
        self._alpha: float = alpha
        self._beta: float = beta

    def mu_0(self) -> float:
        return math.pow(2.0, self._alpha + self._beta + 1) * math.exp(
            GammaFunction.log_value(self._alpha + 1)
            + GammaFunction.log_value(self._beta + 1)
            - GammaFunction.log_value(self._alpha + self._beta + 2)
        )

    def alpha(self, u: int) -> float:
        # C++ parity: gaussianorthogonalpolynomial.cpp:106-124, including the
        # l'Hospital fallback and the assertion that fires when even that
        # denominator vanishes (alpha_ + beta_ == 0 at u == 0).
        num = self._beta * self._beta - self._alpha * self._alpha
        denom = (2.0 * u + self._alpha + self._beta) * (2.0 * u + self._alpha + self._beta + 2)

        if close_enough(denom, 0.0):
            if not close_enough(num, 0.0):
                qassert.fail("can't compute a_k for jacobi integration\n")
            # l'Hospital
            num = 2 * self._beta
            denom = 2 * (2.0 * u + self._alpha + self._beta + 1)
            qassert.require(
                not close_enough(denom, 0.0), "can't compute a_k for jacobi integration\n"
            )

        return num / denom

    def beta(self, u: int) -> float:
        # C++ parity: gaussianorthogonalpolynomial.cpp:126-143.
        num = 4.0 * u * (u + self._alpha) * (u + self._beta) * (u + self._alpha + self._beta)
        denom = (
            (2.0 * u + self._alpha + self._beta)
            * (2.0 * u + self._alpha + self._beta)
            * (
                (2.0 * u + self._alpha + self._beta) * (2.0 * u + self._alpha + self._beta)
                - 1
            )
        )

        if close_enough(denom, 0.0):
            if not close_enough(num, 0.0):
                qassert.fail("can't compute b_k for jacobi integration\n")
            # l'Hospital
            num = 4.0 * u * (u + self._beta) * (2.0 * u + 2 * self._alpha + self._beta)
            denom = 2.0 * (2.0 * u + self._alpha + self._beta)
            denom *= denom - 1
            qassert.require(
                not close_enough(denom, 0.0), "can't compute b_k for jacobi integration\n"
            )
        return num / denom

    def w(self, x: float) -> float:
        return math.pow(1 - x, self._alpha) * math.pow(1 + x, self._beta)


class GaussLegendrePolynomial(GaussJacobiPolynomial):
    """Gauss-Legendre polynomial — Jacobi(0, 0).

    # C++ parity: gaussianorthogonalpolynomial.cpp:150-152.
    """

    def __init__(self) -> None:
        super().__init__(0.0, 0.0)


class GaussChebyshev2ndPolynomial(GaussJacobiPolynomial):
    """Gauss-Chebyshev polynomial of the second kind — Jacobi(0.5, 0.5).

    # C++ parity: gaussianorthogonalpolynomial.cpp:154-156.
    """

    def __init__(self) -> None:
        super().__init__(0.5, 0.5)


class GaussChebyshevPolynomial(GaussJacobiPolynomial):
    """Gauss-Chebyshev polynomial — Jacobi(-0.5, -0.5).

    # C++ parity: gaussianorthogonalpolynomial.cpp:158-160.
    """

    def __init__(self) -> None:
        super().__init__(-0.5, -0.5)


class GaussGegenbauerPolynomial(GaussJacobiPolynomial):
    """Gauss-Gegenbauer polynomial — Jacobi(lambda-0.5, lambda-0.5).

    # C++ parity: gaussianorthogonalpolynomial.cpp:162-164.
    """

    def __init__(self, lambda_: float) -> None:
        super().__init__(lambda_ - 0.5, lambda_ - 0.5)


class GaussHyperbolicPolynomial(OrthogonalPolynomialBase):
    """Gauss-hyperbolic polynomial, weight ``1/cosh(x)`` on ``R``.

    # C++ parity: gaussianorthogonalpolynomial.cpp:166-180.
    """

    __slots__ = ()

    def mu_0(self) -> float:
        return M_PI

    def alpha(self, u: int) -> float:
        del u
        return 0.0

    def beta(self, u: int) -> float:
        # C++ parity: i != 0U ? M_PI_2*M_PI_2*i*i : M_PI.
        if u != 0:
            return M_PI_2 * M_PI_2 * u * u
        return M_PI

    def w(self, x: float) -> float:
        return 1 / math.cosh(x)


__all__ = [
    "GaussChebyshev2ndPolynomial",
    "GaussChebyshevPolynomial",
    "GaussGegenbauerPolynomial",
    "GaussHermitePolynomial",
    "GaussHyperbolicPolynomial",
    "GaussJacobiPolynomial",
    "GaussLaguerrePolynomial",
    "GaussLegendrePolynomial",
    "GaussianOrthogonalPolynomial",
    "OrthogonalPolynomialBase",
]
