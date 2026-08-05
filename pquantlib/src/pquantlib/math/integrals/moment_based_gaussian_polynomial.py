"""Gaussian orthogonal polynomial defined by the moments of a distribution.

# C++ parity: ql/math/integrals/momentbasedgaussianpolynomial.hpp (v1.43).

Subclasses supply :meth:`MomentBasedGaussianPolynomial.moment` (the ``i``-th
raw moment) and the weight ``w(x)``; the base derives the three-term
recurrence coefficients ``alpha`` / ``beta`` lazily through the modified-moment
determinant recursion ``z(k, i)``.

References:

* G.H. Golub and J.H. Welsch, "Calculation of Gauss quadrature rules",
  Math. Comput. 23 (1969), 221-230.
* M. Morandi Cecchi and M. Redivo Zaglia, "Computing the coefficients of a
  recurrence formula for numerical integration by moments and modified
  moments", J. Comput. Appl. Math. 49 (1993).

# C++ parity divergence: the C++ class is a template ``<mp_real>`` so the
# recursion can run in arbitrary precision (boost.multiprecision) to fight the
# cancellation that the determinant recursion suffers for large ``n``. This is
# the ``mp_real == Real`` specialization only; arbitrary precision would need
# ``mpmath`` and is not ported.
"""

from __future__ import annotations

from abc import abstractmethod
from math import isnan, nan

from pquantlib import qassert
from pquantlib.math.closeness import close_enough
from pquantlib.math.integrals.gaussian_orthogonal_polynomial import (
    OrthogonalPolynomialBase,
)


class MomentBasedGaussianPolynomial(OrthogonalPolynomialBase):
    """Orthogonal polynomial reconstructed from raw moments.

    # C++ parity: momentbasedgaussianpolynomial.hpp:48-68.
    """

    __slots__ = ("_b", "_c", "_z_tab")

    def __init__(self) -> None:
        # C++ parity: z_(1, std::vector<mp_real>()).
        self._z_tab: list[list[float]] = [[]]
        self._b: list[float] = []
        self._c: list[float] = []

    @abstractmethod
    def moment(self, i: int) -> float:
        """The ``i``-th raw moment of the distribution."""

    def mu_0(self) -> float:
        # C++ parity: momentbasedgaussianpolynomial.hpp:157-162 (Real
        # specialization) — the zeroth moment must be normalised to one.
        m0 = self.moment(0)
        qassert.require(close_enough(m0, 1.0), "zero moment must by one.")
        return self.moment(0)

    def alpha(self, u: int) -> float:
        # C++ parity: momentbasedgaussianpolynomial.hpp:139-142.
        return self._alpha(u)

    def beta(self, u: int) -> float:
        # C++ parity: momentbasedgaussianpolynomial.hpp:150-153.
        return self._beta(u)

    # ---- internal modified-moment recursion ---------------------------

    def _z(self, k: int, i: int) -> float:
        # C++ parity: momentbasedgaussianpolynomial.hpp:76-101 (``z``).
        if k == -1:
            return 0.0
        rows = len(self._z_tab)
        cols = len(self._z_tab[0])
        if cols <= i:
            for row in self._z_tab:
                row.extend([nan] * (i + 1 - len(row)))
        if rows <= k:
            width = len(self._z_tab[0])
            for _ in range(k + 1 - rows):
                self._z_tab.append([nan] * width)
        if isnan(self._z_tab[k][i]):
            if k == 0:
                self._z_tab[k][i] = self.moment(i)
            else:
                self._z_tab[k][i] = (
                    self._z(k - 1, i + 1)
                    - self._alpha(k - 1) * self._z(k - 1, i)
                    - self._beta(k - 1) * self._z(k - 2, i)
                )
        return self._z_tab[k][i]

    def _alpha(self, u: int) -> float:
        # C++ parity: momentbasedgaussianpolynomial.hpp:103-120 (``alpha_``).
        if len(self._b) <= u:
            self._b.extend([nan] * (u + 1 - len(self._b)))
        if isnan(self._b[u]):
            if u == 0:
                self._b[u] = self.moment(1)
            else:
                self._b[u] = -self._z(u - 1, u) / self._z(u - 1, u - 1) + self._z(
                    u, u + 1
                ) / self._z(u, u)
        return self._b[u]

    def _beta(self, u: int) -> float:
        # C++ parity: momentbasedgaussianpolynomial.hpp:122-135 (``beta_``).
        if u == 0:
            return 1.0
        if len(self._c) <= u:
            self._c.extend([nan] * (u + 1 - len(self._c)))
        if isnan(self._c[u]):
            self._c[u] = self._z(u, u) / self._z(u - 1, u - 1)
        return self._c[u]


__all__ = ["MomentBasedGaussianPolynomial"]
