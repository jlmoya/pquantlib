"""LsmBasisSystem — polynomial basis for Longstaff-Schwartz regression.

# C++ parity: ql/methods/montecarlo/lsmbasissystem.{hpp,cpp} (v1.42.1).

C++ ships seven polynomial families via a ``PolynomialType`` enum and
delegates evaluation to ``GaussianOrthogonalPolynomial::weightedValue
(n, x) = sqrt(w(x)) * value(n, x)`` for the non-monomial cases —
where ``w(x)`` is the orthogonality weight and ``value(n, x)`` is the
three-term recurrence ``value(n, x) = (x - alpha(n-1)) * value(n-1, x)
- beta(n-1) * value(n-2, x)`` rooted at ``value(0) = 1`` and
``value(1) = x - alpha(0)``.

The C++ ``AmericanPathPricer`` constructor restricts the allowed set
to ``Monomial / Laguerre / Hermite / Hyperbolic / Chebyshev2nd``. The
Python port:

* Implements ``Monomial / Laguerre / Hermite / Chebyshev2nd`` (the
  four allowed types whose orthogonality weights have closed-form
  expressions in stdlib).
* Defers ``Hyperbolic`` (Gauss-Hyperbolic recurrence with custom
  ``w(x) = 1 / cosh(x)``) and ``Legendre / Chebyshev`` (1st-kind) —
  documented as Phase 6 carve-outs in ``phase6-l6-A-design.md``.

The recurrence coefficients ``alpha(i)`` / ``beta(i)`` and the
weights ``w(x)`` are ported directly from the C++ implementations in
``ql/math/integrals/gaussianorthogonalpolynomial.cpp``:

* **Laguerre (s=0):**  ``alpha(i) = 2i+1``, ``beta(i) = i^2``,
  ``w(x) = exp(-x)``.
* **Hermite (mu=0):** ``alpha(i) = 0``, ``beta(i) = i // 2`` (rounding
  via the C++ ``i % 2 ? i/2.0 + mu : i/2.0`` formula with mu=0),
  ``w(x) = exp(-x^2)``.
* **Chebyshev2nd (Gauss-Jacobi(alpha=0.5, beta=0.5)):**
  ``w(x) = (1-x)^0.5 (1+x)^0.5 = sqrt(1-x^2)``,
  recurrence per Gauss-Jacobi alpha(i) / beta(i) formulas with
  ``alpha=beta=0.5`` (so the symmetric branch: alpha(i)=0,
  beta(i) per Jacobi formula). See the C++ code for the explicit
  expressions.

The ``LsmBasisSystem`` class exposes the same two static-factory
methods as the C++ original:

* ``LsmBasisSystem.path_basis_system(order, polynomial_type)`` returns
  ``order+1`` callables ``Real -> Real`` (basis functions of orders
  0 through ``order``).
* ``LsmBasisSystem.multi_path_basis_system(dim, order, polynomial_type)``
  returns a list of ``ndarray -> Real`` callables (products of single-
  variable basis functions, indexed by multi-orders summing to at
  most ``order``).

The multi-dimensional construction mirrors the C++ tuple-generation
loop (``next_order_tuples``) bit-for-bit: at each order ``i`` we
generate the multi-orders ``(o_1, ..., o_dim)`` with
``sum(o_k) == i`` and emit ``product(basis[o_k](x_k) for k)``.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum

import numpy as np
import numpy.typing as npt

from pquantlib import qassert


class PolynomialType(IntEnum):
    """Polynomial family for the LSM regression basis.

    # C++ parity: ``LsmBasisSystem::PolynomialType`` (lsmbasissystem.hpp:39-42).
    """

    Monomial = 0
    Laguerre = 1
    Hermite = 2
    Hyperbolic = 3
    Legendre = 4
    Chebyshev = 5
    Chebyshev2nd = 6


# --- recurrence coefficients for the four supported families -----------------
# Mirrors GaussianOrthogonalPolynomial::value(n, x) form:
#   value(n, x) = (x - alpha(n-1)) * value(n-1, x)  -  beta(n-1) * value(n-2, x)
# rooted at value(0) = 1, value(1) = x - alpha(0).
# weightedValue(n, x) = sqrt(w(x)) * value(n, x).


def _laguerre_alpha(i: int) -> float:
    # C++ parity: GaussLaguerrePolynomial::alpha (with s_=0).
    return 2.0 * i + 1.0


def _laguerre_beta(i: int) -> float:
    # C++ parity: GaussLaguerrePolynomial::beta (with s_=0).
    return float(i * i)


def _laguerre_w(x: float) -> float:
    # C++ parity: GaussLaguerrePolynomial::w (with s_=0): x^s * exp(-x) = exp(-x).
    return float(np.exp(-x))


def _hermite_alpha(i: int) -> float:
    # C++ parity: GaussHermitePolynomial::alpha (with mu_=0): always 0.
    _ = i  # unused but kept for signature parity
    return 0.0


def _hermite_beta(i: int) -> float:
    # C++ parity: GaussHermitePolynomial::beta (with mu_=0):
    #   (i % 2) != 0 ? i/2.0 + mu : i/2.0
    # NOTE: C++ ``i / 2.0`` is floating-point division (e.g. 1/2.0 = 0.5),
    # not integer truncation. So for mu=0 the formula reduces to
    # ``i / 2.0`` for all i. Using float division to mirror.
    return i / 2.0


def _hermite_w(x: float) -> float:
    # C++ parity: GaussHermitePolynomial::w (with mu_=0):
    #   |x|^(2*mu) * exp(-x^2) = exp(-x^2).
    return float(np.exp(-x * x))


def _jacobi_alpha(i: int, a: float, b: float) -> float:
    # C++ parity: GaussJacobiPolynomial::alpha.
    num = b * b - a * a
    denom = (2.0 * i + a + b) * (2.0 * i + a + b + 2.0)
    if abs(denom) < 1e-16:
        # l'Hopital branch (C++ check)
        # Should not trip for the cases we care about.
        num = 2.0 * b
        denom = 2.0 * (2.0 * i + a + b + 1.0)
    return num / denom


def _jacobi_beta(i: int, a: float, b: float) -> float:
    # C++ parity: GaussJacobiPolynomial::beta.
    num = 4.0 * i * (i + a) * (i + b) * (i + a + b)
    s = 2.0 * i + a + b
    denom = (s * s) * (s * s - 1.0)
    if abs(denom) < 1e-16:
        # l'Hopital branch (mirrors C++ default arg sequence).
        num = 4.0 * i * (i + b) * (2.0 * i + 2.0 * a + b)
        denom = 2.0 * (2.0 * i + a + b)
        denom = denom * (denom - 1.0)
    return num / denom


def _chebyshev2nd_alpha(i: int) -> float:
    # GaussChebyshev2ndPolynomial is GaussJacobiPolynomial(0.5, 0.5).
    return _jacobi_alpha(i, 0.5, 0.5)


def _chebyshev2nd_beta(i: int) -> float:
    return _jacobi_beta(i, 0.5, 0.5)


def _chebyshev2nd_w(x: float) -> float:
    # C++ parity: GaussJacobiPolynomial::w with alpha=0.5, beta=0.5:
    #   (1-x)^0.5 * (1+x)^0.5 = sqrt(1 - x^2).
    return float(np.sqrt(max(1.0 - x * x, 0.0)))


def _value(n: int, x: float, alpha_fn: Callable[[int], float], beta_fn: Callable[[int], float]) -> float:
    """Three-term recurrence value of the orthogonal polynomial at ``x``.

    # C++ parity: ``GaussianOrthogonalPolynomial::value(n, x)``
    # (gaussianorthogonalpolynomial.cpp:32-42).
    """
    if n == 0:
        return 1.0
    if n == 1:
        return x - alpha_fn(0)
    prev2 = 1.0
    prev1 = x - alpha_fn(0)
    for k in range(1, n):
        cur = (x - alpha_fn(k)) * prev1 - beta_fn(k) * prev2
        prev2 = prev1
        prev1 = cur
    return prev1


def _weighted_value_factory(
    polynomial_type: PolynomialType,
) -> Callable[[int, float], float]:
    """Return ``f(n, x) = weightedValue(n, x)`` for the given family.

    # C++ parity: ``GaussianOrthogonalPolynomial::weightedValue(n, x)``
    # (gaussianorthogonalpolynomial.cpp:44-46).
    """
    if polynomial_type == PolynomialType.Laguerre:
        a, b, w = _laguerre_alpha, _laguerre_beta, _laguerre_w
    elif polynomial_type == PolynomialType.Hermite:
        a, b, w = _hermite_alpha, _hermite_beta, _hermite_w
    elif polynomial_type == PolynomialType.Chebyshev2nd:
        a, b, w = _chebyshev2nd_alpha, _chebyshev2nd_beta, _chebyshev2nd_w
    else:
        raise NotImplementedError(
            f"weighted value not implemented for {polynomial_type}"
        )

    def fn(n: int, x: float) -> float:
        return float(np.sqrt(w(x))) * _value(n, x, a, b)

    return fn


def _make_monomial(order: int) -> Callable[[float], float]:
    """Return ``x -> x^order``."""

    def fn(x: float) -> float:
        # C++ parity: MonomialFct::operator() — manual power loop.
        result = 1.0
        for _ in range(order):
            result *= x
        return result

    return fn


def _make_weighted(
    polynomial_type: PolynomialType, order: int
) -> Callable[[float], float]:
    """Return ``x -> weightedValue(order, x)`` for one polynomial family."""
    wv = _weighted_value_factory(polynomial_type)

    def fn(x: float) -> float:
        return wv(order, x)

    return fn


def _next_order_tuples(tuples: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """Generate all unique order-(N+1) tuples from order-N tuples.

    # C++ parity: anonymous ``next_order_tuples`` in lsmbasissystem.cpp:83-103.

    Inputs are tuples of nonneg ints all summing to N; outputs sum to N+1.
    Implemented by lifting each tuple +1 on each axis and de-duplicating
    via a ``set``. The set returns tuples in arbitrary order, but the C++
    version stores in ``std::set<std::vector<Size>>`` which is
    lex-sorted; we sort to match that ordering deterministically.
    """
    qassert.require(len(tuples) > 0, "empty tuples")
    dim = len(tuples[0])
    seen: set[tuple[int, ...]] = set()
    for k in range(dim):
        for t in tuples:
            lifted = list(t)
            lifted[k] += 1
            seen.add(tuple(lifted))
    return sorted(seen)


class LsmBasisSystem:
    """Polynomial basis system for Longstaff-Schwartz regression.

    # C++ parity: ``class LsmBasisSystem`` (lsmbasissystem.hpp:37-49).

    Static-only — never instantiated. The two factory methods mirror
    the C++ ``pathBasisSystem`` and ``multiPathBasisSystem`` exactly:
    they return lists of basis-function callables.
    """

    @staticmethod
    def path_basis_system(
        order: int, polynomial_type: PolynomialType
    ) -> list[Callable[[float], float]]:
        """Return ``order+1`` basis callables ``Real -> Real``.

        # C++ parity: ``LsmBasisSystem::pathBasisSystem`` (lsmbasissystem.cpp:109-157).

        Each callable evaluates basis function of order ``i`` (for
        ``i in 0..order``) at the given point. For ``Monomial`` this
        is ``x -> x^i``; for the orthogonal families it is the
        weighted value ``sqrt(w(x)) * value(i, x)`` using the C++
        recurrence coefficients.
        """
        qassert.require(order >= 0, "order must be nonneg")
        if polynomial_type == PolynomialType.Monomial:
            return [_make_monomial(i) for i in range(order + 1)]
        if polynomial_type in (
            PolynomialType.Laguerre,
            PolynomialType.Hermite,
            PolynomialType.Chebyshev2nd,
        ):
            return [_make_weighted(polynomial_type, i) for i in range(order + 1)]
        # Hyperbolic / Legendre / Chebyshev (1st-kind) — Phase 6 carve-outs.
        raise NotImplementedError(
            f"polynomial type {polynomial_type.name} not ported "
            "(see docs/migration/phase6-l6-A-design.md)"
        )

    @staticmethod
    def multi_path_basis_system(
        dim: int, order: int, polynomial_type: PolynomialType
    ) -> list[Callable[[npt.NDArray[np.float64]], float]]:
        """Return multi-dimensional basis callables ``ndarray -> Real``.

        # C++ parity: ``LsmBasisSystem::multiPathBasisSystem`` (lsmbasissystem.cpp:159-182).

        Each callable evaluates a product
        ``prod_k pathBasis[o_k](x_k)`` for some multi-order
        ``(o_1, ..., o_dim)`` with ``sum(o_k) <= order``. The C++
        version emits exactly one callable per such tuple, with the
        ordering ``(0,...,0)`` first then all tuples of order 1 lex-
        sorted, etc.
        """
        qassert.require(dim > 0, "zero dimension")
        single = LsmBasisSystem.path_basis_system(order, polynomial_type)

        def make_term(tuple_orders: tuple[int, ...]) -> Callable[[npt.NDArray[np.float64]], float]:
            # Capture the per-axis basis fns at fixed orders.
            fns = [single[o] for o in tuple_orders]

            def fn(a: npt.NDArray[np.float64]) -> float:
                # C++ parity: MultiDimFct::operator() — product of axis values.
                val = fns[0](float(a[0]))
                for k in range(1, len(fns)):
                    val *= fns[k](float(a[k]))
                return val

            return fn

        # Order 0 — the (0, 0, ..., 0) tuple
        zero_tuple = (0,) * dim
        result: list[Callable[[npt.NDArray[np.float64]], float]] = [
            make_term(zero_tuple)
        ]
        if order == 0:
            return result

        tuples: list[tuple[int, ...]] = [zero_tuple]
        for _ in range(1, order + 1):
            tuples = _next_order_tuples(tuples)
            for t in tuples:
                result.append(make_term(t))
        return result


__all__ = ["LsmBasisSystem", "PolynomialType"]
