"""Bernstein polynomial basis.

# C++ parity: ql/math/bernsteinpolynomial.hpp + bernsteinpolynomial.cpp (v1.42.1).

``BernsteinPolynomial.get(i, n, x)`` returns the i-th Bernstein basis
polynomial of degree n evaluated at x:

    B_i^n(x) = C(n, i) * x^i * (1 - x)^(n-i)

where C(n, i) = n! / (i! (n-i)!).
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.math.factorial import Factorial


class BernsteinPolynomial:
    @staticmethod
    def get(i: int, n: int, x: float) -> float:
        # C++ takes Natural (unsigned). Guard explicitly because
        # ``Factorial.get(n - i)`` would propagate a misleading error if
        # i > n; better to raise at the API surface.
        qassert.require(i >= 0, f"BernsteinPolynomial.get requires i >= 0, got {i}")
        qassert.require(n >= 0, f"BernsteinPolynomial.get requires n >= 0, got {n}")
        qassert.require(i <= n, f"BernsteinPolynomial.get requires i <= n, got i={i}, n={n}")
        coeff = Factorial.get(n) / (Factorial.get(n - i) * Factorial.get(i))
        return coeff * (x**i) * ((1.0 - x) ** (n - i))
