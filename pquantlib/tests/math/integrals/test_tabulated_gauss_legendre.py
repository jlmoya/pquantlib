"""Cross-validate TabulatedGaussLegendre against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/distributions.json`` —
``tabulated_gauss_legendre`` (the class is probed alongside the distributions
because that is where its only C++ caller lives:
``BivariateCumulativeNormalDistributionWe04DP``).

Four integrands, chosen so that agreement cannot come from being merely
accurate: a degree-5 polynomial every supported order integrates exactly, a
degree-11 polynomial that order 6 gets wrong and order 12 gets right, an
analytic non-polynomial, and ``|x|`` whose kink at the origin no Gauss rule
resolves — for that one, matching C++ means having C++'s nodes and weights,
not having a good quadrature.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.integrals.tabulated_gauss_legendre import TabulatedGaussLegendre
from pquantlib.testing import reference_reader, tolerance

_INTEGRANDS: dict[str, Callable[[float], float]] = {
    "poly5": lambda x: 3.0 * x**5 - 2.0 * x**3 + x - 0.5,
    "poly11": lambda x: x**11 + 4.0 * x**8 - x * x,
    "exp": math.exp,
    "runge": lambda x: 1.0 / (1.0 + 25.0 * x * x),
    "abs": abs,
}


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/distributions")


def test_matches_cpp_tight(cpp: dict[str, Any]) -> None:
    for case in cpp["tabulated_gauss_legendre"]:
        quad = TabulatedGaussLegendre(case["order"])
        tolerance.tight(
            quad(_INTEGRANDS[case["f"]]),
            case["v"],
            reason=f"order={case['order']}, f={case['f']}",
        )


def test_kinked_integrand_reproduces_the_cpp_quadrature_error(cpp: dict[str, Any]) -> None:
    """``int_-1^1 |x| dx`` is exactly 1; every order returns something else.

    The pinned value being *wrong by the right amount* is what proves the
    nodes and weights are C++'s. Asserted explicitly so nobody replaces the
    tables with ``numpy.polynomial.legendre.leggauss`` — which computes the
    nodes to full precision and would return a different wrong answer.
    """
    cases = [c for c in cpp["tabulated_gauss_legendre"] if c["f"] == "abs"]
    assert cases
    for case in cases:
        assert case["v"] != 1.0
        tolerance.tight(TabulatedGaussLegendre(case["order"])(abs), case["v"])


def test_exact_on_low_degree_polynomials(cpp: dict[str, Any]) -> None:
    """A rule of order n integrates degree <= 2n-1 exactly; poly5 qualifies everywhere."""
    for case in cpp["tabulated_gauss_legendre"]:
        if case["f"] != "poly5":
            continue
        # int_-1^1 (3x^5 - 2x^3 + x - 0.5) dx = -1 (odd powers vanish).
        tolerance.tight(case["v"], -1.0)


def test_unsupported_order_raises() -> None:
    with pytest.raises(LibraryException, match="order 8 not supported"):
        TabulatedGaussLegendre(8)


def test_order_round_trips() -> None:
    quad = TabulatedGaussLegendre(6)
    assert quad.order() == 6
    quad.set_order(20)
    assert quad.order() == 20
