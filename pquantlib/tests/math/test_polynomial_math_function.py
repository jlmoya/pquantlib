"""Cross-validate PolynomialFunction against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/tail.json`` —
``polynomial_function``. Degrees 0 through 4, including a coefficient vector
with interior zeros so a Horner rewrite that skips them would show up.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.polynomial_math_function import PolynomialFunction
from pquantlib.testing import tolerance


def test_values_derivative_and_primitive_match_cpp_exactly(v143_tail: dict[str, Any]) -> None:
    """EXACT for value and derivative: the same running-power loop, same order."""
    for block in v143_tail["polynomial_function"]:
        f = PolynomialFunction(block["coefficients"])
        assert f.order() == block["order"]
        for t, value, derivative, primitive in block["values"]:
            ctx = f"c={block['coefficients']}, t={t}"
            tolerance.exact(f(t), value, reason=ctx)
            tolerance.exact(f.derivative(t), derivative, reason=ctx)
            tolerance.tight(f.primitive(t), primitive, reason=ctx)


def test_coefficient_vectors_match_cpp_exactly(v143_tail: dict[str, Any]) -> None:
    for block in v143_tail["polynomial_function"]:
        f = PolynomialFunction(block["coefficients"])
        for actual, expected in zip(
            f.derivative_coefficients(), block["derivative_coefficients"], strict=True
        ):
            tolerance.exact(actual, expected)
        for actual, expected in zip(f.primitive_coefficients(), block["primitive_coefficients"], strict=True):
            tolerance.exact(actual, expected)


def test_rolling_window_coefficients_match_cpp_tight(v143_tail: dict[str, Any]) -> None:
    """The integral transform is a Pascal-triangle matrix product; the derivative
    transform is its inverse. C++ forms ``inverse(eqs) * k`` explicitly and the
    port solves the system instead — TIGHT, not EXACT, is what that costs.
    """
    for block in v143_tail["polynomial_function"]:
        f = PolynomialFunction(block["coefficients"])
        ctx = f"c={block['coefficients']}"
        for actual, expected in zip(
            f.definite_integral_coefficients(0.5, 4.0),
            block["definite_integral_coefficients"],
            strict=True,
        ):
            tolerance.tight(actual, expected, reason=ctx)
        for actual, expected in zip(
            f.definite_derivative_coefficients(0.5, 4.0),
            block["definite_derivative_coefficients"],
            strict=True,
        ):
            tolerance.tight(actual, expected, reason=ctx)
        tolerance.exact(f.definite_integral(0.5, 4.0), block["definite_integral"])


def test_empty_coefficient_vector_raises() -> None:
    with pytest.raises(LibraryException, match="empty coefficient vector"):
        PolynomialFunction([])
