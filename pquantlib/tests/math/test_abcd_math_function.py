"""Cross-validate AbcdMathFunction against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/tail.json`` —
``abcd_math_function``. Five parameter sets: the C++ defaults, ``b == 0``
(which short-circuits ``maximumLocation``), ``a < 0`` with an interior
maximum, a negative-``b`` set that still passes ``validate``, and the
all-zero degenerate. Times include ``t < 0``, which the C++ clamps to 0
rather than evaluating.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.abcd_math_function import AbcdMathFunction
from pquantlib.testing import tolerance


def _make(block: dict[str, Any]) -> AbcdMathFunction:
    return AbcdMathFunction.from_coefficients(block["abcd"])


def test_values_derivative_and_primitive_match_cpp_tight(v143_tail: dict[str, Any]) -> None:
    for block in v143_tail["abcd_math_function"]:
        f = _make(block)
        for t, value, derivative, primitive in block["values"]:
            ctx = f"abcd={block['abcd']}, t={t}"
            tolerance.tight(f(t), value, reason=ctx)
            tolerance.tight(f.derivative(t), derivative, reason=ctx)
            tolerance.tight(f.primitive(t), primitive, reason=ctx)


def test_negative_time_is_clamped_to_zero(v143_tail: dict[str, Any]) -> None:
    """C++ returns exactly 0 for t < 0 in all three of value/derivative/primitive."""
    for block in v143_tail["abcd_math_function"]:
        f = _make(block)
        for t, value, derivative, primitive in block["values"]:
            if t >= 0:
                continue
            assert (value, derivative, primitive) == (0.0, 0.0, 0.0)
            tolerance.exact(f(t), 0.0)
            tolerance.exact(f.derivative(t), 0.0)
            tolerance.exact(f.primitive(t), 0.0)


def test_maximum_matches_cpp_exactly(v143_tail: dict[str, Any]) -> None:
    """EXACT: both are one arithmetic expression with no iteration."""
    for block in v143_tail["abcd_math_function"]:
        f = _make(block)
        tolerance.exact(f.maximum_location(), block["maximum_location"])
        tolerance.exact(f.maximum_value(), block["maximum_value"])
        tolerance.exact(f.long_term_value(), block["long_term_value"])


def test_coefficient_vectors_match_cpp_tight(v143_tail: dict[str, Any]) -> None:
    """The rolling-window transforms — only ever consumed by other code."""
    for block in v143_tail["abcd_math_function"]:
        f = _make(block)
        ctx = f"abcd={block['abcd']}"
        for actual, expected in zip(
            f.derivative_coefficients(), block["derivative_coefficients"], strict=True
        ):
            tolerance.tight(actual, expected, reason=ctx)
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


def test_derivative_coefficients_have_no_constant_term(v143_tail: dict[str, Any]) -> None:
    """The fourth entry is 0, not d — easy to copy wrong from the abcd vector."""
    for block in v143_tail["abcd_math_function"]:
        assert block["derivative_coefficients"][3] == 0.0
        assert _make(block).derivative_coefficients()[3] == 0.0


@pytest.mark.parametrize(
    ("abcd", "message"),
    [
        ((0.1, 0.1, 0.0, 0.1), "must be positive"),
        ((0.1, 0.1, 0.5, -0.1), "must be non negative"),
        ((-1.0, 0.1, 0.5, 0.1), "must be non negative"),
        ((0.1, -10.0, 0.5, 0.001), "negative function value at stationary point"),
    ],
)
def test_validate_rejects(abcd: tuple[float, float, float, float], message: str) -> None:
    with pytest.raises(LibraryException, match=message):
        AbcdMathFunction(*abcd)
