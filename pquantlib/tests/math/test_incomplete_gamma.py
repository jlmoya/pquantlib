"""Cross-validate the incomplete Gamma function against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/distributions.json`` —
``incomplete_gamma``. The (a, x) grid straddles the ``x < a + 1`` switch
between the power series and the Lentz continued fraction, including a pair
either side of the switch that differ in the eighth decimal, so a port that
picked the wrong branch boundary would show up rather than average out.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.math.incomplete_gamma import (
    incomplete_gamma_function,
    incomplete_gamma_function_continued_fraction_repr,
    incomplete_gamma_function_series_repr,
)
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/distributions")


def test_incomplete_gamma_matches_cpp_tight(cpp: dict[str, Any]) -> None:
    """TIGHT: the same series, the same accuracy (1e-13), the same cap (100).

    The only difference between the two sides is the order in which the
    compiler contracts the multiply-adds, which is a last-ulp effect.
    """
    for case in cpp["incomplete_gamma"]:
        tolerance.tight(
            incomplete_gamma_function(case["a"], case["x"]),
            case["P"],
            reason=f"a={case['a']}, x={case['x']}",
        )


def test_series_branch_matches_cpp_tight(cpp: dict[str, Any]) -> None:
    for case in cpp["incomplete_gamma"]:
        if case["series"] is None:
            continue
        tolerance.tight(
            incomplete_gamma_function_series_repr(case["a"], case["x"]),
            case["series"],
            reason=f"a={case['a']}, x={case['x']}",
        )


def test_continued_fraction_branch_matches_cpp_tight(cpp: dict[str, Any]) -> None:
    for case in cpp["incomplete_gamma"]:
        if case["continued_fraction"] is None:
            continue
        tolerance.tight(
            incomplete_gamma_function_continued_fraction_repr(case["a"], case["x"]),
            case["continued_fraction"],
            reason=f"a={case['a']}, x={case['x']}",
        )


def test_dispatch_picks_the_branch_cpp_picks(cpp: dict[str, Any]) -> None:
    """``incomplete_gamma_function`` is the series below ``a+1`` and ``1 - cf`` above.

    Asserted structurally rather than numerically: the two representations do
    not agree to machine precision with each other, so "which branch ran" is
    observable and worth pinning separately from the value.
    """
    for case in cpp["incomplete_gamma"]:
        a, x = case["a"], case["x"]
        if x < a + 1.0:
            assert case["series"] is not None
            tolerance.exact(incomplete_gamma_function(a, x), incomplete_gamma_function_series_repr(a, x))
        else:
            assert case["continued_fraction"] is not None
            tolerance.exact(
                incomplete_gamma_function(a, x),
                1.0 - incomplete_gamma_function_continued_fraction_repr(a, x),
            )
