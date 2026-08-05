"""Cross-validate CumulativeGammaDistribution against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/distributions.json`` —
``cumulative_gamma`` and ``gamma_function``. Shape parameters run from 0.25 to
100 and the argument from below zero out to 400, so both the series branch
(``x < a+1``) and the continued-fraction branch are exercised at every shape,
including the saturated ``P ~ 1`` and ``P ~ 0`` ends.
"""

from __future__ import annotations

from typing import Any

from pquantlib.math.distributions.gamma_distribution import (
    CumulativeGammaDistribution,
    GammaFunction,
)
from pquantlib.testing import tolerance


def test_cumulative_gamma_matches_cpp_tight(v143: dict[str, Any]) -> None:
    for case in v143["cumulative_gamma"]:
        tolerance.tight(
            CumulativeGammaDistribution(case["a"])(case["x"]),
            case["v"],
            reason=f"a={case['a']}, x={case['x']}",
        )


def test_non_positive_argument_is_zero(v143: dict[str, Any]) -> None:
    """``x <= 0`` short-circuits to exactly 0 before any series runs."""
    for case in v143["cumulative_gamma"]:
        if case["x"] <= 0.0:
            tolerance.exact(CumulativeGammaDistribution(case["a"])(case["x"]), 0.0)


def test_gamma_function_log_value_matches_cpp_tight(v143: dict[str, Any]) -> None:
    g = GammaFunction()
    for x, expected in v143["gamma_function"]["log_value"]:
        tolerance.tight(g.log_value(x), expected, reason=f"x={x}")


def test_gamma_function_value_matches_cpp_tight(v143: dict[str, Any]) -> None:
    """Covers all three C++ branches: x >= 1, the x<1 recurrence, the x<=-20 reflection."""
    g = GammaFunction()
    for x, expected in v143["gamma_function"]["value"]:
        tolerance.tight(g.value(x), expected, reason=f"x={x}")
