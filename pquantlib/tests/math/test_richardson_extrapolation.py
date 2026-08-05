"""Cross-validate RichardsonExtrapolation against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/tail.json`` —
``richardson``. Two test functions with different true orders (``exp(1+h)``
is first order in h, ``exp(1+h*h)`` second), extrapolated with the right
order, the wrong order, and with the order estimated.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.richardson_extrapolation import RichardsonExtrapolation
from pquantlib.testing import tolerance

_FUNCS: dict[str, Callable[[float], float]] = {
    "exp1h": lambda h: math.exp(1.0 + h),
    "exp1hh": lambda h: math.exp(1.0 + h * h),
}


def test_known_order_matches_cpp_exactly(v143_tail: dict[str, Any]) -> None:
    """EXACT: one closed-form combination of two function values."""
    for case in v143_tail["richardson"]["known_order"]:
        extrap = RichardsonExtrapolation(_FUNCS[case["f"]], case["delta_h"], case["n"])
        tolerance.exact(extrap(case["t"]), case["v"], reason=str(case))


def test_unknown_order_matches_cpp_tight(v143_tail: dict[str, Any]) -> None:
    """The order is found by a 0.1-step bracket scan and a Brent solve to 1e-8.

    TIGHT rather than EXACT because the answer runs through Brent, whose
    iterate depends on the last bit of the consistency equation.
    """
    for case in v143_tail["richardson"]["unknown_order"]:
        extrap = RichardsonExtrapolation(_FUNCS[case["f"]], case["delta_h"])
        tolerance.tight(extrap.with_unknown_order(case["t"], case["s"]), case["v"], reason=str(case))


def test_extrapolation_beats_the_raw_evaluation(v143_tail: dict[str, Any]) -> None:
    """Sanity: extrapolating with the *correct* order must be closer to exp(1)
    than the unextrapolated f(delta_h). Guards against a sign or ordering slip
    that would still match a single pinned number if it were pinned wrong.
    """
    exact = v143_tail["richardson"]["exact"]
    for case in v143_tail["richardson"]["known_order"]:
        if case["f"] != "exp1h" or case["n"] != 1.0:
            continue
        raw = _FUNCS[case["f"]](case["delta_h"])
        assert abs(case["v"] - exact) < abs(raw - exact)


def test_known_order_requires_an_order() -> None:
    extrap = RichardsonExtrapolation(_FUNCS["exp1h"], 0.1)
    with pytest.raises(LibraryException, match="order of convergence must be known"):
        extrap(2.0)


def test_scaling_factors_are_validated() -> None:
    extrap = RichardsonExtrapolation(_FUNCS["exp1h"], 0.1, 1.0)
    with pytest.raises(LibraryException, match="scaling factor must be greater than 1"):
        extrap(1.0)
    with pytest.raises(LibraryException, match="scaling factors must be greater than 1"):
        extrap.with_unknown_order(1.0, 0.5)
    with pytest.raises(LibraryException, match="t must be greater than s"):
        extrap.with_unknown_order(2.0, 4.0)
