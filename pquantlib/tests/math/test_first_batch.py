"""Cross-validate Stage-5 math primitives against the C++ probe.

Probe source: migration-harness/cpp/probes/math/first_batch_probe.cpp
Reference:    migration-harness/references/math/first/batch.json

Modules tested: constants, closeness, rounding, factorial, error_function,
beta, bernstein_polynomial, pascal_triangle.
"""

from __future__ import annotations

import math
import sys
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.bernstein_polynomial import BernsteinPolynomial
from pquantlib.math.beta import beta_function, incomplete_beta_function
from pquantlib.math.closeness import close, close_enough
from pquantlib.math.constants import M_E, M_PI, QL_EPSILON, QL_MAX_REAL
from pquantlib.math.error_function import ErrorFunction
from pquantlib.math.factorial import Factorial
from pquantlib.math.pascal_triangle import PascalTriangle
from pquantlib.math.rounding import (
    CeilingTruncation,
    ClosestRounding,
    DownRounding,
    FloorTruncation,
    Rounding,
    Type,
    UpRounding,
)
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("math/first/batch")


# --- constants ----------------------------------------------------------


def test_constants_match_python_stdlib() -> None:
    tolerance.exact(M_PI, math.pi)
    tolerance.exact(M_E, math.e)
    tolerance.exact(QL_EPSILON, sys.float_info.epsilon)
    tolerance.exact(QL_MAX_REAL, sys.float_info.max)


# --- closeness ----------------------------------------------------------


def test_closeness_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["closeness"]:
        x = float(case["x"])
        y = float(case["y"])
        assert close(x, y) is bool(case["close"]), case
        assert close_enough(x, y) is bool(case["close_enough"]), case
        assert close(x, y, 7) is bool(case["close_n7"]), case


def test_closeness_equal_returns_true() -> None:
    assert close(1.0, 1.0)
    assert close_enough(1.0, 1.0)
    assert close(0.0, 0.0)


def test_closeness_infinity() -> None:
    inf = float("inf")
    assert close(inf, inf)
    assert close_enough(inf, inf)


# --- rounding -----------------------------------------------------------


def _type_from_name(name: str) -> Type:
    return {
        "None": Type.None_,
        "Up": Type.Up,
        "Down": Type.Down,
        "Closest": Type.Closest,
        "Floor": Type.Floor,
        "Ceiling": Type.Ceiling,
    }[name]


def test_rounding_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["rounding"]:
        r = Rounding(int(case["precision"]), _type_from_name(case["type"]), int(case["digit"]))
        tolerance.tight(r(float(case["value"])), float(case["result"]))


def test_rounding_default_is_noop() -> None:
    r = Rounding()
    tolerance.exact(r(1.23456789), 1.23456789)


def test_rounding_subclasses() -> None:
    up = UpRounding(2)
    down = DownRounding(2)
    closest = ClosestRounding(2)
    floor = FloorTruncation(2)
    ceiling = CeilingTruncation(2)
    tolerance.tight(up(1.234), 1.24)
    tolerance.tight(down(1.239), 1.23)
    tolerance.tight(closest(1.235), 1.24)
    tolerance.tight(floor(1.235), 1.24)
    tolerance.tight(ceiling(-1.235), -1.24)


# --- factorial ----------------------------------------------------------


def test_factorial_get_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["factorial"]["get"]:
        n = int(case["n"])
        if n <= 27:
            # Tabulated values must be bit-identical.
            tolerance.exact(Factorial.get(n), float(case["value"]))
        else:
            # Non-tabulated: Python math.lgamma vs C++ GammaFunction.logValue
            # diverge by a few ULPs (~1e-9 relative for n=28..170). Documented
            # divergence; full GammaFunction port is deferred to a distributions
            # cluster.
            tolerance.loose(
                Factorial.get(n),
                float(case["value"]),
                reason="C++ GammaFunction vs Python math.lgamma — ~1e-9 relative",
            )


def test_factorial_ln_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["factorial"]["ln"]:
        n = int(case["n"])
        if n <= 27:
            tolerance.tight(Factorial.ln(n), float(case["value"]))
        else:
            tolerance.loose(
                Factorial.ln(n),
                float(case["value"]),
                reason="C++ GammaFunction.logValue vs Python math.lgamma",
            )


def test_factorial_zero_and_one() -> None:
    assert Factorial.get(0) == 1.0
    assert Factorial.get(1) == 1.0
    tolerance.exact(Factorial.ln(0), 0.0)


def test_factorial_negative_input_raises() -> None:
    # C++ uses Natural (unsigned); Python int allows negatives, but Python
    # negative indexing into _FIRST_FACTORIALS would silently return the
    # last tabulated value. The explicit guard turns that into a clear
    # LibraryException.
    with pytest.raises(LibraryException, match="n >= 0"):
        Factorial.get(-1)
    with pytest.raises(LibraryException, match="n >= 0"):
        Factorial.ln(-1)


# --- error_function -----------------------------------------------------


def test_error_function_matches_cpp(cpp: dict[str, Any]) -> None:
    ef = ErrorFunction()
    for case in cpp["error_function"]:
        # LOOSE tier: the C++ source uses a Sun-Microsystems polynomial fit
        # whose accuracy is ~10^-14 across most of the input range, with
        # known divergence beyond |x| > 4 where the residual approaches 0
        # vs C++ returning ±1 exactly.
        tolerance.loose(
            ef(float(case["x"])),
            float(case["erf"]),
            reason="C++ uses a polynomial-fit erf; Python stdlib uses C99 erf; both agree to ~1e-14",
        )


# --- beta ---------------------------------------------------------------


def test_beta_function_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["beta"]["beta_function"]:
        tolerance.tight(beta_function(float(case["z"]), float(case["w"])), float(case["value"]))


def test_incomplete_beta_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["beta"]["incomplete_beta"]:
        a = float(case["a"])
        b = float(case["b"])
        x = float(case["x"])
        # LOOSE tier for the iterative continued-fraction expansion; both
        # the C++ and Python paths converge to ~1e-15 typically, but the
        # `math.lgamma` vs C++ `GammaFunction.logValue` divergence can push
        # residuals to ~1e-12 at extreme (a, b, x).
        tolerance.loose(
            incomplete_beta_function(a, b, x),
            float(case["value"]),
            reason="C++ GammaFunction.logValue vs Python math.lgamma — agreement to ~1e-12",
        )


def test_incomplete_beta_endpoints() -> None:
    assert incomplete_beta_function(2.0, 3.0, 0.0) == 0.0
    assert incomplete_beta_function(2.0, 3.0, 1.0) == 1.0


# --- bernstein ----------------------------------------------------------


def test_bernstein_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["bernstein"]:
        i, n, x = int(case["i"]), int(case["n"]), float(case["x"])
        tolerance.tight(BernsteinPolynomial.get(i, n, x), float(case["value"]))


def test_bernstein_partition_of_unity() -> None:
    # Σ_i B_i^n(x) = 1 for any x in [0,1].
    n = 5
    for x in (0.0, 0.25, 0.5, 0.75, 1.0):
        total = sum(BernsteinPolynomial.get(i, n, x) for i in range(n + 1))
        tolerance.tight(total, 1.0)


def test_bernstein_invalid_inputs_raise() -> None:
    with pytest.raises(LibraryException, match="i >= 0"):
        BernsteinPolynomial.get(-1, 5, 0.5)
    with pytest.raises(LibraryException, match="n >= 0"):
        BernsteinPolynomial.get(0, -1, 0.5)
    with pytest.raises(LibraryException, match="i <= n"):
        BernsteinPolynomial.get(6, 5, 0.5)


# --- pascal -------------------------------------------------------------


def test_pascal_matches_cpp(cpp: dict[str, Any]) -> None:
    for case in cpp["pascal"]:
        order = int(case["order"])
        expected = tuple(int(v) for v in case["row"])
        assert PascalTriangle.get(order) == expected


def test_pascal_row_sums_are_powers_of_two() -> None:
    for n in range(10):
        assert sum(PascalTriangle.get(n)) == 2**n


def test_pascal_negative_order_raises() -> None:
    with pytest.raises(LibraryException, match="order >= 0"):
        PascalTriangle.get(-1)
