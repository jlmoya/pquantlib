"""Cross-validate the bivariate cumulative normal against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/distributions.json`` —
``bivariate_normal_dr78``, ``bivariate_normal_we04dp`` and
``bivariate_normal_degenerate``.

This file used to assert against hand-derived limiting cases only, because
both classes were one ``scipy.stats.multivariate_normal.cdf`` delegation and
there was nothing sharper to assert. The probe showed that delegation to be
badly wrong in the tails — at ``rho=-0.95, a=-6, b=2`` the true value is
1.55e-42 and scipy returned 5.55e-17 — so the tests are now driven by C++
values, and the grid deliberately runs out to +/-8 where the old
implementation's absolute noise floor lived.

``rho`` covers every branch of Drezner's sign case-analysis and every order
threshold of West's quadrature selection (|rho| < 0.3 -> order 6, < 0.75 ->
12, otherwise 20, with a separate |rho| >= 0.925 tail expansion).
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.distributions.bivariate_normal_distribution import (
    BivariateCumulativeNormalDistribution,
    BivariateCumulativeNormalDistributionDr78,
    BivariateCumulativeNormalDistributionWe04DP,
)
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.testing import reference_reader, tolerance


def test_we04dp_matches_cpp_tight(v143: dict[str, Any]) -> None:
    for block in v143["bivariate_normal_we04dp"]:
        f = BivariateCumulativeNormalDistributionWe04DP(block["rho"])
        for a, b, expected in block["cases"]:
            tolerance.tight(f(a, b), expected, reason=f"rho={block['rho']}, a={a}, b={b}")


def test_dr78_matches_cpp_tight(v143: dict[str, Any]) -> None:
    for block in v143["bivariate_normal_dr78"]:
        f = BivariateCumulativeNormalDistributionDr78(block["rho"])
        for a, b, expected in block["cases"]:
            tolerance.tight(f(a, b), expected, reason=f"rho={block['rho']}, a={a}, b={b}")


def test_degenerate_correlation_matches_cpp_tight(v143: dict[str, Any]) -> None:
    """rho = +/-1 exactly — West's ``|rho| < 1`` guard skips the quadrature entirely."""
    for case in v143["bivariate_normal_degenerate"]:
        f = BivariateCumulativeNormalDistributionWe04DP(case["rho"])
        tolerance.tight(
            f(case["a"], case["b"]),
            case["we04dp"],
            reason=f"rho={case['rho']}, a={case['a']}, b={case['b']}",
        )


def test_we04dp_resolves_the_far_tail(v143: dict[str, Any]) -> None:
    """The tail values are the ones the previous scipy delegation could not reach.

    Asserted separately, and with an explicit check that the probe still
    contains values far below scipy's ~5.5e-17 noise floor, so that a future
    re-delegation fails here rather than passing a body-only comparison.
    """
    tiny = [
        (block["rho"], a, b, expected)
        for block in v143["bivariate_normal_we04dp"]
        for a, b, expected in block["cases"]
        if 0.0 < expected < 1e-20
    ]
    assert tiny, "probe no longer covers the far tail"
    for rho, a, b, expected in tiny:
        tolerance.tight(
            BivariateCumulativeNormalDistributionWe04DP(rho)(a, b),
            expected,
            reason=f"rho={rho}, a={a}, b={b}",
        )


def test_dr78_is_only_good_to_six_decimal_places(v143: dict[str, Any]) -> None:
    """Drezner 1978 is documented as six-decimal-place accurate, and is.

    The point of keeping both classes is that they are *not* interchangeable.
    If a future change made Dr78 agree with We04DP to machine precision, one
    of the two would have stopped being the C++ algorithm.
    """
    dr78 = {(block["rho"], a, b): v for block in v143["bivariate_normal_dr78"] for a, b, v in block["cases"]}
    we04 = {
        (block["rho"], a, b): v for block in v143["bivariate_normal_we04dp"] for a, b, v in block["cases"]
    }
    gaps = [abs(dr78[k] - we04[k]) for k in dr78 if k in we04]
    assert max(gaps) > 1e-9, "Dr78 and We04DP have become indistinguishable"
    assert max(gaps) < 1e-5, "Dr78 has drifted past its documented six decimal places"


def test_rho_zero_factors_to_product_of_marginals() -> None:
    """When rho=0, the bivariate CDF factors: F(a, b) = N(a) * N(b)."""
    cnd = CumulativeNormalDistribution()
    biv = BivariateCumulativeNormalDistribution(0.0)
    for a, b in [(0.5, 0.5), (-0.5, 0.5), (1.0, -0.5), (-1.0, -1.0)]:
        tolerance.tight(biv(a, b), cnd(a) * cnd(b))


def test_rho_close_to_one_approaches_min() -> None:
    """As rho -> 1, the bivariate CDF approaches N(min(a, b)).

    Custom tier: the gap to the rho = 1 singular limit is O(1 - rho) = 1e-3 at
    rho = 0.999, so 1e-2 bounds the *limit*, not the implementation. (Under the
    old scipy delegation the same 1e-2 was blamed on the integrator; with the
    West algorithm the residual is the limit itself.)
    """
    cnd = CumulativeNormalDistribution()
    biv = BivariateCumulativeNormalDistribution(0.999)
    for a, b in [(0.5, 0.5), (-0.5, 0.5), (1.0, -0.5), (-1.0, -1.0)]:
        tolerance.custom(
            biv(a, b),
            cnd(min(a, b)),
            abs_tol=1e-2,
            rel_tol=0.0,
            reason="distance to the rho -> 1 singular limit is O(1 - rho) = 1e-3 here",
        )


def test_at_origin() -> None:
    """F(0, 0; rho) = 0.25 + arcsin(rho) / (2*pi)."""
    biv = BivariateCumulativeNormalDistribution(0.5)
    tolerance.tight(biv(0.0, 0.0), 0.25 + math.asin(0.5) / (2.0 * math.pi))


@pytest.mark.parametrize(
    "cls",
    [BivariateCumulativeNormalDistributionDr78, BivariateCumulativeNormalDistributionWe04DP],
)
@pytest.mark.parametrize("rho", [1.5, -1.5])
def test_rho_out_of_range_raises(cls: type, rho: float) -> None:
    with pytest.raises(LibraryException, match="rho must be"):
        cls(rho)


def test_default_typedef_is_we04dp() -> None:
    """C++ ``typedef ...We04DP BivariateCumulativeNormalDistribution;``.

    Was previously aliased to a single scipy-backed class shared with Dr78;
    the two are genuinely different algorithms and the alias now says so.
    """
    assert BivariateCumulativeNormalDistribution is BivariateCumulativeNormalDistributionWe04DP
    assert BivariateCumulativeNormalDistributionDr78 is not BivariateCumulativeNormalDistribution


# --- |rho| == 1 (the singular endpoints) ------------------------------------
#
# From coverage wave 5. The constructor admits rho in the CLOSED interval
# [-1, 1] and both endpoints are reached by real engines:
# AnalyticContinuousPartialFloatingLookbackEngine builds rho == +1 whenever
# the lookback window runs to expiry, and
# AnalyticContinuousPartialFixedLookbackEngine builds rho == -1 whenever the
# window starts at expiry. Under the old scipy delegation both endpoints
# raised outright (``LinAlgError: the input matrix must be symmetric positive
# definite``), and wave 5 special-cased them in closed form. The West
# transcription needs no special case: C++ skips its whole Genz series block
# at |rho| == 1 (bivariatenormaldistribution.cpp:213) and keeps only the
# closing correction, which IS the comonotone / countermonotone limit. These
# cases pin what that correction evaluates to, straight from the C++ probe —
# so a future re-delegation, or a "simplification" of the guard, fails here.


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/inst/lookbackvarswap")


def test_rho_plus_one_matches_cpp(cpp: dict[str, Any]) -> None:
    """rho == +1 collapses to ``N(min(a, b))``."""
    cnd = CumulativeNormalDistribution()
    checked = 0
    for name, case in cpp.items():
        if not name.startswith("bvn_rhop1_"):
            continue
        a, b = float(case["inputs"]["x"]), float(case["inputs"]["y"])
        actual = BivariateCumulativeNormalDistribution(1.0)(a, b)
        tolerance.tight(actual, float(case["expected"]["value"]), reason=name)
        tolerance.tight(actual, min(cnd(a), cnd(b)), reason=f"{name}: comonotone limit")
        checked += 1
    assert checked >= 40, f"expected the rho=+1 grid in the reference, got {checked}"


def test_rho_minus_one_matches_cpp(cpp: dict[str, Any]) -> None:
    """rho == -1 collapses to ``max(0, N(a) + N(b) - 1)``.

    The grid straddles every branch of the C++ tail selection: ``a + b``
    below, at and above zero, and ``a`` below and above zero.
    """
    cnd = CumulativeNormalDistribution()
    checked = 0
    zeros = 0
    for name, case in cpp.items():
        if not name.startswith("bvn_rhom1_"):
            continue
        a, b = float(case["inputs"]["x"]), float(case["inputs"]["y"])
        expected = float(case["expected"]["value"])
        actual = BivariateCumulativeNormalDistribution(-1.0)(a, b)
        tolerance.tight(actual, expected, reason=name)
        tolerance.tight(actual, max(0.0, cnd(a) + cnd(b) - 1.0), reason=f"{name}: countermonotone")
        zeros += int(expected == 0.0)
        checked += 1
    assert checked >= 40, f"expected the rho=-1 grid in the reference, got {checked}"
    assert zeros > 0, "the a + b <= 0 branch must be exercised"


def test_rho_one_is_the_limit_of_rho_below_one() -> None:
    """The closed form is the limit the integrator is converging towards."""
    for rho in (0.9, 0.99, 0.999):
        near = BivariateCumulativeNormalDistribution(rho)(0.3, -0.7)
        limit = BivariateCumulativeNormalDistribution(1.0)(0.3, -0.7)
        assert abs(near - limit) < 0.2
    assert abs(
        BivariateCumulativeNormalDistribution(0.999)(0.3, -0.7)
        - BivariateCumulativeNormalDistribution(1.0)(0.3, -0.7)
    ) < abs(
        BivariateCumulativeNormalDistribution(0.9)(0.3, -0.7)
        - BivariateCumulativeNormalDistribution(1.0)(0.3, -0.7)
    )
