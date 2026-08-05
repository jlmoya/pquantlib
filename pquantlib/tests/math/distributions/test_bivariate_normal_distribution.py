"""Tests for BivariateCumulativeNormalDistribution.

# C++ parity:
# ql/math/distributions/bivariatenormaldistribution.{hpp,cpp} @ v1.42.1.

Cross-validates against hand-derived Wolfram-Alpha references and
limiting cases (rho=0 reduces to product of marginals).
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.distributions.bivariate_normal_distribution import (
    BivariateCumulativeNormalDistribution,
    BivariateCumulativeNormalDistributionDr78,
)
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight


def test_rho_zero_factors_to_product_of_marginals() -> None:
    """When rho=0, the bivariate CDF factors: F(a, b) = N(a) * N(b)."""
    cnd = CumulativeNormalDistribution()
    biv = BivariateCumulativeNormalDistribution(0.0)
    for a, b in [(0.5, 0.5), (-0.5, 0.5), (1.0, -0.5), (-1.0, -1.0)]:
        tight(biv(a, b), cnd(a) * cnd(b))


def test_rho_close_to_one_approaches_min() -> None:
    """As rho -> 1, the bivariate CDF approaches N(min(a, b)).

    Custom tier (abs_tol=1e-2): scipy's Genz-Bretz integrator at
    rho=0.999 still has ~1% error vs the rho=1 singular limit. The
    monotone-convergence claim is what matters, not the rate.
    """
    cnd = CumulativeNormalDistribution()
    biv = BivariateCumulativeNormalDistribution(0.999)
    for a, b in [(0.5, 0.5), (-0.5, 0.5), (1.0, -0.5), (-1.0, -1.0)]:
        actual = biv(a, b)
        expected = cnd(min(a, b))
        assert abs(actual - expected) < 1e-2, (
            f"bivariate CDF at rho=0.999 expected ~ N(min(a,b))={expected}, got {actual} (a={a}, b={b})"
        )


def test_at_origin() -> None:
    """F(0, 0; rho) = 0.25 + arcsin(rho) / (2*pi)."""
    biv = BivariateCumulativeNormalDistribution(0.5)
    tight(biv(0.0, 0.0), 0.25 + math.asin(0.5) / (2.0 * math.pi))


def test_rho_out_of_range_raises_positive() -> None:
    with pytest.raises(LibraryException, match="rho must be"):
        BivariateCumulativeNormalDistribution(1.5)


def test_rho_out_of_range_raises_negative() -> None:
    with pytest.raises(LibraryException, match="rho must be"):
        BivariateCumulativeNormalDistribution(-1.5)


def test_dr78_alias_is_same_class() -> None:
    """``Dr78`` alias points to the same default class.

    # C++ parity: typedef ``BivariateCumulativeNormalDistribution``
    # = ``BivariateCumulativeNormalDistributionWe04DP``. The Dr78
    # 6-decimal-place variant is the lower-precision legacy; we expose
    # the same scipy-backed implementation for both names.
    """
    assert BivariateCumulativeNormalDistributionDr78 is BivariateCumulativeNormalDistribution


# --- |rho| == 1 (the singular endpoints) ------------------------------------
#
# align(math): the constructor above admits rho in the CLOSED interval
# [-1, 1], but scipy's ``multivariate_normal.cdf`` rejects the singular
# covariance outright (``LinAlgError: the input matrix must be symmetric
# positive definite``), so both endpoints used to raise. They are reached by
# real engines: AnalyticContinuousPartialFloatingLookbackEngine builds
# rho == +1 whenever the lookback window runs to expiry, and
# AnalyticContinuousPartialFixedLookbackEngine builds rho == -1 whenever the
# window starts at expiry. C++ handles them by skipping its whole Genz series
# block (bivariatenormaldistribution.cpp:212) and keeping only the closing
# correction, which is the comonotone / countermonotone limit; the cases below
# pin what that correction actually evaluates to, straight from the C++ probe.


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
        tight(actual, float(case["expected"]["value"]), reason=name)
        tight(actual, min(cnd(a), cnd(b)), reason=f"{name}: comonotone limit")
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
        tight(actual, expected, reason=name)
        tight(actual, max(0.0, cnd(a) + cnd(b) - 1.0), reason=f"{name}: countermonotone")
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
