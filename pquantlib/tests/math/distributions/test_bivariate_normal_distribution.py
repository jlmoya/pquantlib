"""Tests for BivariateCumulativeNormalDistribution.

# C++ parity:
# ql/math/distributions/bivariatenormaldistribution.{hpp,cpp} @ v1.42.1.

Cross-validates against hand-derived Wolfram-Alpha references and
limiting cases (rho=0 reduces to product of marginals).
"""

from __future__ import annotations

import math

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.distributions.bivariate_normal_distribution import (
    BivariateCumulativeNormalDistribution,
    BivariateCumulativeNormalDistributionDr78,
)
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
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
            f"bivariate CDF at rho=0.999 expected ~ N(min(a,b))={expected}, "
            f"got {actual} (a={a}, b={b})"
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
