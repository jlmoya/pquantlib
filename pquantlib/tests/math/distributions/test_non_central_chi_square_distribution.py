"""Tests for the NonCentralCumulativeChiSquareDistribution.

C++ parity: ql/math/distributions/chisquaredistribution.{hpp,cpp} @ v1.42.1.

The pquantlib implementation delegates to ``scipy.stats.ncx2.cdf``.
The tests here are sanity tests — agreement with the C++ series is
validated transitively via the L4-B CIR/ECIR discount-bond-option tests.
"""

from __future__ import annotations

import math

from pquantlib.math.distributions.non_central_chi_square_distribution import (
    NonCentralCumulativeChiSquareDistribution,
)
from pquantlib.testing.tolerance import tight


def test_ncx2_returns_zero_below_origin() -> None:
    """``x <= 0`` -> 0 (matches C++ early-return)."""
    cdf = NonCentralCumulativeChiSquareDistribution(df=4.0, ncp=2.0)
    tight(cdf(-1.0), 0.0)
    tight(cdf(0.0), 0.0)


def test_ncx2_central_limit_to_chi_square() -> None:
    """At ncp=0 the distribution reduces to the central chi-square.

    Spot-check against the scipy reference for df=2 (an exponential
    distribution with rate 0.5) at x = 4.0 — CDF should be
    1 - exp(-2) ~= 0.86466.
    """
    cdf = NonCentralCumulativeChiSquareDistribution(df=2.0, ncp=0.0)
    tight(cdf(4.0), 1.0 - math.exp(-2.0))


def test_ncx2_monotone_increasing() -> None:
    """CDF is monotone non-decreasing and approaches 1 in the tail."""
    cdf = NonCentralCumulativeChiSquareDistribution(df=4.0, ncp=2.0)
    prev = 0.0
    for x in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        cur = cdf(x)
        assert cur >= prev
        prev = cur
    # CDF approaches 1 in the deep tail (x large enough).
    tail_threshold = 0.999
    assert cdf(100.0) > tail_threshold
