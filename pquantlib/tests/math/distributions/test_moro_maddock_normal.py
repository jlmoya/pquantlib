"""Cross-validate the Moro and Maddock normal variants against the v1.43 probe.

Reference: ``migration-harness/references/v143/math/distributions.json`` —
``moro_inverse_cumulative_normal``, ``maddock_cumulative_normal`` and
``maddock_inverse_cumulative_normal``.

The probability grid deliberately straddles Moro's |x - 0.5| < 0.42 branch
boundary and Acklam's 0.02425 region edges, and reaches 1e-15 and 1 - 1e-12,
because the whole reason these classes coexist is that they disagree in the
tails. A shifted (average, sigma) variant is pinned for each so the affine
wrapper is covered separately from the standard quantile.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
    MaddockCumulativeNormal,
)
from pquantlib.math.distributions.inverse_cumulative_normal import (
    MaddockInverseCumulativeNormal,
    MoroInverseCumulativeNormal,
)
from pquantlib.testing import tolerance


def test_moro_standard_matches_cpp_tight(v143: dict[str, Any]) -> None:
    block = v143["moro_inverse_cumulative_normal"]
    moro = MoroInverseCumulativeNormal()
    for x, expected in block["standard"]:
        tolerance.tight(moro(x), expected, reason=f"x={x}")


def test_moro_shifted_matches_cpp_tight(v143: dict[str, Any]) -> None:
    block = v143["moro_inverse_cumulative_normal"]
    moro = MoroInverseCumulativeNormal(block["average"], block["sigma"])
    for x, expected in block["shifted"]:
        tolerance.tight(moro(x), expected, reason=f"x={x}")


def test_moro_rejects_the_closed_interval_endpoints() -> None:
    moro = MoroInverseCumulativeNormal()
    with pytest.raises(LibraryException, match="must be 0<x<1"):
        moro(0.0)
    with pytest.raises(LibraryException, match="must be 0<x<1"):
        moro(1.0)


def test_maddock_cdf_matches_cpp_tight(v143: dict[str, Any]) -> None:
    block = v143["maddock_cumulative_normal"]
    cdf = MaddockCumulativeNormal()
    for x, expected in block["standard"]:
        tolerance.tight(cdf(x), expected, reason=f"x={x}")


def test_maddock_cdf_shifted_matches_cpp_tight(v143: dict[str, Any]) -> None:
    block = v143["maddock_cumulative_normal"]
    cdf = MaddockCumulativeNormal(block["average"], block["sigma"])
    for x, expected in block["shifted"]:
        tolerance.tight(cdf(x), expected, reason=f"x={x}")


def test_maddock_cdf_is_not_the_asymptotic_cumulative_normal(v143: dict[str, Any]) -> None:
    """The two C++ CDFs part company in the far left tail, by design.

    ``CumulativeNormalDistribution`` switches to the Abramowitz-Stegun
    asymptotic expansion once the principal result drops below 1e-8;
    ``MaddockCumulativeNormal`` never does. If a future refactor collapsed the
    two, this test is what notices.
    """
    maddock = MaddockCumulativeNormal()
    quantlib = CumulativeNormalDistribution()
    x = -20.0
    assert maddock(x) != quantlib(x)


def test_maddock_inverse_matches_cpp_tight(v143: dict[str, Any]) -> None:
    block = v143["maddock_inverse_cumulative_normal"]
    inv = MaddockInverseCumulativeNormal()
    for x, expected in block["standard"]:
        tolerance.tight(inv(x), expected, reason=f"x={x}")


def test_maddock_inverse_shifted_matches_cpp_tight(v143: dict[str, Any]) -> None:
    block = v143["maddock_inverse_cumulative_normal"]
    inv = MaddockInverseCumulativeNormal(block["average"], block["sigma"])
    for x, expected in block["shifted"]:
        tolerance.tight(inv(x), expected, reason=f"x={x}")


def test_moro_and_maddock_are_not_interchangeable(v143: dict[str, Any]) -> None:
    """Sanity check on the delegation premise.

    SciPy's ``ndtri`` and Boost's ``quantile`` are two implementations of the
    *same* function and agree to TIGHT above — that is what the two tests
    before this one assert, against C++ values. Moro's is a different, cruder
    approximation, so it must **not** agree to TIGHT; if it did, one of the two
    ports would be computing the wrong thing.

    Measured across the pinned probability grid the largest relative gap is
    1.43e-8 (at p = 0.50001, where Moro's Beasley-Springer central branch is at
    its weakest). The threshold below is three orders of magnitude above TIGHT
    and an order of magnitude under the measured worst case, so it discriminates
    without pinning a number the approximation never promised.
    """
    moro = MoroInverseCumulativeNormal()
    maddock = MaddockInverseCumulativeNormal()
    gaps: list[float] = []
    for pair in v143["moro_inverse_cumulative_normal"]["standard"]:
        x = float(pair[0])
        a, b = moro(x), maddock(x)
        gaps.append(abs(a - b) / abs(b) if b != 0.0 else abs(a - b))
    assert max(gaps) > 1e-9
