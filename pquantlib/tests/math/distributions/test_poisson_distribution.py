"""Cross-validate the Poisson family against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/distributions.json`` —
``poisson`` and ``inverse_cumulative_poisson``. Intensities span 0 (the
degenerate branch) to 50, and ``k`` runs well past the mode so both the
saturated head and the vanishing tail of the pmf are pinned.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.distributions.poisson_distribution import (
    CumulativePoissonDistribution,
    InverseCumulativePoisson,
    PoissonDistribution,
)
from pquantlib.testing import tolerance


def test_pdf_matches_cpp_tight(v143: dict[str, Any]) -> None:
    for block in v143["poisson"]:
        pdf = PoissonDistribution(block["mu"])
        for k, expected in block["pdf"]:
            tolerance.tight(pdf(k), expected, reason=f"mu={block['mu']}, k={k}")


def test_cdf_matches_cpp_tight(v143: dict[str, Any]) -> None:
    for block in v143["poisson"]:
        cdf = CumulativePoissonDistribution(block["mu"])
        for k, expected in block["cdf"]:
            tolerance.tight(cdf(k), expected, reason=f"mu={block['mu']}, k={k}")


def test_zero_intensity_is_a_point_mass(v143: dict[str, Any]) -> None:
    """mu == 0 takes a separate C++ branch and must be bit-exact, not merely close."""
    block = next(b for b in v143["poisson"] if b["mu"] == 0.0)
    pdf = PoissonDistribution(0.0)
    for k, expected in block["pdf"]:
        tolerance.exact(pdf(k), expected)


def test_negative_intensity_raises() -> None:
    with pytest.raises(LibraryException, match="mu must be non negative"):
        PoissonDistribution(-1.0)


def test_inverse_matches_cpp_exactly(v143: dict[str, Any]) -> None:
    """EXACT: the answer is an integer count (or the x == 1 sentinel)."""
    for block in v143["inverse_cumulative_poisson"]:
        inv = InverseCumulativePoisson(block["lambda"])
        for x, expected in block["cases"]:
            tolerance.exact(inv(x), expected, reason=f"lambda={block['lambda']}, x={x}")


def test_inverse_at_one_is_max_real(v143: dict[str, Any]) -> None:
    for block in v143["inverse_cumulative_poisson"]:
        expected = dict(block["cases"])[1.0]
        assert expected == sys.float_info.max
        tolerance.exact(InverseCumulativePoisson(block["lambda"])(1.0), expected)


def test_inverse_at_zero_reproduces_the_cpp_unsigned_wraparound(v143: dict[str, Any]) -> None:
    """x == 0 skips the loop and C++ returns ``Real(BigNatural(0) - 1)``.

    That is 2^64 - 1 as a double, not -1. Pinned separately because it is the
    one place where reproducing C++ means reproducing an integer wraparound,
    and a port that "fixed" it would disagree with every C++ caller.
    """
    for block in v143["inverse_cumulative_poisson"]:
        expected = dict(block["cases"])[0.0]
        assert expected == float(2**64 - 1)
        tolerance.exact(InverseCumulativePoisson(block["lambda"])(0.0), expected)


def test_inverse_rejects_arguments_outside_the_unit_interval() -> None:
    inv = InverseCumulativePoisson(1.0)
    with pytest.raises(LibraryException, match="only defined on the interval"):
        inv(-0.1)
    with pytest.raises(LibraryException, match="only defined on the interval"):
        inv(1.1)


def test_non_positive_lambda_raises() -> None:
    with pytest.raises(LibraryException, match="lambda must be positive"):
        InverseCumulativePoisson(0.0)
