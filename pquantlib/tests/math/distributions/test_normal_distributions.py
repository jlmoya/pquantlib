"""Cross-validate normal pdf / cdf / inverse-cdf against the C++ probe.

Probe key: cluster/b -> "distributions" -> {normal_pdf, normal_cdf, inverse_normal_cdf}.

Tolerances:
- pdf: TIGHT — closed-form ``(1/sqrt(2*pi*sigma^2)) * exp(-...)`` matches
  C++ bit-for-bit up to TIGHT.
- cdf: TIGHT for most inputs; some tail values shift by a few ULPs because
  the C++ uses its own Sun-Microsystems-derived erf approximation while
  the port uses ``math.erf`` (C99). Documented in error_function.py.
- inverse cdf: TIGHT — both sides use the same Acklam rational polynomial.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.distributions.inverse_cumulative_normal import (
    InverseCumulativeNormal,
)
from pquantlib.math.distributions.normal_distribution import NormalDistribution
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("cluster/b")


def test_normal_pdf_matches_cpp(cpp: dict[str, Any]) -> None:
    nd = NormalDistribution()
    for case in cpp["distributions"]["normal_pdf"]:
        tolerance.tight(nd(float(case["x"])), float(case["v"]))


def test_normal_cdf_matches_cpp(cpp: dict[str, Any]) -> None:
    cnd = CumulativeNormalDistribution()
    for case in cpp["distributions"]["normal_cdf"]:
        tolerance.tight(cnd(float(case["x"])), float(case["v"]))


def test_inverse_normal_cdf_matches_cpp(cpp: dict[str, Any]) -> None:
    icn = InverseCumulativeNormal()
    for case in cpp["distributions"]["inverse_normal_cdf"]:
        # Acklam-rational tolerance: documented 1.15e-9 relative error; the
        # C++ probe outputs the same approximation, so TIGHT (1e-14 abs)
        # should be achievable except where p sits very close to 0 or 1.
        tolerance.tight(icn(float(case["p"])), float(case["v"]))


# --- non-default parameters ---------------------------------------------


def test_normal_pdf_with_average_and_sigma() -> None:
    # Sanity check: shifted/scaled normal pdf at the mean equals
    # 1/(sigma*sqrt(2*pi)).
    nd = NormalDistribution(average=3.0, sigma=2.0)
    expected = 1.0 / (2.0 * math.sqrt(2.0 * math.pi))
    tolerance.tight(nd(3.0), expected)


def test_normal_pdf_derivative_zero_at_mean() -> None:
    nd = NormalDistribution()
    tolerance.tight(nd.derivative(0.0), 0.0)


def test_cumulative_normal_at_zero_is_half() -> None:
    cnd = CumulativeNormalDistribution()
    tolerance.tight(cnd(0.0), 0.5)


def test_inverse_cumulative_normal_at_half_is_zero() -> None:
    icn = InverseCumulativeNormal()
    tolerance.tight(icn(0.5), 0.0)


def test_inverse_cumulative_normal_rejects_negative() -> None:
    icn = InverseCumulativeNormal()
    with pytest.raises(LibraryException):
        icn(-0.1)


def test_normal_distribution_rejects_zero_sigma() -> None:
    with pytest.raises(LibraryException):
        NormalDistribution(average=0.0, sigma=0.0)
