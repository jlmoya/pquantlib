"""Cross-validate GaussianCopula against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/tail.json`` —
``gaussian_copula``. Five correlations from -0.9 to 0.9 and marginals reaching
0.001 and 0.999, so the inverse-normal argument is pushed into the tails where
the previous scipy-backed bivariate CDF used to lose all its digits.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.copulas.gaussian_copula import GaussianCopula
from pquantlib.math.distributions.inverse_cumulative_normal import InverseCumulativeNormal
from pquantlib.math.distributions.normal_distribution import NormalDistribution
from pquantlib.testing import tolerance


def test_matches_cpp_tight(v143_tail: dict[str, Any]) -> None:
    for block in v143_tail["gaussian_copula"]:
        copula = GaussianCopula(block["rho"])
        for x, y, expected in block["cases"]:
            tolerance.tight(copula(x, y), expected, reason=f"rho={block['rho']}, x={x}, y={y}")


def test_independence_at_zero_correlation(v143_tail: dict[str, Any]) -> None:
    """rho = 0 gives the product copula C(x, y) = x*y — a property, not a pin.

    The identity is exact in exact arithmetic, but the copula reaches it
    through Acklam's inverse normal, whose relative error is documented in
    ``normaldistribution.hpp`` as up to 1.15e-9 (the Halley refinement that
    would remove it is ``#ifdef``'d out in C++ and is not compiled). Writing
    ``z_x = Phi^-1(x) (1 + e)`` with ``|e| <= 1.15e-9``:

        C(x, y) = Phi(z_x) Phi(z_y),  dC = y phi(z_x) z_x e + x phi(z_y) z_y e

    so the per-case bound below is derived, not tuned. The extra 1e-15 is one
    ulp of a unit-scale result, which the two CDF evaluations cost regardless.
    """
    block = next(b for b in v143_tail["gaussian_copula"] if b["rho"] == 0.0)
    copula = GaussianCopula(0.0)
    inverse = InverseCumulativeNormal()
    pdf = NormalDistribution()
    acklam_relative_error = 1.15e-9
    for x, y, _ in block["cases"]:
        z_x, z_y = inverse(x), inverse(y)
        bound = (y * pdf(z_x) * abs(z_x) + x * pdf(z_y) * abs(z_y)) * acklam_relative_error + 1e-15
        tolerance.custom(
            copula(x, y),
            x * y,
            abs_tol=bound,
            rel_tol=0.0,
            reason=f"propagated Acklam quantile error at x={x}, y={y}",
        )


def test_arguments_outside_the_unit_square_raise() -> None:
    copula = GaussianCopula(0.5)
    with pytest.raises(LibraryException, match="1st argument"):
        copula(-0.1, 0.5)
    with pytest.raises(LibraryException, match="2nd argument"):
        copula(0.5, 1.1)


def test_rho_outside_the_unit_interval_raises() -> None:
    with pytest.raises(LibraryException, match=r"must be in \[-1,1\]"):
        GaussianCopula(1.5)
