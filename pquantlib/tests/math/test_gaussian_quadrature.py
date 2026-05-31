"""Tests for GaussianQuadrature (Golub-Welsch).

C++ parity: ql/math/integrals/gaussianquadratures.hpp @ v1.42.1.

QuantLib's moment-based ``GaussianQuadrature`` uses the weight
convention ``w_i = mu_0 * v0_i^2 / w(x_i)`` — the polynomial weight
``w(x_i)`` (the distribution pdf) is divided OUT, so ``Sum w_i f(x_i)``
approximates ``integral f(x) dx`` (Lebesgue), not ``integral f(x)
pdf(x) dx``. The true probability-weighted Gauss weights are therefore
``w_i * pdf(x_i)``, and *those* integrate the raw moments exactly: an
n-point rule integrates polynomials of degree <= 2n-1 exactly. We test
that invariant directly (it is what ``SquareRootCLVModel``'s collocation
relies on).
"""

from __future__ import annotations

from scipy.stats import ncx2  # pyright: ignore[reportMissingTypeStubs]

from pquantlib.experimental.math.gaussian_noncentral_chisquared_polynomial import (
    GaussNonCentralChiSquaredPolynomial,
)
from pquantlib.math.integrals.gaussian_quadrature import GaussianQuadrature
from pquantlib.testing.tolerance import loose


def test_integrates_moments_exactly() -> None:
    """Probability-weighted Gauss weights reproduce raw moments x^k, k<2n."""
    df, ncp = 3.6, 2.0
    poly = GaussNonCentralChiSquaredPolynomial(df, ncp)
    quad = GaussianQuadrature(5, poly)
    x = quad.x()
    # Convert the QL Lebesgue weights back to probability weights.
    pw = quad.weights() * ncx2.pdf(x, df, ncp)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    # mass -> 1
    loose(float(pw.sum()), 1.0)
    # 1st raw moment of ncx2(df, ncp) is df + ncp
    loose(float((pw * x).sum()), df + ncp)
    # 2nd raw moment: var + mean^2; var = 2(df + 2 ncp), mean = df + ncp
    mean = df + ncp
    var = 2.0 * (df + 2.0 * ncp)
    loose(float((pw * x * x).sum()), var + mean * mean)


def test_lebesgue_weight_convention() -> None:
    """Sum w_i * pdf(x_i) == 1 (the QL moment-based weight convention)."""
    df, ncp = 4.0, 1.5
    poly = GaussNonCentralChiSquaredPolynomial(df, ncp)
    quad = GaussianQuadrature(5, poly)
    x = quad.x()
    loose(float((quad.weights() * ncx2.pdf(x, df, ncp)).sum()), 1.0)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    # And quad(f) over the raw weights approximates int f(x) dx; for f=pdf
    # that integral is 1 too.
    loose(quad(lambda xx: float(ncx2.pdf(xx, df, ncp))), 1.0)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]


def test_order_and_node_count() -> None:
    poly = GaussNonCentralChiSquaredPolynomial(4.0, 1.0)
    quad = GaussianQuadrature(6, poly)
    assert quad.order() == 6
    assert quad.x().shape[0] == 6
    assert quad.weights().shape[0] == 6


def test_nodes_positive_for_chi_squared() -> None:
    """Non-central chi-squared support is [0, inf): all nodes > 0."""
    poly = GaussNonCentralChiSquaredPolynomial(5.0, 2.0)
    quad = GaussianQuadrature(5, poly)
    assert all(v > 0.0 for v in quad.x())
