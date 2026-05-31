"""Cross-validate GaussianNonCentralChiSquaredPolynomial + moore_penrose_inverse.

Probe source: migration-harness/cpp/probes/cluster_w6c/probe.cpp
Reference:    migration-harness/references/cluster/w6c.json
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.experimental.math.gaussian_noncentral_chisquared_polynomial import (
    GaussianNonCentralChiSquaredPolynomial,
    GaussNonCentralChiSquaredPolynomial,
)
from pquantlib.experimental.math.moore_penrose_inverse import moore_penrose_inverse
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w6c")


# ---- non-central chi-squared polynomial ----


def test_chisq_moments(cpp_ref: dict[str, Any]) -> None:
    p = GaussNonCentralChiSquaredPolynomial(4.0, 1.0)
    tolerance.tight(p.moment(0), cpp_ref["ncchisq_4_1_moment_0"])
    tolerance.tight(p.moment(1), cpp_ref["ncchisq_4_1_moment_1"])
    tolerance.tight(p.moment(2), cpp_ref["ncchisq_4_1_moment_2"])
    tolerance.tight(p.moment(5), cpp_ref["ncchisq_4_1_moment_5"])


def test_chisq_weight(cpp_ref: dict[str, Any]) -> None:
    p = GaussNonCentralChiSquaredPolynomial(4.0, 1.0)
    tolerance.tight(p.w(3.0), cpp_ref["ncchisq_4_1_w_at_3"])


def test_chisq_recurrence_coefficients(cpp_ref: dict[str, Any]) -> None:
    p = GaussNonCentralChiSquaredPolynomial(4.0, 1.0)
    tolerance.tight(p.alpha(0), cpp_ref["ncchisq_4_1_alpha_0"])
    tolerance.tight(p.alpha(2), cpp_ref["ncchisq_4_1_alpha_2"])
    tolerance.tight(p.beta(1), cpp_ref["ncchisq_4_1_beta_1"])
    tolerance.tight(p.beta(3), cpp_ref["ncchisq_4_1_beta_3"])


def test_chisq_node_sums(cpp_ref: dict[str, Any]) -> None:
    # Golub-Welsch: nodes are the eigenvalues of the symmetric tridiagonal
    # Jacobi matrix built from (alpha_i, sqrt(beta_i)). The sum of nodes is
    # the Gautschi rule-of-sum test (gaussianquadratures.cpp).
    p = GaussNonCentralChiSquaredPolynomial(4.0, 1.0)
    for n in range(4, 10):
        diag = [p.alpha(i) for i in range(n)]
        off = [p.beta(i) ** 0.5 for i in range(1, n)]
        jac = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
        node_sum = float(np.linalg.eigvalsh(jac).sum())
        # C++ uses 1e-5 here (double-precision recurrence vs multiprecision ref).
        tolerance.loose(
            node_sum,
            cpp_ref[f"ncchisq_4_1_nodesum_n{n}"],
            reason="double-precision moment recurrence; C++ test tol is 1e-5.",
        )


def test_chisq_alias() -> None:
    assert GaussNonCentralChiSquaredPolynomial is GaussianNonCentralChiSquaredPolynomial


def test_chisq_moment_out_of_range() -> None:
    p = GaussNonCentralChiSquaredPolynomial(4.0, 1.0)
    with pytest.raises(LibraryException, match="must be <"):
        p.moment(28)


# ---- Moore-Penrose pseudo-inverse ----

_MATLAB_A = [
    [64, 2, 3, 61, 60, 6],
    [9, 55, 54, 12, 13, 51],
    [17, 47, 46, 20, 21, 43],
    [40, 26, 27, 37, 36, 30],
    [32, 34, 35, 29, 28, 38],
    [41, 23, 22, 44, 45, 19],
    [49, 15, 14, 52, 53, 11],
    [8, 58, 59, 5, 4, 62],
]


def test_moore_penrose_minimal_norm_solution(cpp_ref: dict[str, Any]) -> None:
    p = moore_penrose_inverse(_MATLAB_A)
    b = np.full(8, 260.0)
    x = p @ b
    for i in range(6):
        tolerance.loose(
            float(x[i]),
            cpp_ref[f"mpinv_minnorm_x{i}"],
            reason="SVD pseudo-inverse: agreement to ~500*eps (C++ test tol).",
        )


def test_moore_penrose_entries(cpp_ref: dict[str, Any]) -> None:
    p = moore_penrose_inverse(_MATLAB_A)
    tolerance.loose(float(p[0, 0]), cpp_ref["mpinv_P_0_0"])
    tolerance.loose(float(p[2, 5]), cpp_ref["mpinv_P_2_5"])


def test_moore_penrose_pseudoinverse_identity() -> None:
    # The defining Penrose condition: A A+ A == A.
    a = np.asarray(_MATLAB_A, dtype=float)
    ap = moore_penrose_inverse(a)
    recon = a @ ap @ a
    for i in range(a.shape[0]):
        for jcol in range(a.shape[1]):
            tolerance.loose(
                float(recon[i, jcol]),
                float(a[i, jcol]),
                reason="A A+ A == A (defining pseudo-inverse property).",
            )


def test_moore_penrose_matches_numpy_pinv() -> None:
    a = np.asarray(_MATLAB_A, dtype=float)
    ours = moore_penrose_inverse(a)
    ref = np.linalg.pinv(a)
    assert np.allclose(ours, ref, atol=1e-12, rtol=1e-12)
