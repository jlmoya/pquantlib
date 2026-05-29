"""Cross-validate OneFactorGaussianCopula + OneFactorStudentCopula vs C++.

Probe source: migration-harness/cpp/probes/cluster_w3b/probe.cpp
Reference:    migration-harness/references/cluster/w3b.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.experimental.credit.one_factor_copula import (
    OneFactorGaussianCopula,
    OneFactorStudentCopula,
)
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w3b")


# -----------------------------------------------------------------------------
# OneFactorGaussianCopula at rho = 0.25
# -----------------------------------------------------------------------------


def test_gauss_copula_correlation_round_trips(cpp_ref: dict[str, Any]) -> None:
    cop = OneFactorGaussianCopula(0.25)
    tolerance.tight(cop.correlation(), cpp_ref["gauss_copula"]["correlation"])


def test_gauss_copula_density_at_zero_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    cop = OneFactorGaussianCopula(0.25)
    tolerance.tight(cop.density(0.0), cpp_ref["gauss_copula"]["density_at_0"])


def test_gauss_copula_cumulative_z_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    cop = OneFactorGaussianCopula(0.25)
    tolerance.tight(cop.cumulative_z(1.0), cpp_ref["gauss_copula"]["cumZ_at_1"])


def test_gauss_copula_cumulative_y_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    cop = OneFactorGaussianCopula(0.25)
    tolerance.tight(cop.cumulative_y(1.0), cpp_ref["gauss_copula"]["cumY_at_1"])


def test_gauss_copula_inverse_cumulative_y_matches_cpp(
    cpp_ref: dict[str, Any],
) -> None:
    cop = OneFactorGaussianCopula(0.25)
    tolerance.tight(cop.inverse_cumulative_y(0.7), cpp_ref["gauss_copula"]["invY_at_0p7"])


def test_gauss_copula_conditional_probability_matches_cpp(
    cpp_ref: dict[str, Any],
) -> None:
    cop = OneFactorGaussianCopula(0.25)
    ref = cpp_ref["gauss_copula"]
    tolerance.tight(cop.conditional_probability(0.2, 0.0), ref["condProb_p02_m0"])
    tolerance.tight(cop.conditional_probability(0.2, 1.0), ref["condProb_p02_m1"])
    tolerance.tight(
        cop.conditional_probability(0.5, -0.5), ref["condProb_p05_m_neg0p5"]
    )


def test_gauss_copula_integral_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    cop = OneFactorGaussianCopula(0.25)
    # LOOSE: Euler integration over 50 steps achieves ~1e-5 accuracy,
    # not 1e-12. We match the C++ output exactly because both use the
    # identical midpoint rule on the same grid.
    tolerance.tight(cop.integral(0.2), cpp_ref["gauss_copula"]["integral_p02"])


# -----------------------------------------------------------------------------
# OneFactorGaussianCopula at rho = 0.50
# -----------------------------------------------------------------------------


def test_gauss_copula_corr050_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    cop = OneFactorGaussianCopula(0.50)
    ref = cpp_ref["gauss_copula_corr050"]
    tolerance.tight(cop.conditional_probability(0.1, 0.0), ref["condProb_p01_m0"])
    tolerance.tight(cop.conditional_probability(0.3, 1.5), ref["condProb_p03_m1p5"])
    tolerance.tight(cop.integral(0.3), ref["integral_p03"])


# -----------------------------------------------------------------------------
# Vasicek closed-form sanity reference
# -----------------------------------------------------------------------------


def test_gauss_copula_matches_vasicek_closed_form(cpp_ref: dict[str, Any]) -> None:
    """conditional_probability at rho=0.2, p=0.1 vs Vasicek closed form.

    p_hat(m) = Phi( (Phi^-1(p) - sqrt(rho) m) / sqrt(1-rho) )

    # TIGHT: both implementations use the identical scipy.stats.norm CDF
    # so they should be bit-for-bit equivalent.
    """
    ref = cpp_ref["vasicek_ref"]
    cop = OneFactorGaussianCopula(0.20)
    tolerance.tight(cop.conditional_probability(0.10, -2.0), ref["p_hat_m_neg2"])
    tolerance.tight(cop.conditional_probability(0.10, -1.0), ref["p_hat_m_neg1"])
    tolerance.tight(cop.conditional_probability(0.10, 0.0), ref["p_hat_m_0"])
    tolerance.tight(cop.conditional_probability(0.10, 1.0), ref["p_hat_m_1"])
    tolerance.tight(cop.conditional_probability(0.10, 2.0), ref["p_hat_m_2"])


# -----------------------------------------------------------------------------
# OneFactorStudentCopula at df = 10
# -----------------------------------------------------------------------------


def test_student_copula_correlation_round_trips(cpp_ref: dict[str, Any]) -> None:
    cop = OneFactorStudentCopula(0.25, nz=10, nm=10)
    tolerance.tight(cop.correlation(), cpp_ref["stud_copula_df10"]["correlation"])


def test_student_copula_density_at_zero_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    cop = OneFactorStudentCopula(0.25, nz=10, nm=10)
    tolerance.tight(cop.density(0.0), cpp_ref["stud_copula_df10"]["density_at_0"])


def test_student_copula_cumulative_z_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    cop = OneFactorStudentCopula(0.25, nz=10, nm=10)
    tolerance.tight(cop.cumulative_z(1.0), cpp_ref["stud_copula_df10"]["cumZ_at_1"])


def test_student_copula_conditional_probability_matches_cpp(
    cpp_ref: dict[str, Any],
) -> None:
    cop = OneFactorStudentCopula(0.25, nz=10, nm=10)
    ref = cpp_ref["stud_copula_df10"]
    # # CUSTOM 1e-3: scipy.stats.t uses a more accurate cumulative CDF
    # than the C++ ``CumulativeStudentDistribution`` (which approximates
    # the incomplete-beta tail via a recursive series). The Euler-grid
    # F_Y table is built by quadrature using the more-accurate scipy CDF,
    # so the Python F_Y differs from the C++ F_Y at ~1e-3.
    tolerance.custom(
        cop.conditional_probability(0.2, 0.0),
        ref["condProb_p02_m0"],
        abs_tol=1e-3,
        rel_tol=1e-3,
        reason="scipy.stats.t vs C++ recursive Student CDF differ at ~1e-3",
    )
    tolerance.custom(
        cop.conditional_probability(0.2, 1.0),
        ref["condProb_p02_m1"],
        abs_tol=1e-3,
        rel_tol=1e-3,
        reason="scipy.stats.t vs C++ recursive Student CDF differ at ~1e-3",
    )


def test_student_copula_integral_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    cop = OneFactorStudentCopula(0.25, nz=10, nm=10)
    # # CUSTOM 1e-3: see test_student_copula_conditional_probability_matches_cpp.
    tolerance.custom(
        cop.integral(0.2),
        cpp_ref["stud_copula_df10"]["integral_p02"],
        abs_tol=1e-3,
        rel_tol=1e-3,
        reason="scipy.stats.t vs C++ recursive Student CDF differ at ~1e-3",
    )


# -----------------------------------------------------------------------------
# Validation: correlation out of range
# -----------------------------------------------------------------------------


def test_gauss_copula_rejects_out_of_range_correlation() -> None:
    with pytest.raises(LibraryException, match="correlation out of range"):
        OneFactorGaussianCopula(1.5)
    with pytest.raises(LibraryException, match="correlation out of range"):
        OneFactorGaussianCopula(-2.0)


def test_student_copula_rejects_low_df() -> None:
    with pytest.raises(LibraryException, match="nz must be > 2"):
        OneFactorStudentCopula(0.25, nz=2, nm=10)
    with pytest.raises(LibraryException, match="nm must be > 2"):
        OneFactorStudentCopula(0.25, nz=10, nm=2)


# -----------------------------------------------------------------------------
# Tiny prob short-circuits to 0 (mirrors C++ guard)
# -----------------------------------------------------------------------------


def test_gauss_copula_tiny_prob_returns_zero() -> None:
    cop = OneFactorGaussianCopula(0.25)
    assert cop.conditional_probability(1e-15, 0.0) == 0.0


# -----------------------------------------------------------------------------
# Conditional probability vector overload
# -----------------------------------------------------------------------------


def test_gauss_copula_conditional_probability_vec_matches_scalar() -> None:
    cop = OneFactorGaussianCopula(0.25)
    probs = [0.1, 0.2, 0.3, 0.4]
    out = cop.conditional_probability_vec(probs, 0.5)
    for i, p in enumerate(probs):
        tolerance.tight(out[i], cop.conditional_probability(p, 0.5))


# -----------------------------------------------------------------------------
# Update propagation: changing correlation invalidates table (Student case)
# -----------------------------------------------------------------------------


def test_student_copula_correlation_change_invalidates_table() -> None:
    cop = OneFactorStudentCopula(0.25, nz=10, nm=10)
    # Force tabulation
    cop.calculate()
    initial = cop.inverse_cumulative_y(0.5)
    cop.set_correlation(0.50)
    cop.calculate()
    after = cop.inverse_cumulative_y(0.5)
    # Same y -> different inverse since correlation changed
    # Both should be near 0 since the distribution is symmetric so the
    # difference may be small; we instead check the density-table is rebuilt
    # by sampling a more sensitive point.
    initial95 = cop.inverse_cumulative_y(0.95)
    cop.set_correlation(0.25)
    cop.calculate()
    after95 = cop.inverse_cumulative_y(0.95)
    # 0.95 quantile changes when correlation changes
    assert abs(initial95 - after95) > 1e-3
    # symmetric point: also use median
    _ = initial, after  # silence unused warnings


# -----------------------------------------------------------------------------
# check_moments smoke test for Gaussian
# -----------------------------------------------------------------------------


def test_gauss_copula_check_moments_passes() -> None:
    cop = OneFactorGaussianCopula(0.25)
    # Use 1e-2 tolerance; 50-step Euler is too coarse for tighter.
    assert cop.check_moments(tolerance=1e-2) == 0
