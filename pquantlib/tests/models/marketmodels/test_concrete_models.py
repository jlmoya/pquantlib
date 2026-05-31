"""Tests for the W10-A concrete MarketModel volatility models.

Cross-validates against ``migration-harness/references/cluster/w10a.json``.

C++ parity:
  ql/models/marketmodels/models/flatvol.{hpp,cpp}
  ql/models/marketmodels/models/abcdvol.{hpp,cpp}
  ql/models/marketmodels/models/piecewiseconstantabcdvariance.{hpp,cpp}
  ql/models/marketmodels/models/pseudorootfacade.{hpp,cpp}
  ql/models/marketmodels/models/fwdtocotswapadapter.{hpp,cpp}
  ql/models/marketmodels/models/cotswaptofwdadapter.{hpp,cpp}
  ql/math/matrixutilities/pseudosqrt.{hpp,cpp}
  @ v1.42.1 (099987f0).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.models.marketmodels.correlations.exp_correlations import (
    exponential_correlations,
)
from pquantlib.models.marketmodels.correlations.time_homogeneous_forward_correlation import (
    TimeHomogeneousForwardCorrelation,
)
from pquantlib.models.marketmodels.evolution_description import EvolutionDescription
from pquantlib.models.marketmodels.models.abcd_function import AbcdFunction
from pquantlib.models.marketmodels.models.abcd_vol import AbcdVol
from pquantlib.models.marketmodels.models.flat_vol import FlatVol
from pquantlib.models.marketmodels.models.piecewise_constant_abcd_variance import (
    PiecewiseConstantAbcdVariance,
)
from pquantlib.models.marketmodels.models.pseudo_sqrt import (
    SalvagingAlgorithm,
    rank_reduced_sqrt,
)
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import exact, tight


@pytest.fixture
def ref() -> dict[str, Any]:
    return load_reference("cluster/w10a")


# --- shared test fixtures (mirror the W10-A probe grid) ---------------------


def rate_times6() -> list[float]:
    return [0.5 * (i + 1) for i in range(6)]  # 0.5..3.0


def fwds5() -> list[float]:
    return [0.04, 0.045, 0.05, 0.055, 0.06]


def make_flat_vol(factors: int) -> FlatVol:
    rt = rate_times6()
    evolution = EvolutionDescription(rt)
    fwd_corr = exponential_correlations(rt, 0.5, 0.2, 1.0, 0.0)
    corr = TimeHomogeneousForwardCorrelation(fwd_corr, rt)
    vols = [0.20] * 5
    return FlatVol(vols, corr, evolution, factors, fwds5(), [0.0] * 5)


def make_abcd_vol() -> AbcdVol:
    rt = rate_times6()
    evolution = EvolutionDescription(rt)
    fwd_corr = exponential_correlations(rt, 0.5, 0.2, 1.0, 0.0)
    corr = TimeHomogeneousForwardCorrelation(fwd_corr, rt)
    return AbcdVol(-0.02, 0.5, 1.0, 0.14, [1.0] * 5, corr, evolution, 5, fwds5(), [0.0] * 5)


# --- (a) FlatVol ------------------------------------------------------------


def test_flat_vol_dimensions(ref: dict[str, Any]) -> None:
    fv = make_flat_vol(5)
    exact(float(fv.number_of_rates()), ref["fv_n_rates"])
    exact(float(fv.number_of_factors()), ref["fv_n_factors"])
    exact(float(fv.number_of_steps()), ref["fv_n_steps"])
    pr0 = fv.pseudo_root(0)
    exact(float(pr0.shape[0]), ref["fv_pr0_rows"])
    exact(float(pr0.shape[1]), ref["fv_pr0_cols"])


def test_flat_vol_covariance(ref: dict[str, Any]) -> None:
    # The pseudo-root itself is NOT bit-identical (spectral sign/rotation
    # freedom) but the covariance it reconstructs IS — TIGHT.
    fv = make_flat_vol(5)
    cov0 = fv.covariance(0)
    tight(float(cov0[0, 0]), ref["fv_cov0_0_0"])
    tight(float(cov0[0, 1]), ref["fv_cov0_0_1"])
    tight(float(cov0[4, 4]), ref["fv_cov0_4_4"])
    cov2 = fv.covariance(2)
    tight(float(cov2[2, 2]), ref["fv_cov2_2_2"])
    tight(float(cov2[2, 3]), ref["fv_cov2_2_3"])
    tc = fv.total_covariance(4)
    tight(float(tc[4, 4]), ref["fv_totcov4_4_4"])
    tight(float(tc[0, 4]), ref["fv_totcov4_0_4"])


def test_flat_vol_time_dependent_volatility(ref: dict[str, Any]) -> None:
    fv = make_flat_vol(5)
    tdv = fv.time_dependent_volatility(4)
    tight(tdv[0], ref["fv_tdv4_step0"])
    tight(tdv[4], ref["fv_tdv4_step4"])


def test_flat_vol_rank_reduced_preserves_diagonal(ref: dict[str, Any]) -> None:
    # With 3 factors the pseudo-root has 3 columns but normalizePseudoRoot
    # pins the covariance diagonal exactly.
    fv3 = make_flat_vol(3)
    pr0 = fv3.pseudo_root(0)
    exact(float(pr0.shape[1]), ref["fv3_pr0_cols"])
    cov0 = fv3.covariance(0)
    tight(float(cov0[0, 0]), ref["fv3_cov0_0_0"])
    tight(float(cov0[4, 4]), ref["fv3_cov0_4_4"])


def test_flat_vol_pseudo_root_reconstructs_covariance() -> None:
    # B @ B.T must equal the covariance (full rank) to TIGHT.
    fv = make_flat_vol(5)
    for k in range(fv.number_of_steps()):
        pr = fv.pseudo_root(k)
        reconstructed = pr @ pr.T
        cov = fv.covariance(k)
        assert np.allclose(reconstructed, cov, atol=1e-13, rtol=1e-11)


# --- (a) AbcdVol ------------------------------------------------------------


def test_abcd_vol_dimensions(ref: dict[str, Any]) -> None:
    av = make_abcd_vol()
    exact(float(av.number_of_rates()), ref["av_n_rates"])
    exact(float(av.number_of_steps()), ref["av_n_steps"])
    pr0 = av.pseudo_root(0)
    exact(float(pr0.shape[0]), ref["av_pr0_rows"])
    exact(float(pr0.shape[1]), ref["av_pr0_cols"])


def test_abcd_vol_covariance(ref: dict[str, Any]) -> None:
    av = make_abcd_vol()
    cov0 = av.covariance(0)
    tight(float(cov0[0, 0]), ref["av_cov0_0_0"])
    tight(float(cov0[4, 4]), ref["av_cov0_4_4"])
    cov3 = av.covariance(3)
    tight(float(cov3[4, 4]), ref["av_cov3_4_4"])
    tight(float(cov3[3, 4]), ref["av_cov3_3_4"])
    tc = av.total_covariance(4)
    tight(float(tc[4, 4]), ref["av_totcov4_4_4"])


# --- (a) PiecewiseConstantAbcdVariance --------------------------------------


def test_piecewise_constant_abcd_variance(ref: dict[str, Any]) -> None:
    rt = rate_times6()
    pv = PiecewiseConstantAbcdVariance(-0.02, 0.5, 1.0, 0.14, 4, rt)
    tight(pv.variance(0), ref["pcav_var0"])
    tight(pv.variance(2), ref["pcav_var2"])
    tight(pv.variance(4), ref["pcav_var4"])
    tight(pv.volatility(0), ref["pcav_vol0"])
    tight(pv.volatility(4), ref["pcav_vol4"])
    tight(pv.total_variance(4), ref["pcav_totvar4"])
    tight(pv.total_volatility(4), ref["pcav_totvol4"])


def test_piecewise_constant_abcd_variance_reset2(ref: dict[str, Any]) -> None:
    rt = rate_times6()
    pv = PiecewiseConstantAbcdVariance(-0.02, 0.5, 1.0, 0.14, 2, rt)
    tight(pv.variance(0), ref["pcav2_var0"])
    tight(pv.variance(2), ref["pcav2_var2"])
    tight(pv.total_volatility(2), ref["pcav2_totvol2"])
    # getABCD round trip
    a, b, c, d = pv.get_abcd()
    assert (a, b, c, d) == (-0.02, 0.5, 1.0, 0.14)


def test_piecewise_constant_abcd_variance_invalid_reset() -> None:
    rt = rate_times6()
    with pytest.raises(LibraryException):
        PiecewiseConstantAbcdVariance(-0.02, 0.5, 1.0, 0.14, 5, rt)  # resetIndex == n


# --- AbcdFunction direct (covariance/variance integrals) --------------------


def test_abcd_function_variance_matches_piecewise(ref: dict[str, Any]) -> None:
    # PiecewiseConstantAbcdVariance.variance(0) for reset 4 == AbcdFunction
    # variance integral over [0, rt[0]] for the rt[4]-fixing rate.
    rt = rate_times6()
    abcd = AbcdFunction(-0.02, 0.5, 1.0, 0.14)
    var0 = abcd.variance(0.0, rt[0], rt[4])
    tight(var0, ref["pcav_var0"])


def test_abcd_function_c_zero_branch() -> None:
    # c == 0 reduces to a polynomial primitive; check covariance is finite and
    # equals the brute-force trapezoidal integral of f(T-t)f(S-t).
    abcd = AbcdFunction(0.1, 0.2, 0.0, 0.15)
    t_fix, s = 2.0, 3.0
    analytic = abcd.covariance(0.0, 1.5, t_fix, s)
    grid = np.linspace(0.0, 1.5, 200001)
    integrand = np.array([abcd(t_fix - t) * abcd(s - t) for t in grid])
    numeric = float(np.trapezoid(integrand, grid))
    assert abs(analytic - numeric) < 1e-6


# --- rank_reduced_sqrt direct ----------------------------------------------


def test_rank_reduced_sqrt_full_rank_reconstructs() -> None:
    # A symmetric positive-DEFINITE matrix (full rank, all eigenvalues > 0 so
    # the None-salvaging eigenvalue check passes); full-rank pseudo-root
    # reconstructs it exactly.
    rng = np.random.default_rng(42)
    a = rng.standard_normal((5, 5))
    m = a @ a.T + np.eye(5)  # PD, rank 5
    b = rank_reduced_sqrt(m, 5, 1.0, SalvagingAlgorithm.NONE)
    assert np.allclose(b @ b.T, m, atol=1e-12)
    # diagonal pinned exactly
    for i in range(5):
        assert abs(float((b @ b.T)[i, i]) - float(m[i, i])) < 1e-13


def test_rank_reduced_sqrt_spectral_clips_negatives() -> None:
    # A non-PSD correlation matrix (unit diagonal, but a negative eigenvalue):
    # Spectral salvaging clips it. corr=-0.6 on all off-diagonals of a 3x3 has
    # eigenvalues {1.6, 1.6, -0.2} -> one negative, diagonal stays 1.0.
    m = np.array([[1.0, -0.6, -0.6], [-0.6, 1.0, -0.6], [-0.6, -0.6, 1.0]])
    assert np.linalg.eigvalsh(m).min() < 0.0  # sanity: genuinely non-PSD
    b = rank_reduced_sqrt(m, 3, 1.0, SalvagingAlgorithm.SPECTRAL)
    # reconstructed has non-negative eigenvalues
    eig = np.linalg.eigvalsh(b @ b.T)
    assert eig.min() >= -1e-12
    # diagonal still pinned to 1.0 by normalizePseudoRoot
    for i in range(3):
        assert abs(float((b @ b.T)[i, i]) - 1.0) < 1e-13


def test_rank_reduced_sqrt_none_rejects_negative() -> None:
    m = np.array([[1.0, -0.6, -0.6], [-0.6, 1.0, -0.6], [-0.6, -0.6, 1.0]])
    with pytest.raises(LibraryException):
        rank_reduced_sqrt(m, 3, 1.0, SalvagingAlgorithm.NONE)
