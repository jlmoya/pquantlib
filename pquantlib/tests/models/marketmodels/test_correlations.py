"""Tests for the W9-B market-model correlation structures.

Cross-validates against ``migration-harness/references/cluster/w9b.json``.

C++ parity:
  ql/models/marketmodels/correlations/expcorrelations.{hpp,cpp}
  ql/models/marketmodels/correlations/timehomogeneousforwardcorrelation.{hpp,cpp}
  ql/models/marketmodels/correlations/cotswapfromfwdcorrelation.{hpp,cpp}
  @ v1.42.1 (099987f0).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.models.marketmodels.correlations.cot_swap_from_fwd_correlation import (
    CotSwapFromFwdCorrelation,
)
from pquantlib.models.marketmodels.correlations.exp_correlations import (
    ExponentialForwardCorrelation,
    exponential_correlations,
    exponential_forward_correlation,
)
from pquantlib.models.marketmodels.correlations.time_homogeneous_forward_correlation import (
    TimeHomogeneousForwardCorrelation,
)
from pquantlib.models.marketmodels.curvestates.lmm_curve_state import LMMCurveState
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import exact, tight


@pytest.fixture
def ref() -> dict[str, Any]:
    return load_reference("cluster/w9b")


def _rate_times6() -> list[float]:
    return [0.5 * (i + 1) for i in range(6)]


def _fwds5() -> list[float]:
    return [0.04, 0.045, 0.05, 0.055, 0.06]


# --- exponential_forward_correlation -----------------------------------------


def test_exponential_correlations_gamma1(ref: dict[str, Any]) -> None:
    rt = _rate_times6()
    c = exponential_forward_correlation(rt, 0.5, 0.2, 1.0, 0.0)
    exact(float(c.shape[0]), ref["expc_rows"])
    # TIGHT: closed-form exp(-beta|...|) correlation, pure transcendental.
    exact(c[0, 0], ref["expc_0_0"])
    tight(c[0, 1], ref["expc_0_1"])
    tight(c[0, 4], ref["expc_0_4"])
    tight(c[2, 4], ref["expc_2_4"])
    exact(c[4, 4], ref["expc_4_4"])
    # symmetry
    tight(c[4, 0], c[0, 4])


def test_exponential_correlations_eval_time(ref: dict[str, Any]) -> None:
    rt = _rate_times6()
    # t=0.75: rate 0 (t=0.5) is expired -> its row/col is zero.
    ct = exponential_forward_correlation(rt, 0.5, 0.2, 1.0, 0.75)
    tight(ct[1, 1], ref["expct_1_1"])
    tight(ct[1, 4], ref["expct_1_4"])
    # expired rate 0 has zero diagonal (time > rateTimes[0]).
    assert ct[0, 0] == 0.0


def test_exponential_correlations_gamma_ne1(ref: dict[str, Any]) -> None:
    rt = _rate_times6()
    cg = exponential_forward_correlation(rt, 0.5, 0.2, 0.8, 0.0)
    tight(cg[0, 4], ref["expcg_0_4"])
    tight(cg[2, 4], ref["expcg_2_4"])


def test_exponential_correlations_alias() -> None:
    # The C++ spelling alias must resolve to the same function.
    assert exponential_correlations is exponential_forward_correlation


def test_exponential_correlations_validation() -> None:
    rt = _rate_times6()
    with pytest.raises(LibraryException):
        exponential_forward_correlation(rt, 1.5, 0.2)  # L > 1
    with pytest.raises(LibraryException):
        exponential_forward_correlation(rt, 0.5, -0.1)  # beta < 0
    with pytest.raises(LibraryException):
        exponential_forward_correlation(rt, 0.5, 0.2, 1.5)  # gamma > 1


# --- TimeHomogeneousForwardCorrelation ---------------------------------------


def test_evolved_matrices(ref: dict[str, Any]) -> None:
    rt = _rate_times6()
    fwd = exponential_forward_correlation(rt, 0.5, 0.2, 1.0, 0.0)
    evolved = TimeHomogeneousForwardCorrelation.evolved_matrices(fwd)
    exact(float(len(evolved)), ref["thfc_count"])
    # step 0 == fwd
    exact(evolved[0][0, 0], ref["thfc_k0_0_0"])
    tight(evolved[0][0, 4], ref["thfc_k0_0_4"])
    tight(evolved[0][2, 4], ref["thfc_k0_2_4"])
    # step 1: lower-right-shifted
    tight(evolved[1][1, 1], ref["thfc_k1_1_1"])
    tight(evolved[1][1, 4], ref["thfc_k1_1_4"])
    exact(evolved[1][0, 0], ref["thfc_k1_0_0"])  # expired in step 1 -> 0
    # step 2
    tight(evolved[2][2, 4], ref["thfc_k2_2_4"])
    exact(evolved[2][4, 4], ref["thfc_k2_4_4"])


def test_time_homogeneous_forward_correlation_object(ref: dict[str, Any]) -> None:
    rt = _rate_times6()
    fwd = exponential_forward_correlation(rt, 0.5, 0.2, 1.0, 0.0)
    thfc = TimeHomogeneousForwardCorrelation(fwd, rt)
    exact(float(thfc.number_of_rates()), ref["thfc_obj_n"])
    tight(thfc.correlations()[0][0, 4], ref["thfc_obj_corr0_0_4"])
    exact(thfc.times()[0], ref["thfc_obj_times0"])
    # PiecewiseConstantCorrelation.correlation(i) accessor
    tight(thfc.correlation(0)[2, 4], ref["thfc_obj_corr_via_accessor_2_4"])
    # times = rateTimes minus last
    assert thfc.times() == rt[:-1]
    assert thfc.rate_times() == rt


def test_time_homogeneous_dimension_mismatch() -> None:
    rt = _rate_times6()  # n = 5
    bad = np.zeros((4, 4), dtype=np.float64)
    with pytest.raises(LibraryException):
        TimeHomogeneousForwardCorrelation(bad, rt)


# --- ExponentialForwardCorrelation -------------------------------------------


def test_exponential_forward_correlation_gamma1(ref: dict[str, Any]) -> None:
    rt = _rate_times6()
    efc = ExponentialForwardCorrelation(rt, 0.5, 0.2)
    exact(float(efc.number_of_rates()), ref["efc_n"])
    exact(float(len(efc.correlations())), ref["efc_count"])
    tight(efc.correlations()[0][0, 4], ref["efc_k0_0_4"])
    tight(efc.correlations()[1][1, 4], ref["efc_k1_1_4"])
    exact(efc.times()[0], ref["efc_times0"])
    exact(efc.times()[-1], ref["efc_times_back"])
    # gamma=1 path is time-homogeneous: equals evolvedMatrices of the
    # base exp correlation.
    base = exponential_forward_correlation(rt, 0.5, 0.2, 1.0, 0.0)
    evolved = TimeHomogeneousForwardCorrelation.evolved_matrices(base)
    assert np.allclose(efc.correlations()[0], evolved[0])


def test_exponential_forward_correlation_gamma_ne1(ref: dict[str, Any]) -> None:
    rt = _rate_times6()
    efcg = ExponentialForwardCorrelation(rt, 0.5, 0.2, 0.8)
    exact(float(len(efcg.correlations())), ref["efcg_count"])
    # midpoint-integrated matrices (different from gamma=1 path).
    tight(efcg.correlations()[0][0, 4], ref["efcg_k0_0_4"])
    tight(efcg.correlations()[1][1, 4], ref["efcg_k1_1_4"])


def test_exponential_forward_correlation_too_few_rates() -> None:
    with pytest.raises(LibraryException):
        ExponentialForwardCorrelation([0.5, 1.0], 0.5, 0.2)  # only 1 rate


# --- CotSwapFromFwdCorrelation -----------------------------------------------


def test_cot_swap_from_fwd_correlation(ref: dict[str, Any]) -> None:
    rt = _rate_times6()
    cs = LMMCurveState(rt)
    cs.set_on_forward_rates(_fwds5())
    fwd_corr = ExponentialForwardCorrelation(rt, 0.5, 0.2)
    csfc = CotSwapFromFwdCorrelation(fwd_corr, cs, 0.02)
    exact(float(csfc.number_of_rates()), ref["csfc_n"])
    exact(float(len(csfc.correlations())), ref["csfc_count"])
    # TIGHT: Z C Z^T sandwich + correlation extraction.
    tight(csfc.correlations()[0][0, 0], ref["csfc_k0_0_0"])
    tight(csfc.correlations()[0][0, 4], ref["csfc_k0_0_4"])
    tight(csfc.correlations()[0][2, 4], ref["csfc_k0_2_4"])
    tight(csfc.correlations()[0][4, 4], ref["csfc_k0_4_4"])
    # later step: expired-rate zeroing
    tight(csfc.correlations()[2][2, 4], ref["csfc_k2_2_4"])
    exact(csfc.correlations()[2][0, 0], ref["csfc_k2_0_0"])  # expired -> 0
    # delegated times / rate_times
    exact(csfc.times()[-1], ref["csfc_times_back"])
    exact(csfc.rate_times()[-1], ref["csfc_ratetimes_back"])


def test_cot_swap_from_fwd_correlation_mismatch() -> None:
    rt = _rate_times6()
    rt_short = [0.5 * (i + 1) for i in range(5)]  # n = 4
    cs = LMMCurveState(rt_short)
    cs.set_on_forward_rates([0.04, 0.045, 0.05, 0.055])
    fwd_corr = ExponentialForwardCorrelation(rt, 0.5, 0.2)  # n = 5
    with pytest.raises(LibraryException):
        CotSwapFromFwdCorrelation(fwd_corr, cs, 0.0)
