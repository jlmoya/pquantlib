"""Tests for the W10-C alpha-form + periodic caplet calibrations.

Cross-validates against ``migration-harness/references/cluster/w10c.json``.

C++ parity:
  ql/models/marketmodels/models/capletcoterminalalphacalibration.{hpp,cpp}
  ql/models/marketmodels/models/capletcoterminalperiodic.{hpp,cpp}
  @ v1.42.1 (099987f0).

Setup mirrors marketmodel_smmcapletalphacalibration.cpp /
marketmodel_smmcaplethomocalibration.cpp::testPeriodFunction.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pquantlib.models.marketmodels.correlations.cot_swap_from_fwd_correlation import (
    CotSwapFromFwdCorrelation,
)
from pquantlib.models.marketmodels.correlations.exp_correlations import (
    ExponentialForwardCorrelation,
)
from pquantlib.models.marketmodels.curvestates.lmm_curve_state import LMMCurveState
from pquantlib.models.marketmodels.evolution_description import EvolutionDescription
from pquantlib.models.marketmodels.models.caplet_coterminal_alpha_calibration import (
    CTSMMCapletAlphaFormCalibration,
)
from pquantlib.models.marketmodels.models.caplet_coterminal_periodic import (
    caplet_swaption_periodic_calibration,
)
from pquantlib.models.marketmodels.models.cot_swap_to_fwd_adapter import (
    CotSwapToFwdAdapter,
)
from pquantlib.models.marketmodels.models.fwd_period_adapter import FwdPeriodAdapter
from pquantlib.models.marketmodels.models.fwd_to_cot_swap_adapter import (
    FwdToCotSwapAdapter,
)
from pquantlib.models.marketmodels.models.piecewise_constant_abcd_variance import (
    PiecewiseConstantAbcdVariance,
)
from pquantlib.models.marketmodels.models.pseudo_root_facade import PseudoRootFacade
from pquantlib.models.marketmodels.models.volatility_interpolation_specifier_abcd import (
    VolatilityInterpolationSpecifierabcd,
)
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import loose, tight

_CAPLET_VOLS = [
    0.1640, 0.1740, 0.1840, 0.1940, 0.1840,
    0.1740, 0.1640, 0.1540, 0.1440, 0.1340376439125532,
]


@pytest.fixture
def ref() -> dict[str, Any]:
    return load_reference("cluster/w10c")


def make_fixture() -> dict[str, Any]:
    rate_times = [0.5 * (i + 1) for i in range(11)]
    forwards = [0.03 + 0.0025 * i for i in range(10)]
    cs = LMMCurveState(rate_times)
    cs.set_on_forward_rates(forwards)
    fwd_corr = ExponentialForwardCorrelation(rate_times, 0.5, 0.2)
    corr = CotSwapFromFwdCorrelation(fwd_corr, cs, 0.0)
    swap_variances = [
        PiecewiseConstantAbcdVariance(0.0, 0.17, 1.0, 0.10, i, rate_times)
        for i in range(10)
    ]
    return {
        "rate_times": rate_times,
        "cs": cs,
        "corr": corr,
        "swap_variances": swap_variances,
        "caplet_vols": list(_CAPLET_VOLS),
        "displacement": 0.0,
        "number_of_rates": 10,
    }


# --- alpha-form calibration -------------------------------------------------


def test_alpha_form_calibration(ref: dict[str, Any]) -> None:
    f = make_fixture()
    n = f["number_of_rates"]
    evolution = EvolutionDescription(f["rate_times"])
    calibrator = CTSMMCapletAlphaFormCalibration(
        evolution, f["corr"], f["swap_variances"], f["caplet_vols"], f["cs"],
        f["displacement"],
        alpha_initial=[0.0] * n, alpha_max=[1.0] * n, alpha_min=[-1.0] * n,
        maximize_homogeneity=False,
    )
    result = calibrator.calibrate(3, 10, 1e-4)
    assert result is bool(ref["af_cal_result"])
    assert calibrator.failures() == int(ref["af_cal_failures"])
    loose(calibrator.caplet_rms_error(), ref["af_cal_caplet_rms"])
    # alpha() must be available post-calibration
    assert len(calibrator.alpha()) == n

    rate_times = f["rate_times"]
    swap_pseudo_roots = calibrator.swap_pseudo_roots()
    smm = PseudoRootFacade(
        swap_pseudo_roots, rate_times, f["cs"].coterminal_swap_rates(),
        [f["displacement"]] * n,
    )
    flmm = CotSwapToFwdAdapter(smm)
    caplet_tot_cov = flmm.total_covariance(n - 1)
    for i in range(n):
        v = math.sqrt(caplet_tot_cov[i, i] / rate_times[i])
        loose(v, ref[f"af_cal_caplet_vol_{i}"])
        assert math.fabs(v - f["caplet_vols"][i]) < 1e-4

    swap_term_cov = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        pr = np.asarray(swap_pseudo_roots[i], dtype=np.float64)
        swap_term_cov = swap_term_cov + pr @ pr.T
        sv = math.sqrt(swap_term_cov[i, i] / rate_times[i])
        tight(sv, ref[f"af_cal_swaption_vol_{i}"])


# --- periodic calibration ---------------------------------------------------


def test_periodic_calibration(ref: dict[str, Any]) -> None:
    f = make_fixture()
    n = f["number_of_rates"]
    period = 2
    offset = n % period
    number_big_rates = n // period
    rate_times = f["rate_times"]

    evolution = EvolutionDescription(rate_times)
    big_rate_times = [rate_times[i * period + offset] for i in range(number_big_rates + 1)]
    swap_variances = [
        PiecewiseConstantAbcdVariance(0.0, 0.17, 1.0, 0.10, i, big_rate_times)
        for i in range(number_big_rates)
    ]
    variance_interpolator = VolatilityInterpolationSpecifierabcd(
        period, offset, swap_variances, rate_times
    )

    result = caplet_swaption_periodic_calibration(
        evolution, f["corr"], variance_interpolator, f["caplet_vols"], f["cs"],
        f["displacement"], 1.0, 3, period,
        max1d_iterations=100, tolerance1d=1e-8,
        max_unperiodic_iterations=10, tolerance_unperiodic=1e-5,
        max_period_iterations=30, period_tolerance=1e-5,
    )

    assert result.failures == int(ref["pc_failures"])
    assert result.iterations_done == int(ref["pc_iterations_done"])
    loose(result.total_swaption_error, ref["pc_total_swaption_error"])
    for i in range(number_big_rates):
        loose(result.final_scales[i], ref[f"pc_final_scale_{i}"])

    # implied caplet vols
    smm = PseudoRootFacade(
        result.swap_covariance_pseudo_roots, rate_times,
        f["cs"].coterminal_swap_rates(), [f["displacement"]] * n,
    )
    flmm = CotSwapToFwdAdapter(smm)
    caplet_tot_cov = flmm.total_covariance(n - 1)
    for i in range(n):
        v = math.sqrt(caplet_tot_cov[i, i] / rate_times[i])
        loose(v, ref[f"pc_caplet_vol_{i}"])
        assert math.fabs(v - f["caplet_vols"][i]) < 1e-4

    # periodic swaption fit (LOOSE)
    adapted_displacements = [f["displacement"]] * number_big_rates
    adapted_flmm = FwdPeriodAdapter(flmm, period, offset, adapted_displacements)
    adapted_smm = FwdToCotSwapAdapter(adapted_flmm)
    swap_terminal_cov = adapted_smm.total_covariance(adapted_smm.number_of_steps() - 1)
    for i in range(number_big_rates):
        time = adapted_smm.evolution().rate_times()[i]
        sv = math.sqrt(swap_terminal_cov[i, i] / time)
        loose(sv, ref[f"pc_swaption_vol_{i}"])
