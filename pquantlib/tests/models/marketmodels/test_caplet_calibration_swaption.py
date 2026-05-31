"""Tests for the W10-C CTSMMCapletCalibration base + original calibration.

Cross-validates against ``migration-harness/references/cluster/w10c.json``.

C++ parity:
  ql/models/marketmodels/models/ctsmmcapletcalibration.{hpp,cpp}
  ql/models/marketmodels/models/capletcoterminalswaptioncalibration.{hpp,cpp}
  @ v1.42.1 (099987f0).

Setup mirrors marketmodel_smmcaplethomocalibration.cpp: a 10-rate semiannual
coterminal swap market model with abcd swap variances and an exponential
forward correlation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.models.marketmodels.correlations.cot_swap_from_fwd_correlation import (
    CotSwapFromFwdCorrelation,
)
from pquantlib.models.marketmodels.correlations.exp_correlations import (
    ExponentialForwardCorrelation,
)
from pquantlib.models.marketmodels.curvestates.lmm_curve_state import LMMCurveState
from pquantlib.models.marketmodels.evolution_description import EvolutionDescription
from pquantlib.models.marketmodels.models.caplet_coterminal_swaption_calibration import (
    CTSMMCapletOriginalCalibration,
)
from pquantlib.models.marketmodels.models.cot_swap_to_fwd_adapter import (
    CotSwapToFwdAdapter,
)
from pquantlib.models.marketmodels.models.piecewise_constant_abcd_variance import (
    PiecewiseConstantAbcdVariance,
)
from pquantlib.models.marketmodels.models.pseudo_root_facade import PseudoRootFacade
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import loose, tight


@pytest.fixture
def ref() -> dict[str, Any]:
    return load_reference("cluster/w10c")


# --- shared calibration fixture (mirror the W10-C probe) -------------------

_CAPLET_VOLS = [
    0.1640, 0.1740, 0.1840, 0.1940, 0.1840,
    0.1740, 0.1640, 0.1540, 0.1440, 0.1340376439125532,
]


def make_fixture() -> dict[str, Any]:
    rate_times = [0.5 * (i + 1) for i in range(11)]  # 11 times -> 10 rates
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
        "forwards": forwards,
        "cs": cs,
        "corr": corr,
        "swap_variances": swap_variances,
        "caplet_vols": list(_CAPLET_VOLS),
        "displacement": 0.0,
        "number_of_rates": 10,
    }


def test_original_calibration(ref: dict[str, Any]) -> None:
    f = make_fixture()
    evolution = EvolutionDescription(f["rate_times"])
    alpha = [0.0] * f["number_of_rates"]
    calibrator = CTSMMCapletOriginalCalibration(
        evolution, f["corr"], f["swap_variances"], f["caplet_vols"], f["cs"],
        f["displacement"], alpha, lowest_root=False, use_full_approx=True,
    )
    result = calibrator.calibrate(3, 10, 1e-4)
    assert result is bool(ref["oc_result"])
    assert calibrator.failures() == int(ref["oc_failures"])

    # iterative deviation metrics (LOOSE — depends on convergence path)
    loose(calibrator.caplet_rms_error(), ref["oc_caplet_rms"])
    loose(calibrator.caplet_max_error(), ref["oc_caplet_max"])
    # swaption fit is perfect by construction (errors ~1e-17)
    assert calibrator.swaption_rms_error() < 1e-12
    assert calibrator.swaption_max_error() < 1e-12

    n = f["number_of_rates"]
    rate_times = f["rate_times"]
    swap_pseudo_roots = calibrator.swap_pseudo_roots()

    # implied caplet vols via the cot-swap -> forward adapter
    smm = PseudoRootFacade(
        swap_pseudo_roots, rate_times, f["cs"].coterminal_swap_rates(),
        [f["displacement"]] * n,
    )
    flmm = CotSwapToFwdAdapter(smm)
    caplet_tot_cov = flmm.total_covariance(n - 1)
    for i in range(n):
        v = math.sqrt(caplet_tot_cov[i, i] / rate_times[i])
        loose(v, ref[f"oc_caplet_vol_{i}"])

    # perfect swaption fit: rebuild swap terminal covariance
    swap_term_cov = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        pr = np.asarray(swap_pseudo_roots[i], dtype=np.float64)
        swap_term_cov = swap_term_cov + pr @ pr.T
        sv = math.sqrt(swap_term_cov[i, i] / rate_times[i])
        tight(sv, ref[f"oc_swaption_vol_{i}"])


def test_calibrate_requires_alpha_size() -> None:
    f = make_fixture()
    evolution = EvolutionDescription(f["rate_times"])
    with pytest.raises(LibraryException):
        CTSMMCapletOriginalCalibration(
            evolution, f["corr"], f["swap_variances"], f["caplet_vols"], f["cs"],
            f["displacement"], [0.0] * 3, lowest_root=False, use_full_approx=True,
        )


def test_inspectors_require_calibrated() -> None:
    f = make_fixture()
    evolution = EvolutionDescription(f["rate_times"])
    calibrator = CTSMMCapletOriginalCalibration(
        evolution, f["corr"], f["swap_variances"], f["caplet_vols"], f["cs"],
        f["displacement"], [0.0] * 10, lowest_root=False, use_full_approx=True,
    )
    with pytest.raises(LibraryException):
        calibrator.swap_pseudo_roots()
    with pytest.raises(LibraryException):
        calibrator.failures()
