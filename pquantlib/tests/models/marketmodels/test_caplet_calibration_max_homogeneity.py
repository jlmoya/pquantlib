"""Tests for the W10-C max-homogeneity caplet calibration + sphere-cylinder.

Cross-validates against ``migration-harness/references/cluster/w10c.json``.

C++ parity:
  ql/math/optimization/spherecylinder.{hpp,cpp}
  ql/math/matrixutilities/basisincompleteordered.{hpp,cpp}
  ql/models/marketmodels/models/capletcoterminalmaxhomogeneity.{hpp,cpp}
  @ v1.42.1 (099987f0).

The calibration setup mirrors marketmodel_smmcaplethomocalibration.cpp — the
canonical QuantLib coterminal-swap-market-model caplet calibration test.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.optimization.sphere_cylinder import SphereCylinderOptimizer
from pquantlib.models.marketmodels.correlations.cot_swap_from_fwd_correlation import (
    CotSwapFromFwdCorrelation,
)
from pquantlib.models.marketmodels.correlations.exp_correlations import (
    ExponentialForwardCorrelation,
)
from pquantlib.models.marketmodels.curvestates.lmm_curve_state import LMMCurveState
from pquantlib.models.marketmodels.evolution_description import EvolutionDescription
from pquantlib.models.marketmodels.models.caplet_coterminal_max_homogeneity import (
    CTSMMCapletMaxHomogeneityCalibration,
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


# --- SphereCylinderOptimizer ------------------------------------------------


def test_sphere_cylinder_case1(ref: dict[str, Any]) -> None:
    z = 1.0 / math.sqrt(3.0)
    opt = SphereCylinderOptimizer(1.0, 0.5, 1.5, z, z, z)
    assert opt.is_intersection_non_empty() is bool(ref["sc1_nonempty"])
    y1, y2, y3 = opt.find_closest(100, 1e-8)
    tight(y1, ref["sc1_close_y1"])
    tight(y2, ref["sc1_close_y2"])
    tight(y3, ref["sc1_close_y3"])
    y1, y2, y3 = opt.find_by_projection()
    tight(y1, ref["sc1_proj_y1"])
    tight(y2, ref["sc1_proj_y2"])
    tight(y3, ref["sc1_proj_y3"])


def test_sphere_cylinder_case2(ref: dict[str, Any]) -> None:
    opt = SphereCylinderOptimizer(5.0, 1.0, 1.0, 1.0, 2.0, math.sqrt(20.0))
    assert opt.is_intersection_non_empty() is bool(ref["sc2_nonempty"])
    y1, y2, y3 = opt.find_closest(100, 1e-8)
    # golden-section minimiser — match C++ to the iteration (TIGHT).
    tight(y1, ref["sc2_close_y1"])
    tight(y2, ref["sc2_close_y2"])
    tight(y3, ref["sc2_close_y3"])
    y1, y2, y3 = opt.find_by_projection()
    tight(y1, ref["sc2_proj_y1"])
    tight(y2, ref["sc2_proj_y2"])
    tight(y3, ref["sc2_proj_y3"])


# --- shared calibration fixture (mirror the W10-C probe) -------------------

_CAPLET_VOLS = [
    0.1640, 0.1740, 0.1840, 0.1940, 0.1840,
    0.1740, 0.1640, 0.1540, 0.1440, 0.1340376439125532,
]


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


def test_max_homogeneity_calibration(ref: dict[str, Any]) -> None:
    f = make_fixture()
    evolution = EvolutionDescription(f["rate_times"])
    calibrator = CTSMMCapletMaxHomogeneityCalibration(
        evolution, f["corr"], f["swap_variances"], f["caplet_vols"], f["cs"],
        f["displacement"], 1.0,
    )
    result = calibrator.calibrate(3, 10, 1e-4, 100, 1e-8)
    assert result is bool(ref["mh_result"])
    assert calibrator.failures() == int(ref["mh_failures"])

    # iterative deviation metrics (LOOSE — depends on convergence path)
    loose(calibrator.caplet_rms_error(), ref["mh_caplet_rms"])
    loose(calibrator.caplet_max_error(), ref["mh_caplet_max"])
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
        loose(v, ref[f"mh_caplet_vol_{i}"])
        # 1bp absolute tolerance vs the market caplet vols (C++ test assert)
        assert math.fabs(v - f["caplet_vols"][i]) < 1e-4

    # perfect swaption fit (TIGHT)
    swap_term_cov = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        pr = np.asarray(swap_pseudo_roots[i], dtype=np.float64)
        swap_term_cov = swap_term_cov + pr @ pr.T
        sv = math.sqrt(swap_term_cov[i, i] / rate_times[i])
        tight(sv, ref[f"mh_swaption_vol_{i}"])
        # matches the input swaption vol perfectly (C++ swapTolerance 1e-14)
        exp_swaption_vol = f["swap_variances"][i].total_volatility(i)
        assert math.fabs(sv - exp_swaption_vol) < 1e-14


def test_priority_out_of_range_raises() -> None:
    f = make_fixture()
    evolution = EvolutionDescription(f["rate_times"])
    with pytest.raises(LibraryException):
        CTSMMCapletMaxHomogeneityCalibration(
            evolution, f["corr"], f["swap_variances"], f["caplet_vols"], f["cs"],
            f["displacement"], 1.5,
        )
