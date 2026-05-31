"""Tests for the W9-B market-model drift calculators.

Cross-validates against ``migration-harness/references/cluster/w9b.json``.

C++ parity:
  ql/models/marketmodels/driftcomputation/lmmdriftcalculator.{hpp,cpp}
  ql/models/marketmodels/driftcomputation/lmmnormaldriftcalculator.{hpp,cpp}
  ql/models/marketmodels/driftcomputation/smmdriftcalculator.{hpp,cpp}
  ql/models/marketmodels/driftcomputation/cmsmmdriftcalculator.{hpp,cpp}
  @ v1.42.1 (099987f0).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.matrix import Matrix
from pquantlib.models.marketmodels.curvestates.cm_swap_curve_state import CMSwapCurveState
from pquantlib.models.marketmodels.curvestates.coterminal_swap_curve_state import (
    CoterminalSwapCurveState,
)
from pquantlib.models.marketmodels.curvestates.lmm_curve_state import LMMCurveState
from pquantlib.models.marketmodels.driftcomputation.cms_mm_drift_calculator import (
    CMSMMDriftCalculator,
)
from pquantlib.models.marketmodels.driftcomputation.lmm_drift_calculator import (
    LMMDriftCalculator,
)
from pquantlib.models.marketmodels.driftcomputation.lmm_normal_drift_calculator import (
    LMMNormalDriftCalculator,
)
from pquantlib.models.marketmodels.driftcomputation.smm_drift_calculator import (
    SMMDriftCalculator,
)
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import tight


@pytest.fixture
def ref() -> dict[str, Any]:
    return load_reference("cluster/w9b")


def _rate_times6() -> list[float]:
    return [0.5 * (i + 1) for i in range(6)]


def _fwds5() -> list[float]:
    return [0.04, 0.045, 0.05, 0.055, 0.06]


def _pseudo_root5() -> Matrix:
    """Deterministic full-factor 5x5 pseudo-root (matches the probe)."""
    p: Matrix = np.zeros((5, 5), dtype=np.float64)
    for i in range(5):
        for r in range(i + 1):
            p[i, r] = 0.10 * (1.0 + 0.05 * i) / math.sqrt(i + 1)
    return p


def _pseudo_root5x2() -> Matrix:
    """Deterministic rank-2 5x2 pseudo-root (matches the probe)."""
    p: Matrix = np.zeros((5, 2), dtype=np.float64)
    for i in range(5):
        p[i, 0] = 0.10 * (1.0 + 0.05 * i)
        p[i, 1] = 0.02 * (1.0 - 0.1 * i)
    return p


def _lmm_state() -> LMMCurveState:
    cs = LMMCurveState(_rate_times6())
    cs.set_on_forward_rates(_fwds5())
    return cs


# --- LMMDriftCalculator ------------------------------------------------------


def test_lmm_drift_plain(ref: dict[str, Any]) -> None:
    cs = _lmm_state()
    dc = LMMDriftCalculator(_pseudo_root5(), [0.0] * 5, cs.rate_taus(), 5, 0)
    drifts = [0.0] * 5
    dc.compute_plain(cs, drifts)
    # TIGHT: pure linear algebra (covariance * forward factor).
    tight(drifts[0], ref["lmm_drift_plain_0"])
    tight(drifts[1], ref["lmm_drift_plain_1"])
    tight(drifts[2], ref["lmm_drift_plain_2"])
    tight(drifts[3], ref["lmm_drift_plain_3"])
    tight(drifts[4], ref["lmm_drift_plain_4"])  # terminal numeraire -> 0


def test_lmm_drift_compute_dispatch(ref: dict[str, Any]) -> None:
    cs = _lmm_state()
    dc = LMMDriftCalculator(_pseudo_root5(), [0.0] * 5, cs.rate_taus(), 5, 0)
    drifts = [0.0] * 5
    dc.compute(cs, drifts)  # full factor -> plain
    tight(drifts[0], ref["lmm_drift_compute_0"])
    tight(drifts[4], ref["lmm_drift_compute_4"])
    # full-factor compute() must equal compute_plain().
    drifts_plain = [0.0] * 5
    dc.compute_plain(cs, drifts_plain)
    assert np.allclose(drifts, drifts_plain, atol=1e-15)


def test_lmm_drift_reduced(ref: dict[str, Any]) -> None:
    cs = _lmm_state()
    dc = LMMDriftCalculator(_pseudo_root5(), [0.0] * 5, cs.rate_taus(), 5, 0)
    drifts = [0.0] * 5
    dc.compute_reduced(cs, drifts)
    tight(drifts[0], ref["lmm_drift_reduced_0"])
    tight(drifts[4], ref["lmm_drift_reduced_4"])
    # full-factor: reduced and plain agree (Joshi 2003 consistency).
    drifts_plain = [0.0] * 5
    dc.compute_plain(cs, drifts_plain)
    assert np.allclose(drifts, drifts_plain, atol=1e-12)


def test_lmm_drift_displaced_nonterminal(ref: dict[str, Any]) -> None:
    cs = _lmm_state()
    dc = LMMDriftCalculator(_pseudo_root5(), [0.01] * 5, cs.rate_taus(), 3, 1)
    drifts = [0.0] * 5
    dc.compute_plain(cs, drifts)
    tight(drifts[1], ref["lmm_drift_num3_alive1_1"])
    tight(drifts[2], ref["lmm_drift_num3_alive1_2"])  # numeraire-1 -> 0
    tight(drifts[4], ref["lmm_drift_num3_alive1_4"])


def test_lmm_drift_rank2_reduced(ref: dict[str, Any]) -> None:
    cs = _lmm_state()
    dc = LMMDriftCalculator(_pseudo_root5x2(), [0.0] * 5, cs.rate_taus(), 5, 0)
    drifts = [0.0] * 5
    dc.compute(cs, drifts)  # rank-2 -> reduced
    tight(drifts[0], ref["lmm_drift_rank2_0"])
    tight(drifts[4], ref["lmm_drift_rank2_4"])


def test_lmm_drift_from_plain_fwds_list(ref: dict[str, Any]) -> None:
    # compute() accepts a forward-rate list directly, not only a curve state.
    cs = _lmm_state()
    dc = LMMDriftCalculator(_pseudo_root5(), [0.0] * 5, cs.rate_taus(), 5, 0)
    drifts = [0.0] * 5
    dc.compute(_fwds5(), drifts)
    tight(drifts[0], ref["lmm_drift_plain_0"])


# --- LMMNormalDriftCalculator ------------------------------------------------


def test_lmm_normal_drift_plain(ref: dict[str, Any]) -> None:
    cs = _lmm_state()
    dc = LMMNormalDriftCalculator(_pseudo_root5(), cs.rate_taus(), 5, 0)
    drifts = [0.0] * 5
    dc.compute_plain(cs, drifts)
    tight(drifts[0], ref["lmmn_drift_plain_0"])
    tight(drifts[2], ref["lmmn_drift_plain_2"])
    tight(drifts[4], ref["lmmn_drift_plain_4"])


def test_lmm_normal_drift_reduced(ref: dict[str, Any]) -> None:
    cs = _lmm_state()
    dc = LMMNormalDriftCalculator(_pseudo_root5(), cs.rate_taus(), 5, 0)
    drifts = [0.0] * 5
    dc.compute_reduced(cs, drifts)
    tight(drifts[0], ref["lmmn_drift_reduced_0"])
    tight(drifts[4], ref["lmmn_drift_reduced_4"])


def test_lmm_normal_drift_nonterminal(ref: dict[str, Any]) -> None:
    cs = _lmm_state()
    dc = LMMNormalDriftCalculator(_pseudo_root5(), cs.rate_taus(), 3, 1)
    drifts = [0.0] * 5
    dc.compute_plain(cs, drifts)
    tight(drifts[1], ref["lmmn_drift_num3_alive1_1"])
    tight(drifts[4], ref["lmmn_drift_num3_alive1_4"])


# --- SMMDriftCalculator ------------------------------------------------------


def _coterminal_state() -> CoterminalSwapCurveState:
    lmm = _lmm_state()
    cot_rates = [lmm.coterminal_swap_rate(i) for i in range(5)]
    cs = CoterminalSwapCurveState(_rate_times6())
    cs.set_on_coterminal_swap_rates(cot_rates)
    return cs


def test_smm_drift(ref: dict[str, Any]) -> None:
    cs = _coterminal_state()
    dc = SMMDriftCalculator(_pseudo_root5(), [0.0] * 5, cs.rate_taus(), 5, 0)
    drifts = [0.0] * 5
    dc.compute(cs, drifts)
    tight(drifts[0], ref["smm_drift_0"])
    tight(drifts[1], ref["smm_drift_1"])
    tight(drifts[2], ref["smm_drift_2"])
    tight(drifts[3], ref["smm_drift_3"])
    tight(drifts[4], ref["smm_drift_4"])


def test_smm_drift_displaced(ref: dict[str, Any]) -> None:
    cs = _coterminal_state()
    dc = SMMDriftCalculator(_pseudo_root5(), [0.01] * 5, cs.rate_taus(), 5, 0)
    drifts = [0.0] * 5
    dc.compute(cs, drifts)
    tight(drifts[0], ref["smm_drift_disp_0"])
    tight(drifts[4], ref["smm_drift_disp_4"])


# --- CMSMMDriftCalculator ----------------------------------------------------


def _cm_state(spanning: int = 2) -> CMSwapCurveState:
    lmm = _lmm_state()
    cm_rates = [lmm.cm_swap_rate(i, spanning) for i in range(5)]
    cs = CMSwapCurveState(_rate_times6(), spanning)
    cs.set_on_cm_swap_rates(cm_rates)
    return cs


def test_cmsmm_drift(ref: dict[str, Any]) -> None:
    cs = _cm_state(2)
    dc = CMSMMDriftCalculator(_pseudo_root5(), [0.0] * 5, cs.rate_taus(), 5, 0, 2)
    drifts = [0.0] * 5
    dc.compute(cs, drifts)
    tight(drifts[0], ref["cmsmm_drift_0"])
    tight(drifts[1], ref["cmsmm_drift_1"])
    tight(drifts[2], ref["cmsmm_drift_2"])
    tight(drifts[3], ref["cmsmm_drift_3"])
    tight(drifts[4], ref["cmsmm_drift_4"])


def test_cmsmm_drift_displaced(ref: dict[str, Any]) -> None:
    cs = _cm_state(2)
    dc = CMSMMDriftCalculator(_pseudo_root5(), [0.01] * 5, cs.rate_taus(), 5, 0, 2)
    drifts = [0.0] * 5
    dc.compute(cs, drifts)
    tight(drifts[0], ref["cmsmm_drift_disp_0"])
    tight(drifts[4], ref["cmsmm_drift_disp_4"])


# --- validation --------------------------------------------------------------


def test_drift_calculator_validation() -> None:
    taus = [0.5] * 5
    p = _pseudo_root5()
    with pytest.raises(LibraryException):
        LMMDriftCalculator(p, [0.0] * 4, taus, 5, 0)  # bad displacements
    with pytest.raises(LibraryException):
        LMMDriftCalculator(p, [0.0] * 5, taus, 5, 5)  # alive out of bounds
    with pytest.raises(LibraryException):
        LMMDriftCalculator(p, [0.0] * 5, taus, 0, 1)  # numeraire < alive
