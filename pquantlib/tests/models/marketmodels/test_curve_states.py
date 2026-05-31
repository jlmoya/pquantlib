"""Tests for the W9-A market-model curve-state geometry + grid utilities.

Cross-validates against ``migration-harness/references/cluster/w9a.json``.

C++ parity:
  ql/models/marketmodels/utilities.{hpp,cpp}
  ql/models/marketmodels/evolutiondescription.{hpp,cpp}
  ql/models/marketmodels/curvestates/lmmcurvestate.{hpp,cpp}
  ql/models/marketmodels/curvestates/cmswapcurvestate.{hpp,cpp}
  ql/models/marketmodels/curvestates/coterminalswapcurvestate.{hpp,cpp}
  @ v1.42.1 (099987f0).
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.models.marketmodels.curvestates.cm_swap_curve_state import CMSwapCurveState
from pquantlib.models.marketmodels.curvestates.coterminal_swap_curve_state import (
    CoterminalSwapCurveState,
)
from pquantlib.models.marketmodels.curvestates.lmm_curve_state import LMMCurveState
from pquantlib.models.marketmodels.evolution_description import (
    EvolutionDescription,
    money_market_measure,
    terminal_measure,
)
from pquantlib.models.marketmodels.utilities import (
    check_increasing_times,
    is_in_subset,
    merge_times,
)
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import exact, tight


@pytest.fixture
def ref() -> dict[str, Any]:
    return load_reference("cluster/w9a")


def _rate_times6() -> list[float]:
    return [0.5 * (i + 1) for i in range(6)]


def _fwds5() -> list[float]:
    return [0.04, 0.045, 0.05, 0.055, 0.06]


# --- utilities ---------------------------------------------------------------


def test_merge_times(ref: dict[str, Any]) -> None:
    times = [[1.0, 2.0, 3.0], [2.0, 4.0], [1.0, 5.0]]
    merged, is_present = merge_times(times)
    # EXACT: deterministic set merge — bit-identical merged grid.
    exact(float(len(merged)), ref["merge_n"])
    exact(merged[0], ref["merge_t0"])
    exact(merged[1], ref["merge_t1"])
    exact(merged[2], ref["merge_t2"])
    exact(merged[3], ref["merge_t3"])
    exact(merged[4], ref["merge_t4"])
    exact(float(sum(is_present[0])), ref["merge_present_row0_sum"])
    exact(float(sum(is_present[1])), ref["merge_present_row1_sum"])
    exact(float(sum(is_present[2])), ref["merge_present_row2_sum"])


def test_is_in_subset(ref: dict[str, Any]) -> None:
    s = [1.0, 2.0, 3.0, 4.0, 5.0]
    sub = is_in_subset(s, [2.0, 4.0])
    exact(float(sum(sub)), ref["subset_sum"])
    exact(1.0 if sub[1] else 0.0, ref["subset_at1"])
    exact(1.0 if sub[3] else 0.0, ref["subset_at3"])


def test_check_increasing_times_rejects_non_increasing() -> None:
    with pytest.raises(LibraryException):
        check_increasing_times([1.0, 1.0, 2.0])
    with pytest.raises(LibraryException):
        check_increasing_times([0.0, 1.0])


# --- EvolutionDescription ----------------------------------------------------


def test_evolution_description(ref: dict[str, Any]) -> None:
    evo = EvolutionDescription(_rate_times6())
    # TIGHT: pure index / grid algebra.
    tight(float(evo.number_of_rates()), ref["evo_n_rates"])
    tight(float(evo.number_of_steps()), ref["evo_n_steps"])
    tight(evo.rate_taus()[0], ref["evo_tau0"])
    tight(evo.rate_taus()[4], ref["evo_tau4"])
    tight(evo.evolution_times()[-1], ref["evo_evoltime_back"])
    far = evo.first_alive_rate()
    tight(float(far[0]), ref["evo_far0"])
    tight(float(far[1]), ref["evo_far1"])
    tight(float(far[2]), ref["evo_far2"])
    tight(float(far[3]), ref["evo_far3"])
    tight(float(far[4]), ref["evo_far4"])
    rr = evo.relevance_rates()
    tight(float(rr[0][0]), ref["evo_relevance0_first"])
    tight(float(rr[0][1]), ref["evo_relevance0_second"])


def test_evolution_description_numeraire_measures() -> None:
    evo = EvolutionDescription(_rate_times6())
    n = len(evo.rate_times()) - 1  # 5
    # terminal measure: last bond (index n) for every step
    assert terminal_measure(evo) == [n] * evo.number_of_steps()
    # money market measure (offset 0): min(j, max) per step where j is the
    # count of rate times strictly less than the step's evolution time.
    # evolution times = {0.5,1,1.5,2,2.5}; rate times = {0.5,...,3.0}, so for
    # each step exactly one fewer rate time is strictly-less → [0,1,2,3,4].
    assert money_market_measure(evo) == [0, 1, 2, 3, 4]


# --- LMMCurveState -----------------------------------------------------------


def test_lmm_curve_state(ref: dict[str, Any]) -> None:
    cs = LMMCurveState(_rate_times6())
    cs.set_on_forward_rates(_fwds5())
    tight(float(cs.number_of_rates()), ref["lmm_n_rates"])
    tight(cs.forward_rate(0), ref["lmm_fwd0"])
    tight(cs.forward_rate(4), ref["lmm_fwd4"])
    tight(cs.discount_ratio(0, 5), ref["lmm_dr_0_5"])
    tight(cs.discount_ratio(2, 5), ref["lmm_dr_2_5"])
    tight(cs.discount_ratio(0, 3), ref["lmm_dr_0_3"])
    tight(cs.coterminal_swap_rate(0), ref["lmm_cot_swap_rate_0"])
    tight(cs.coterminal_swap_rate(2), ref["lmm_cot_swap_rate_2"])
    tight(cs.coterminal_swap_annuity(5, 0), ref["lmm_cot_annuity_num5_0"])
    tight(cs.coterminal_swap_annuity(5, 2), ref["lmm_cot_annuity_num5_2"])
    tight(cs.cm_swap_rate(0, 2), ref["lmm_cm_swap_rate_sp2_0"])
    tight(cs.cm_swap_rate(2, 2), ref["lmm_cm_swap_rate_sp2_2"])
    tight(cs.swap_rate(0, 5), ref["lmm_swap_rate_0_5"])
    tight(cs.swap_rate(1, 4), ref["lmm_swap_rate_1_4"])


def test_lmm_set_on_discount_ratios_roundtrip() -> None:
    cs = LMMCurveState(_rate_times6())
    cs.set_on_forward_rates(_fwds5())
    drs = [cs.discount_ratio(i, 0) for i in range(6)]
    cs2 = LMMCurveState(_rate_times6())
    cs2.set_on_discount_ratios(drs)
    for i in range(5):
        tight(cs2.forward_rate(i), _fwds5()[i])


def test_lmm_clone_independent() -> None:
    cs = LMMCurveState(_rate_times6())
    cs.set_on_forward_rates(_fwds5())
    clone = cs.clone()
    tight(clone.forward_rate(2), cs.forward_rate(2))
    tight(clone.discount_ratio(0, 5), cs.discount_ratio(0, 5))


def test_lmm_uninitialized_raises() -> None:
    cs = LMMCurveState(_rate_times6())
    with pytest.raises(LibraryException):
        cs.forward_rate(0)


# --- CMSwapCurveState --------------------------------------------------------


def test_cm_swap_curve_state(ref: dict[str, Any]) -> None:
    spanning = 2
    lmm = LMMCurveState(_rate_times6())
    lmm.set_on_forward_rates(_fwds5())
    cm_rates = [lmm.cm_swap_rate(i, spanning) for i in range(5)]
    cs = CMSwapCurveState(_rate_times6(), spanning)
    cs.set_on_cm_swap_rates(cm_rates)
    tight(float(cs.number_of_rates()), ref["cm_n_rates"])
    tight(cs.cm_swap_rate(0, spanning), ref["cm_cm_rate_0"])
    tight(cs.cm_swap_rate(2, spanning), ref["cm_cm_rate_2"])
    tight(cs.discount_ratio(0, 5), ref["cm_dr_0_5"])
    tight(cs.discount_ratio(2, 5), ref["cm_dr_2_5"])
    tight(cs.forward_rate(0), ref["cm_fwd0"])
    tight(cs.coterminal_swap_rate(0), ref["cm_cot_swap_rate_0"])


# --- CoterminalSwapCurveState ------------------------------------------------


def test_coterminal_swap_curve_state(ref: dict[str, Any]) -> None:
    lmm = LMMCurveState(_rate_times6())
    lmm.set_on_forward_rates(_fwds5())
    cot_rates = [lmm.coterminal_swap_rate(i) for i in range(5)]
    cs = CoterminalSwapCurveState(_rate_times6())
    cs.set_on_coterminal_swap_rates(cot_rates)
    tight(float(cs.number_of_rates()), ref["cot_n_rates"])
    tight(cs.coterminal_swap_rate(0), ref["cot_cot_rate_0"])
    tight(cs.coterminal_swap_rate(2), ref["cot_cot_rate_2"])
    tight(cs.discount_ratio(0, 5), ref["cot_dr_0_5"])
    tight(cs.discount_ratio(2, 5), ref["cot_dr_2_5"])
    tight(cs.forward_rate(0), ref["cot_fwd0"])
    tight(cs.forward_rate(4), ref["cot_fwd4"])
    tight(cs.coterminal_swap_annuity(5, 0), ref["cot_annuity_num5_0"])
