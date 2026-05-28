"""Tests for the concrete binomial trees (CRR, JarrowRudd, Tian, LeisenReimer).

Cross-validates against C++ probe reference values captured in
``migration-harness/references/cluster/l5b.json``.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.math.distributions.binomial_distribution import (
    peizer_pratt_method2_inversion,
)
from pquantlib.methods.lattices.binomial_tree import (
    BinomialTree,
    CoxRossRubinstein,
    JarrowRudd,
    LeisenReimer,
    Tian,
)
from pquantlib.methods.lattices.tree import Tree
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import tolerance
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def cluster_refs() -> dict[str, Any]:
    return load_reference("cluster/l5b")


# --- shared fixture: GBSM-like flat process matching the C++ probe --------


def _make_process() -> GeneralizedBlackScholesProcess:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    cal = TARGET()
    dc = Actual360()
    spot = SimpleQuote(100.0)
    r = FlatForward.from_rate(
        eval_date, 0.05, dc, Compounding.Continuous, Frequency.Annual
    )
    q = FlatForward.from_rate(
        eval_date, 0.0, dc, Compounding.Continuous, Frequency.Annual
    )
    vol = BlackConstantVol(reference_date=eval_date, calendar=cal, day_counter=dc, volatility=0.20)
    return GeneralizedBlackScholesProcess(
        x0=spot, dividend_ts=q, risk_free_ts=r, black_vol_ts=vol
    )


# --- CRR ----------------------------------------------------------------


def test_crr_pu_pd_match_probe(cluster_refs: dict[str, Any]) -> None:
    expected: dict[str, Any] = cluster_refs["binomial_crr"]
    process = _make_process()
    crr = CoxRossRubinstein(process, end=1.0, steps=4, strike=100.0)
    # EXACT: pu/pd are deterministic functions of (sigma, drift, dt).
    tolerance.tight(crr.probability(0, 0, 1), float(expected["pu"]))
    tolerance.tight(crr.probability(0, 0, 0), float(expected["pd"]))
    # pu + pd = 1 by construction.
    tolerance.exact(
        crr.probability(0, 0, 0) + crr.probability(0, 0, 1), 1.0
    )


def test_crr_terminal_underlying_matches_probe(
    cluster_refs: dict[str, Any],
) -> None:
    expected: dict[str, Any] = cluster_refs["binomial_crr"]
    expected_terminal: list[float] = cast(list[float], expected["underlying_terminal"])
    process = _make_process()
    crr = CoxRossRubinstein(process, end=1.0, steps=4, strike=100.0)
    # Terminal-slice values cross-validate the underlying() formula.
    # TIGHT: math.exp(j*dx) accumulates a few ULPs.
    for j, expected_val in enumerate(expected_terminal):
        tolerance.tight(crr.underlying(4, j), float(expected_val))


def test_crr_descendant_size_branches() -> None:
    process = _make_process()
    crr = CoxRossRubinstein(process, end=1.0, steps=4, strike=100.0)
    assert crr.branches == 2
    assert crr.columns() == 5  # steps + 1
    assert crr.size(0) == 1
    assert crr.size(4) == 5
    assert crr.descendant(2, 1, 0) == 1
    assert crr.descendant(2, 1, 1) == 2


# --- JarrowRudd ---------------------------------------------------------


def test_jarrow_rudd_equal_probabilities() -> None:
    process = _make_process()
    jr = JarrowRudd(process, end=1.0, steps=4, strike=100.0)
    # Probabilities are 0.5 by construction (equal-probabilities tree).
    tolerance.exact(jr.probability(0, 0, 0), 0.5)
    tolerance.exact(jr.probability(0, 0, 1), 0.5)
    tolerance.exact(jr.probability(3, 2, 1), 0.5)


def test_jarrow_rudd_underlying_centred_on_initial() -> None:
    process = _make_process()
    jr = JarrowRudd(process, end=1.0, steps=4, strike=100.0)
    # Root node is the initial state x0 = 100.
    tolerance.tight(jr.underlying(0, 0), 100.0)
    # The middle node at step 4 (where j = 0) should reflect 4 drift
    # steps without any up-down log shift.
    # j = 2*index - i = 2*2 - 4 = 0 — so underlying = 100 * exp(4*drift).
    drift = jr.drift_per_step
    tolerance.tight(jr.underlying(4, 2), 100.0 * pow(2.718281828459045, 4 * drift))


# --- Tian ----------------------------------------------------------------


def test_tian_pu_pd_sum_to_one() -> None:
    process = _make_process()
    tian = Tian(process, end=1.0, steps=4, strike=100.0)
    pu = tian.probability(0, 0, 1)
    pd = tian.probability(0, 0, 0)
    tolerance.tight(pu + pd, 1.0)
    assert 0.0 <= pu <= 1.0
    assert 0.0 <= pd <= 1.0


def test_tian_underlying_root_is_x0() -> None:
    process = _make_process()
    tian = Tian(process, end=1.0, steps=4, strike=100.0)
    # Root node — both ``up^0`` and ``down^0`` are 1, so x0 * 1 * 1 = x0.
    tolerance.exact(tian.underlying(0, 0), 100.0)


# --- LeisenReimer --------------------------------------------------------


def test_leisen_reimer_pu_pd_match_probe(cluster_refs: dict[str, Any]) -> None:
    expected: dict[str, Any] = cluster_refs["binomial_lr"]
    process = _make_process()
    lr = LeisenReimer(process, end=1.0, steps=4, strike=100.0)
    # TIGHT: pu/pd come from a closed-form PP method-2 inversion of d2.
    tolerance.tight(lr.probability(0, 0, 1), float(expected["pu"]))
    tolerance.tight(lr.probability(0, 0, 0), float(expected["pd"]))


def test_leisen_reimer_forces_odd_steps(
    cluster_refs: dict[str, Any],
) -> None:
    expected: dict[str, Any] = cluster_refs["binomial_lr"]
    process = _make_process()
    # Requested 4 steps; LR forces odd → 5 columns = 5 + 1 = 6.
    lr = LeisenReimer(process, end=1.0, steps=4, strike=100.0)
    assert lr.columns() == int(expected["actual_steps"]) + 1


def test_leisen_reimer_terminal_underlying_matches_probe(
    cluster_refs: dict[str, Any],
) -> None:
    expected: dict[str, Any] = cluster_refs["binomial_lr"]
    expected_terminal: list[float] = cast(list[float], expected["underlying_terminal"])
    actual_steps = int(expected["actual_steps"])
    process = _make_process()
    lr = LeisenReimer(process, end=1.0, steps=4, strike=100.0)
    for j, expected_val in enumerate(expected_terminal):
        tolerance.tight(lr.underlying(actual_steps, j), float(expected_val))


def test_leisen_reimer_negative_strike_rejected() -> None:
    process = _make_process()
    with pytest.raises(Exception, match="strike must be positive"):
        LeisenReimer(process, end=1.0, steps=4, strike=-1.0)


# --- shared inheritance contract ----------------------------------------


def test_all_binomial_trees_share_branches_2() -> None:
    process = _make_process()
    crr = CoxRossRubinstein(process, end=1.0, steps=4, strike=100.0)
    jr = JarrowRudd(process, end=1.0, steps=4, strike=100.0)
    tian = Tian(process, end=1.0, steps=4, strike=100.0)
    lr = LeisenReimer(process, end=1.0, steps=4, strike=100.0)
    for tree in (crr, jr, tian, lr):
        assert isinstance(tree, BinomialTree)
        assert isinstance(tree, Tree)
        assert tree.branches == 2


def test_peizer_pratt_rejects_even_n() -> None:
    with pytest.raises(Exception, match="n must be an odd number"):
        peizer_pratt_method2_inversion(0.5, 4)


def test_peizer_pratt_sanity_at_centre() -> None:
    # At z = 0 the inversion returns 0.5 exactly (the centre of the
    # cumulative).
    tolerance.exact(peizer_pratt_method2_inversion(0.0, 7), 0.5)
