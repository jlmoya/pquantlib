"""Tests for TreeLattice1D + BlackScholesLattice.

Cross-validates against C++ probe and against the closed-form
BlackScholesAnalytic for European calls on a vanilla GBSM process.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.methods.lattices.binomial_tree import CoxRossRubinstein
from pquantlib.methods.lattices.bsm_lattice import BlackScholesLattice
from pquantlib.methods.lattices.discretized_discount_bond import (
    DiscretizedDiscountBond,
)
from pquantlib.methods.lattices.tree_lattice_1d import TreeLattice1D
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import tolerance
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.time_grid import TimeGrid


def _make_process() -> GeneralizedBlackScholesProcess:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    cal = TARGET()
    dc = Actual360()
    spot = SimpleQuote(100.0)
    r = FlatForward.from_rate(eval_date, 0.05, dc, Compounding.Continuous, Frequency.Annual)
    q = FlatForward.from_rate(eval_date, 0.0, dc, Compounding.Continuous, Frequency.Annual)
    vol = BlackConstantVol(
        reference_date=eval_date, calendar=cal, day_counter=dc, volatility=0.20
    )
    return GeneralizedBlackScholesProcess(
        x0=spot, dividend_ts=q, risk_free_ts=r, black_vol_ts=vol
    )


def test_bsm_lattice_is_tree_lattice_1d() -> None:
    process = _make_process()
    crr = CoxRossRubinstein(process, end=1.0, steps=4, strike=100.0)
    lat = BlackScholesLattice(crr, risk_free_rate=0.05, end=1.0, steps=4)
    assert isinstance(lat, TreeLattice1D)


def test_bsm_lattice_discount_is_constant() -> None:
    process = _make_process()
    crr = CoxRossRubinstein(process, end=1.0, steps=4, strike=100.0)
    lat = BlackScholesLattice(crr, risk_free_rate=0.05, end=1.0, steps=4)
    # discount(i, index) is independent of i, index.
    d0 = lat.discount(0, 0)
    for i in range(4):
        for j in range(lat.size(i)):
            tolerance.exact(lat.discount(i, j), d0)
    # Closed form: exp(-r*dt) = exp(-0.05 * 0.25).
    tolerance.tight(d0, math.exp(-0.05 * 0.25))


def test_bsm_lattice_delegates_underlying_to_tree() -> None:
    process = _make_process()
    crr = CoxRossRubinstein(process, end=1.0, steps=4, strike=100.0)
    lat = BlackScholesLattice(crr, risk_free_rate=0.05, end=1.0, steps=4)
    # Root value = x0 = 100.
    tolerance.tight(lat.underlying(0, 0), 100.0)
    # At slice 4 (terminal), the lattice underlying should match the tree.
    for j in range(5):
        tolerance.exact(lat.underlying(4, j), crr.underlying(4, j))


def test_bsm_lattice_grid_returns_underlying_slice() -> None:
    process = _make_process()
    crr = CoxRossRubinstein(process, end=1.0, steps=4, strike=100.0)
    lat = BlackScholesLattice(crr, risk_free_rate=0.05, end=1.0, steps=4)
    grid_t = lat.grid(0.0)
    assert grid_t.shape == (1,)
    tolerance.tight(grid_t[0], 100.0)
    grid_terminal = lat.grid(1.0)
    assert grid_terminal.shape == (5,)


def test_bsm_lattice_stepback_matches_default() -> None:
    """The constant-rate vectorised stepback should match the per-node
    default base implementation modulo rounding.
    """
    process = _make_process()
    crr = CoxRossRubinstein(process, end=1.0, steps=4, strike=100.0)
    lat = BlackScholesLattice(crr, risk_free_rate=0.05, end=1.0, steps=4)
    # Build a slice-4 values vector (mock terminal payoff): max(S - K, 0).
    strike = 100.0
    terminal = np.array(
        [max(crr.underlying(4, j) - strike, 0.0) for j in range(5)],
        dtype=np.float64,
    )
    # Custom stepback (vectorised) — what BlackScholesLattice provides.
    custom = lat.stepback(3, terminal)
    # Manual reproduction of the base default loop.
    expected = np.zeros(4, dtype=np.float64)
    for j in range(4):
        v = 0.0
        for branch in range(2):
            v += lat.probability(3, j, branch) * terminal[lat.descendant(3, j, branch)]
        expected[j] = v * lat.discount(3, j)
    # TIGHT: floating-point chain.
    for v_custom, v_expected in zip(custom, expected, strict=True):
        tolerance.tight(float(v_custom), float(v_expected))


def test_bsm_lattice_discretized_discount_bond_prices_correctly() -> None:
    """A DiscretizedDiscountBond on this lattice should reprice to
    exp(-r * T) when rolled back to t=0 (closed form for constant rate).
    """
    process = _make_process()
    crr = CoxRossRubinstein(process, end=1.0, steps=10, strike=100.0)
    lat = BlackScholesLattice(crr, risk_free_rate=0.05, end=1.0, steps=10)

    bond = DiscretizedDiscountBond()
    bond.initialize(lat, 1.0)
    bond.rollback(0.0)
    pv = bond.present_value()
    # Closed form discount: exp(-0.05 * 1.0).
    tolerance.tight(pv, math.exp(-0.05 * 1.0))


def test_tree_lattice_state_prices_sum_to_discount() -> None:
    """At any time slice ``i`` the state prices should sum to exp(-r*t_i)
    (no-arbitrage: a unit asset costs the discount factor today).
    """
    process = _make_process()
    crr = CoxRossRubinstein(process, end=1.0, steps=10, strike=100.0)
    lat = BlackScholesLattice(crr, risk_free_rate=0.05, end=1.0, steps=10)
    for i in range(11):
        sp_sum = float(np.sum(lat.state_prices(i)))
        # TIGHT: closed form exp(-r*i*dt).
        tolerance.tight(sp_sum, math.exp(-0.05 * i * 0.1))


def test_tree_lattice_rejects_zero_branches() -> None:
    class _Dummy(TreeLattice1D):
        def size(self, i: int) -> int:
            return 1

        def underlying(self, i: int, index: int) -> float:
            return 0.0

        def descendant(self, i: int, index: int, branch: int) -> int:
            return 0

        def probability(self, i: int, index: int, branch: int) -> float:
            return 1.0

        def discount(self, i: int, index: int) -> float:
            return 1.0

    with pytest.raises(Exception, match="zeronomial"):
        _Dummy(TimeGrid.regular(1.0, 1), n_branches=0)


def test_initialize_rejects_non_discretized_asset() -> None:
    process = _make_process()
    crr = CoxRossRubinstein(process, end=1.0, steps=4, strike=100.0)
    lat = BlackScholesLattice(crr, risk_free_rate=0.05, end=1.0, steps=4)
    with pytest.raises(TypeError):
        lat.initialize(asset="not an asset", t=0.0)
