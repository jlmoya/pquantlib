"""Tests for ShortRateTree + ShortRateModel.tree(grid).

Cross-validates against Vasicek/HW closed-form discount_bond by
rolling a DiscretizedDiscountBond back through the lattice and
comparing against the analytical answer.
"""

from __future__ import annotations

import math

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.methods.lattices.discretized_discount_bond import (
    DiscretizedDiscountBond,
)
from pquantlib.methods.lattices.tree_lattice_1d import TreeLattice1D
from pquantlib.models.shortrate.onefactor.cox_ingersoll_ross import (
    CoxIngersollRoss,
)
from pquantlib.models.shortrate.onefactor.extended_cox_ingersoll_ross import (
    ExtendedCoxIngersollRoss,
)
from pquantlib.models.shortrate.onefactor.hull_white import HullWhite
from pquantlib.models.shortrate.onefactor.vasicek import Vasicek
from pquantlib.models.shortrate.short_rate_tree import ShortRateTree
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import tolerance
from pquantlib.testing.tolerance import custom
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.time_grid import TimeGrid


def _flat_curve() -> FlatForward:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    return FlatForward.from_rate(
        eval_date, 0.05, Actual360(), Compounding.Continuous, Frequency.Annual
    )


def test_vasicek_tree_subclasses_tree_lattice() -> None:
    v = Vasicek(r0=0.05, a=0.1, b=0.05, sigma=0.01)
    grid = TimeGrid.regular(end=1.0, steps=10)
    tree = v.tree(grid)
    assert isinstance(tree, ShortRateTree)
    assert isinstance(tree, TreeLattice1D)
    assert tree.n_branches == 3


def test_vasicek_tree_root_discount_matches_analytical() -> None:
    """Vasicek tree's root one-step discount = exp(-r0 * dt).

    Since the OU process state has x0 = r0 - b, dynamics map x -> r,
    and the dynamics' short_rate at the root yields r0.
    """
    v = Vasicek(r0=0.05, a=0.1, b=0.05, sigma=0.01)
    dt = 0.1
    grid = TimeGrid.regular(end=1.0, steps=10)
    tree = v.tree(grid)
    root_disc = tree.discount(0, 0)
    tolerance.tight(root_disc, math.exp(-0.05 * dt))


def test_vasicek_tree_reprices_discount_bond_loose() -> None:
    """Roll a DiscretizedDiscountBond back through the Vasicek tree
    and compare to the closed-form Vasicek discount_bond.

    Vasicek has constant ``r0`` and constant drift parameters; the
    state state-prices summation at the root yields the model-implied
    discount = A(0, T) * exp(-B(0, T) * r0). Discretisation error at
    50 steps is ~1e-3 (LOOSE-level by tier rules).
    """
    v = Vasicek(r0=0.05, a=0.1, b=0.05, sigma=0.01)
    end = 1.0
    steps = 50
    grid = TimeGrid.regular(end=end, steps=steps)
    tree = v.tree(grid)

    bond = DiscretizedDiscountBond()
    bond.initialize(tree, end)
    bond.rollback(0.0)
    pv = bond.present_value()

    expected = v.discount(end)
    # Vasicek tree is centred around x0 = r0 - b = 0 (b=r0=0.05); the
    # trinomial spacing is sigma*sqrt(3*dt) — discretisation error is
    # ~ sigma^2 * dt ~ 1e-6 at 50 steps. LOOSE (1e-8) is too tight; we
    # justify a 1e-5 tolerance for tree convergence at finite N.
    custom(
        pv, expected,
        abs_tol=1e-5,
        rel_tol=1e-5,
        reason="trinomial tree convergence at N=50",
    )


def test_hull_white_tree_reprices_curve_discount() -> None:
    """Hull-White's tree fits the input curve by construction (the OU
    state is centred around phi(t)). Pricing a zero-bond on the
    lattice should match curve.discount(T) closely.
    """
    curve = _flat_curve()
    hw = HullWhite(curve, a=0.1, sigma=0.01)
    end = 1.0
    steps = 50
    grid = TimeGrid.regular(end=end, steps=steps)
    tree = hw.tree(grid)

    bond = DiscretizedDiscountBond()
    bond.initialize(tree, end)
    bond.rollback(0.0)
    pv = bond.present_value()
    expected = curve.discount(end)
    # HW tree convergence at 50 steps is similarly tight (the phi(t)
    # closed-form fits the curve analytically; only the discretisation
    # of the OU state introduces error).
    custom(
        pv, expected,
        abs_tol=1e-5,
        rel_tol=1e-5,
        reason="trinomial tree convergence at N=50",
    )


def test_cir_tree_returns_short_rate_tree() -> None:
    """CIR builds a tree off the CIR process — verifies the OneFactorModel
    inheritance works for non-OU dynamics.
    """
    cir = CoxIngersollRoss(r0=0.05, k=0.1, theta=0.05, sigma=0.05)
    grid = TimeGrid.regular(end=1.0, steps=20)
    tree = cir.tree(grid)
    assert isinstance(tree, ShortRateTree)


def test_extended_cir_tree_returns_short_rate_tree() -> None:
    curve = _flat_curve()
    ecir = ExtendedCoxIngersollRoss(curve, k=0.1, theta=0.05, sigma=0.05, x0=0.05)
    grid = TimeGrid.regular(end=1.0, steps=20)
    tree = ecir.tree(grid)
    assert isinstance(tree, ShortRateTree)


def test_short_rate_tree_spread_shifts_discount() -> None:
    """Setting a spread on the tree should multiply discount factors by
    exp(-spread * dt) per step.
    """
    v = Vasicek(r0=0.05, a=0.1, b=0.05, sigma=0.01)
    grid = TimeGrid.regular(end=1.0, steps=10)
    tree = v.tree(grid)
    d_unspread = tree.discount(0, 0)
    tree.set_spread(0.01)
    d_spread = tree.discount(0, 0)
    # New discount = d_unspread * exp(-0.01 * dt).
    dt = 0.1
    tolerance.tight(d_spread, d_unspread * math.exp(-0.01 * dt))


def test_short_rate_tree_state_prices_at_root() -> None:
    """State prices at the root slice ``0`` are always [1.0]."""
    v = Vasicek(r0=0.05, a=0.1, b=0.05, sigma=0.01)
    grid = TimeGrid.regular(end=1.0, steps=10)
    tree = v.tree(grid)
    sp = tree.state_prices(0)
    assert sp.shape == (1,)
    tolerance.exact(float(sp[0]), 1.0)
