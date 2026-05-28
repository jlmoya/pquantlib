"""Tests for the BlackKarasinski log-normal short-rate model.

Cross-validates against C++ probe reference values + sanity invariants.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.methods.lattices.discretized_discount_bond import (
    DiscretizedDiscountBond,
)
from pquantlib.models.shortrate.onefactor.black_karasinski import BlackKarasinski
from pquantlib.models.shortrate.onefactor.one_factor_model import (
    OneFactorModel,
    ShortRateDynamics,
)
from pquantlib.models.shortrate.short_rate_tree import ShortRateTree
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import custom, tight
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.time_grid import TimeGrid


@pytest.fixture(scope="module")
def cluster_refs() -> dict[str, Any]:
    return load_reference("cluster/l5b")


def _curve() -> FlatForward:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    return FlatForward.from_rate(
        eval_date, 0.05, Actual360(), Compounding.Continuous, Frequency.Annual
    )


def test_black_karasinski_construction_defaults() -> None:
    curve = _curve()
    bk = BlackKarasinski(curve)
    # Default a=0.1, sigma=0.1 per the C++ header signature.
    tight(bk.a(), 0.1)
    tight(bk.sigma(), 0.1)


def test_black_karasinski_is_one_factor_model() -> None:
    curve = _curve()
    bk = BlackKarasinski(curve, a=0.1, sigma=0.1)
    # Inheritance: BK subclasses OneFactorModel + TermStructureConsistentModel.
    assert isinstance(bk, OneFactorModel)
    assert bk.term_structure is curve


def test_black_karasinski_dynamics_round_trip() -> None:
    """``short_rate(t, variable(t, r)) == r`` for r > 0 at any grid time.

    The numerical phi fitter sets values at ``grid[0]..grid[n-1]``; we
    probe at those exact times.
    """
    curve = _curve()
    bk = BlackKarasinski(curve, a=0.1, sigma=0.1)
    grid = TimeGrid.regular(end=1.0, steps=10)
    bk.tree(grid)

    dyn: ShortRateDynamics = bk.dynamics()
    # Round-trip at the populated grid times only.
    for i in (1, 5, 9):
        t = grid[i]
        for r in (0.01, 0.05, 0.1):
            x = dyn.variable(t, r)
            r_back = dyn.short_rate(t, x)
            tight(r_back, r)


def test_black_karasinski_tree_returns_short_rate_tree() -> None:
    curve = _curve()
    bk = BlackKarasinski(curve, a=0.1, sigma=0.1)
    grid = TimeGrid.regular(end=2.0, steps=50)
    tree = bk.tree(grid)
    assert isinstance(tree, ShortRateTree)


def test_black_karasinski_tree_reprices_curve_discount(
    cluster_refs: dict[str, Any],
) -> None:
    """Cross-validate against the C++ probe: BK tree fits the input
    curve numerically, so a DiscretizedDiscountBond rolled back to 0
    must equal curve.discount(end) to high precision (the fitter
    drives the residual to Brent's 1e-7 tolerance).
    """
    expected: dict[str, Any] = cluster_refs["bk_zero_bond"]
    curve = _curve()
    bk = BlackKarasinski(curve, a=float(expected["a"]), sigma=float(expected["sigma"]))
    end = float(expected["end"])
    steps = int(expected["steps"])
    grid = TimeGrid.regular(end=end, steps=steps)
    tree = bk.tree(grid)

    bond = DiscretizedDiscountBond()
    bond.initialize(tree, end)
    bond.rollback(0.0)
    pv = bond.present_value()

    # C++ probe: pv_at_zero ≈ 0.9048374180359845; curve_discount_2y ≈
    # 0.9048374180359595. The C++ tree picks up ~3e-14 additional
    # rounding from accumulated state-price arithmetic — same order as
    # our port. We compare directly against the curve.
    # The C++ and Python results both agree with the curve to ~5e-14;
    # we accept a custom 1e-10 tolerance (well inside the Brent
    # tolerance the fitter uses internally — 1e-7).
    custom(
        pv,
        curve.discount(end),
        abs_tol=1e-10,
        rel_tol=1e-10,
        reason="BK tree fitter — curve reproduction within Brent tolerance",
    )


def test_black_karasinski_short_rate_is_positive() -> None:
    """By construction ``r = exp(phi + x)`` is strictly positive even
    when ``x`` swings deep into negative territory.

    We probe at the *penultimate* slice — phi(t) is fitted at every
    interior grid time t_0 .. t_{n-1} (the final t_n = end is never a
    phi-set point in the numerical fit).
    """
    curve = _curve()
    bk = BlackKarasinski(curve, a=0.1, sigma=0.5)  # high vol to stress the tail
    grid = TimeGrid.regular(end=2.0, steps=20)
    tree = bk.tree(grid)

    # Penultimate slice — phi is set at every grid[i] for i < size-1.
    penult = grid.size() - 2
    n_penult = tree.size(penult)
    dyn = bk.dynamics()
    rs = [
        dyn.short_rate(grid[penult], tree.underlying(penult, j))
        for j in range(n_penult)
    ]
    assert min(rs) > 0.0


def test_black_karasinski_arguments_count() -> None:
    """OneFactorModel(2) — exactly two arguments slots: a + sigma."""
    curve = _curve()
    bk = BlackKarasinski(curve)
    assert len(bk.arguments) == 2
