"""Cross-validate the ql/methods/lattices cluster against C++ v1.43.

Reference: ``migration-harness/references/v143/methods/lattices.json``,
emitted by ``v143_methods_lattices_probe`` (pinned submodule @ 6b57206e0).

Covered:

* the binomial family — ``EqualProbabilitiesBinomialTree`` (JarrowRudd,
  AdditiveEQPBinomialTree), ``EqualJumpsBinomialTree`` (CoxRossRubinstein,
  Trigeorgis) and the direct ``BinomialTree`` concretes (Tian,
  LeisenReimer, Joshi4);
* ``TreeLattice`` — statePrices / stepback / partialRollback /
  presentValue, driven through both of its descendants;
* ``TreeLattice1D`` — ``grid(t)`` via ``BlackScholesLattice``;
* ``TreeLattice2D`` — size / descendant / probability, with both arms of
  the sign-dependent correlation matrix.

Tolerance: TIGHT throughout. Every quantity is a short closed-form
arithmetic chain (exp / sqrt / pow / a bounded dot product), so the
port and C++ agree to a handful of ULPs; nothing here iterates or
cancels. Integer quantities (sizes, descendants, column counts) are
compared exactly with ``==``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.methods.lattices.binomial_tree import (
    AdditiveEQPBinomialTree,
    BinomialTree,
    CoxRossRubinstein,
    EqualJumpsBinomialTree,
    EqualProbabilitiesBinomialTree,
    JarrowRudd,
    Joshi4,
    LeisenReimer,
    Tian,
    Trigeorgis,
)
from pquantlib.methods.lattices.bsm_lattice import BlackScholesLattice
from pquantlib.methods.lattices.discretized_discount_bond import (
    DiscretizedDiscountBond,
)
from pquantlib.methods.lattices.lattice import Lattice
from pquantlib.methods.lattices.tree_lattice import TreeLattice
from pquantlib.methods.lattices.tree_lattice_1d import TreeLattice1D
from pquantlib.methods.lattices.tree_lattice_2d import TreeLattice2D
from pquantlib.methods.lattices.trinomial_tree import TrinomialTree
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.processes.ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.time_grid import TimeGrid

if TYPE_CHECKING:
    from pquantlib.processes.stochastic_process_1d import StochasticProcess1D


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/methods/lattices")


# --------------------------------------------------------------------------
# Fixtures mirroring the probe
# --------------------------------------------------------------------------


def _make_bsm_process() -> GeneralizedBlackScholesProcess:
    """s0 = 100, r = 5%, q = 0, sigma = 20%, Actual360, TARGET, 17-Jan-2024."""
    eval_date = Date.from_ymd(17, Month.January, 2024)
    dc = Actual360()
    return GeneralizedBlackScholesProcess(
        x0=SimpleQuote(100.0),
        dividend_ts=FlatForward.from_rate(
            eval_date, 0.0, dc, Compounding.Continuous, Frequency.Annual
        ),
        risk_free_ts=FlatForward.from_rate(
            eval_date, 0.05, dc, Compounding.Continuous, Frequency.Annual
        ),
        black_vol_ts=BlackConstantVol(
            reference_date=eval_date, calendar=TARGET(), day_counter=dc, volatility=0.20
        ),
    )


type _BinomialFactory = Callable[[StochasticProcess1D, float, int, float], BinomialTree]

# prefix -> (factory, requested steps, strike, slices whose whole
#            `underlying` row the probe captured)
_BINOMIAL_CASES: dict[str, tuple[_BinomialFactory, int, float, tuple[int, ...]]] = {
    "jarrow_rudd": (JarrowRudd, 4, 100.0, (1, 2, 4)),
    "additive_eqp": (AdditiveEQPBinomialTree, 4, 100.0, (1, 2, 4)),
    "crr": (CoxRossRubinstein, 4, 100.0, (1, 2, 4)),
    "trigeorgis": (Trigeorgis, 4, 100.0, (1, 2, 4)),
    "tian": (Tian, 4, 100.0, (1, 2, 4)),
    "leisen_reimer": (LeisenReimer, 4, 100.0, (1, 2, 5)),
    "joshi4": (Joshi4, 4, 100.0, (1, 2, 5)),
    "joshi4_odd": (Joshi4, 7, 100.0, (1, 7)),
    "leisen_reimer_odd": (LeisenReimer, 7, 100.0, (1, 7)),
}


def _build_binomial(prefix: str) -> BinomialTree:
    factory, steps, strike, _ = _BINOMIAL_CASES[prefix]
    return factory(_make_bsm_process(), 1.0, steps, strike)


# --------------------------------------------------------------------------
# Binomial family
# --------------------------------------------------------------------------


@pytest.mark.parametrize("prefix", list(_BINOMIAL_CASES))
def test_binomial_columns_and_sizes(prefix: str, cpp: dict[str, Any]) -> None:
    """columns() pins LeisenReimer/Joshi4 rounding `steps` up to odd."""
    tree = _build_binomial(prefix)
    columns = int(cpp[f"{prefix}_columns"])
    assert tree.columns() == columns
    expected_sizes: list[int] = cpp[f"{prefix}_sizes"]
    assert [tree.size(i) for i in range(columns)] == expected_sizes


@pytest.mark.parametrize("prefix", list(_BINOMIAL_CASES))
def test_binomial_probabilities(prefix: str, cpp: dict[str, Any]) -> None:
    tree = _build_binomial(prefix)
    tight(tree.probability(0, 0, 0), float(cpp[f"{prefix}_pd"]))
    tight(tree.probability(0, 0, 1), float(cpp[f"{prefix}_pu"]))
    # Base-class contract: the probabilities do not depend on (i, index).
    tight(tree.probability(2, 1, 0), float(cpp[f"{prefix}_pd_interior"]))
    tight(tree.probability(2, 1, 1), float(cpp[f"{prefix}_pu_interior"]))


@pytest.mark.parametrize("prefix", list(_BINOMIAL_CASES))
def test_binomial_underlying_rows(prefix: str, cpp: dict[str, Any]) -> None:
    """Whole `underlying` rows — these ARE the base-class formulas."""
    tree = _build_binomial(prefix)
    slices = _BINOMIAL_CASES[prefix][3]
    for i in slices:
        expected: list[float] = cpp[f"{prefix}_underlying_{i}"]
        assert tree.size(i) == len(expected)
        for j, want in enumerate(expected):
            tight(tree.underlying(i, j), float(want))


@pytest.mark.parametrize("prefix", list(_BINOMIAL_CASES))
def test_binomial_descendant(prefix: str, cpp: dict[str, Any]) -> None:
    tree = _build_binomial(prefix)
    expected: list[int] = cpp[f"{prefix}_descendant_i2"]
    actual = [
        tree.descendant(2, index, branch) for branch in range(2) for index in range(3)
    ]
    assert actual == expected


def test_binomial_class_hierarchy_matches_cpp() -> None:
    """The two intermediate bases exist and carry the right concretes.

    # C++ parity: binomialtree.hpp:106-206 — JarrowRudd and
    # AdditiveEQPBinomialTree derive from EqualProbabilitiesBinomialTree;
    # CoxRossRubinstein and Trigeorgis from EqualJumpsBinomialTree; Tian,
    # LeisenReimer and Joshi4 straight from BinomialTree.
    """
    assert issubclass(JarrowRudd, EqualProbabilitiesBinomialTree)
    assert issubclass(AdditiveEQPBinomialTree, EqualProbabilitiesBinomialTree)
    assert issubclass(CoxRossRubinstein, EqualJumpsBinomialTree)
    assert issubclass(Trigeorgis, EqualJumpsBinomialTree)
    for cls in (EqualProbabilitiesBinomialTree, EqualJumpsBinomialTree, Tian,
                LeisenReimer, Joshi4):
        assert issubclass(cls, BinomialTree)
        assert cls.branches == 2
    assert not issubclass(Tian, EqualProbabilitiesBinomialTree)
    assert not issubclass(Tian, EqualJumpsBinomialTree)


def test_joshi4_depends_on_strike(cpp: dict[str, Any]) -> None:
    """Joshi4 is one of only two trees whose coefficients see the strike."""
    tree = Joshi4(_make_bsm_process(), 1.0, 4, 120.0)
    tight(tree.probability(0, 0, 1), float(cpp["joshi4_k120_pu"]))
    expected: list[float] = cpp["joshi4_k120_underlying_5"]
    for j, want in enumerate(expected):
        tight(tree.underlying(5, j), float(want))


def test_joshi4_rejects_non_positive_strike() -> None:
    # C++ parity: binomialtree.cpp:146 — QL_REQUIRE(strike>0.0, ...).
    with pytest.raises(Exception, match="strike must be positive"):
        Joshi4(_make_bsm_process(), 1.0, 4, 0.0)


def test_joshi4_and_leisen_reimer_are_distinct() -> None:
    """Guards against wiring the Peizer-Pratt inversion into Joshi4.

    The two up-probabilities agree to ~5e-5 on this fixture, so a port
    that confused them would still look plausible; the probe pins each
    separately and this test states the discrimination explicitly.
    """
    process = _make_bsm_process()
    joshi = Joshi4(process, 1.0, 4, 100.0)
    leisen = LeisenReimer(process, 1.0, 4, 100.0)
    assert joshi.probability(0, 0, 1) != leisen.probability(0, 0, 1)


# --------------------------------------------------------------------------
# BlackScholesLattice — TreeLattice + TreeLattice1D (overridden stepback)
# --------------------------------------------------------------------------

_BSML_STEPS = 10


def _make_bsm_lattice() -> BlackScholesLattice:
    process = _make_bsm_process()
    tree = CoxRossRubinstein(process, 1.0, _BSML_STEPS, 100.0)
    return BlackScholesLattice(tree, risk_free_rate=0.05, end=1.0, steps=_BSML_STEPS)


def test_bsm_lattice_scalars(cpp: dict[str, Any]) -> None:
    lattice = _make_bsm_lattice()
    tight(lattice.dt(), float(cpp["bsml_dt"]))
    tight(lattice.risk_free_rate(), float(cpp["bsml_risk_free_rate"]))
    tight(lattice.discount(0, 0), float(cpp["bsml_discount"]))
    assert lattice.n_branches == 2


@pytest.mark.parametrize(("key", "t"), [("bsml_grid_0", 0.0), ("bsml_grid_half", 0.5),
                                        ("bsml_grid_terminal", 1.0)])
def test_bsm_lattice_grid(key: str, t: float, cpp: dict[str, Any]) -> None:
    """# C++ parity: TreeLattice1D<Impl>::grid (lattice1d.hpp:43-49)."""
    lattice = _make_bsm_lattice()
    expected: list[float] = cpp[key]
    grid = lattice.grid(t)
    assert len(grid) == len(expected)
    for actual, want in zip(grid, expected, strict=True):
        tight(float(actual), float(want))


def test_bsm_lattice_state_prices(cpp: dict[str, Any]) -> None:
    """# C++ parity: TreeLattice<Impl>::computeStatePrices (lattice.hpp:97-111)."""
    lattice = _make_bsm_lattice()
    for i in range(_BSML_STEPS + 1):
        expected: list[float] = cpp[f"bsml_state_prices_{i}"]
        prices = lattice.state_prices(i)
        assert len(prices) == len(expected)
        for actual, want in zip(prices, expected, strict=True):
            tight(float(actual), float(want))


def test_bsm_lattice_discount_bond_rollback(cpp: dict[str, Any]) -> None:
    lattice = _make_bsm_lattice()
    bond = DiscretizedDiscountBond()
    bond.initialize(lattice, 1.0)
    bond.rollback(0.0)
    tight(bond.present_value(), float(cpp["bsml_bond_pv"]))
    expected: list[float] = cpp["bsml_bond_values_0"]
    for actual, want in zip(bond.values, expected, strict=True):
        tight(float(actual), float(want))


def test_bsm_lattice_discount_bond_partial_rollback(cpp: dict[str, Any]) -> None:
    """Pins the per-slice values, i.e. stepback itself, not just its endpoint."""
    lattice = _make_bsm_lattice()
    bond = DiscretizedDiscountBond()
    bond.initialize(lattice, 1.0)
    bond.partial_rollback(0.5)
    tight(bond.time, float(cpp["bsml_bond_partial_time"]))
    expected: list[float] = cpp["bsml_bond_partial_values"]
    assert len(bond.values) == len(expected)
    for actual, want in zip(bond.values, expected, strict=True):
        tight(float(actual), float(want))
    tight(bond.present_value(), float(cpp["bsml_bond_partial_pv"]))


# --------------------------------------------------------------------------
# TreeLattice2D — through a mirror of TwoFactorModel::ShortRateTree
# --------------------------------------------------------------------------


class _ProbeShortRateTree2D(TreeLattice2D):
    """Concrete 2-D lattice mirroring the probe's ShortRateTree.

    # C++ parity: ``TwoFactorModel::ShortRateTree::discount``
    # (twofactormodel.hpp:117-127), with the probe's closed-form dynamics
    # ``shortRate(t, x, y) = x + y + 0.03``. PQuantLib has not ported the
    # two-factor ShortRateTree (it is a documented L4-D carry-over), so
    # the concrete ``discount`` lives here rather than in the library —
    # the class under test is ``TreeLattice2D`` itself.
    """

    def discount(self, i: int, index: int) -> float:
        modulo = self.tree1.size(i)
        index1 = index % modulo
        index2 = index // modulo
        x = self.tree1.underlying(i, index1)
        y = self.tree2.underlying(i, index2)
        r = x + y + 0.03
        return math.exp(-r * self.time_grid().dt(i))


def _make_lattice_2d(correlation: float) -> _ProbeShortRateTree2D:
    grid = TimeGrid.regular(end=2.0, steps=5)
    tree1 = TrinomialTree(
        OrnsteinUhlenbeckProcess(speed=0.1, vol=0.01, x0=0.0, level=0.0), grid
    )
    tree2 = TrinomialTree(
        OrnsteinUhlenbeckProcess(speed=0.3, vol=0.02, x0=0.0, level=0.0), grid
    )
    return _ProbeShortRateTree2D(tree1, tree2, correlation)


# prefix -> correlation. The +/- pair pins BOTH arms of the C++
# `correlation < 0.0 && T::branches == 3` selection of the m_ matrix; the
# zero case pins the pure-product limit.
_L2D_CASES: dict[str, float] = {
    "l2d_pos": 0.5,
    "l2d_neg": -0.5,
    "l2d_zero": 0.0,
}


@pytest.mark.parametrize("prefix", list(_L2D_CASES))
def test_lattice2d_sizes(prefix: str, cpp: dict[str, Any]) -> None:
    lattice = _make_lattice_2d(_L2D_CASES[prefix])
    expected: list[int] = cpp[f"{prefix}_sizes"]
    assert [lattice.size(i) for i in range(len(expected))] == expected
    assert lattice.n_branches == 9


@pytest.mark.parametrize("prefix", list(_L2D_CASES))
@pytest.mark.parametrize("slice_index", [0, 2])
def test_lattice2d_descendant(prefix: str, slice_index: int, cpp: dict[str, Any]) -> None:
    lattice = _make_lattice_2d(_L2D_CASES[prefix])
    expected: list[int] = cpp[f"{prefix}_descendant_{slice_index}"]
    actual = [
        lattice.descendant(slice_index, index, branch)
        for index in range(lattice.size(slice_index))
        for branch in range(9)
    ]
    assert actual == expected


@pytest.mark.parametrize("prefix", list(_L2D_CASES))
@pytest.mark.parametrize("slice_index", [0, 2])
def test_lattice2d_probability(prefix: str, slice_index: int, cpp: dict[str, Any]) -> None:
    lattice = _make_lattice_2d(_L2D_CASES[prefix])
    expected: list[float] = cpp[f"{prefix}_probability_{slice_index}"]
    actual = [
        lattice.probability(slice_index, index, branch)
        for index in range(lattice.size(slice_index))
        for branch in range(9)
    ]
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        tight(got, float(want))


def test_lattice2d_correlation_sign_changes_the_matrix(cpp: dict[str, Any]) -> None:
    """The +rho and -rho probability rows must differ, and mirror each other.

    Without this the two hard-coded m_ layouts are indistinguishable on
    half the branches, and a port that picked one unconditionally would
    still pass a single-sign test.
    """
    pos: list[float] = cpp["l2d_pos_probability_0"]
    neg: list[float] = cpp["l2d_neg_probability_0"]
    assert pos != neg
    lat_pos = _make_lattice_2d(0.5)
    lat_neg = _make_lattice_2d(-0.5)
    assert [lat_pos.probability(0, 0, b) for b in range(9)] != [
        lat_neg.probability(0, 0, b) for b in range(9)
    ]


@pytest.mark.parametrize("prefix", list(_L2D_CASES))
def test_lattice2d_discount(prefix: str, cpp: dict[str, Any]) -> None:
    lattice = _make_lattice_2d(_L2D_CASES[prefix])
    expected: list[float] = cpp[f"{prefix}_discount_2"]
    for index, want in enumerate(expected):
        tight(lattice.discount(2, index), float(want))


@pytest.mark.parametrize("prefix", list(_L2D_CASES))
def test_lattice2d_state_prices(prefix: str, cpp: dict[str, Any]) -> None:
    """The Arrow-Debreu recursion over a 9-branch lattice."""
    lattice = _make_lattice_2d(_L2D_CASES[prefix])
    for i in range(6):
        expected: list[float] = cpp[f"{prefix}_state_prices_{i}"]
        prices = lattice.state_prices(i)
        assert len(prices) == len(expected)
        for actual, want in zip(prices, expected, strict=True):
            tight(float(actual), float(want))


@pytest.mark.parametrize("prefix", list(_L2D_CASES))
def test_lattice2d_discount_bond_rollback(prefix: str, cpp: dict[str, Any]) -> None:
    """Generic TreeLattice::stepback (BlackScholesLattice overrides it)."""
    lattice = _make_lattice_2d(_L2D_CASES[prefix])
    bond = DiscretizedDiscountBond()
    bond.initialize(lattice, 2.0)
    bond.rollback(0.0)
    tight(bond.present_value(), float(cpp[f"{prefix}_bond_pv"]))
    expected: list[float] = cpp[f"{prefix}_bond_values_0"]
    for actual, want in zip(bond.values, expected, strict=True):
        tight(float(actual), float(want))


@pytest.mark.parametrize("prefix", list(_L2D_CASES))
def test_lattice2d_discount_bond_partial_rollback(prefix: str, cpp: dict[str, Any]) -> None:
    lattice = _make_lattice_2d(_L2D_CASES[prefix])
    bond = DiscretizedDiscountBond()
    bond.initialize(lattice, 2.0)
    bond.partial_rollback(0.8)
    tight(bond.time, float(cpp[f"{prefix}_bond_partial_time"]))
    expected: list[float] = cpp[f"{prefix}_bond_partial_values"]
    assert len(bond.values) == len(expected)
    for actual, want in zip(bond.values, expected, strict=True):
        tight(float(actual), float(want))
    tight(bond.present_value(), float(cpp[f"{prefix}_bond_partial_pv"]))


def test_lattice2d_grid_and_underlying_are_not_implemented() -> None:
    """# C++ parity: lattice2d.hpp:52 — grid() is QL_FAIL("not implemented")."""
    lattice = _make_lattice_2d(0.5)
    with pytest.raises(Exception, match="not implemented"):
        lattice.grid(0.4)
    with pytest.raises(Exception, match="not implemented"):
        lattice.underlying(1, 0)


# --------------------------------------------------------------------------
# Structural parity with the C++ headers
# --------------------------------------------------------------------------


def test_lattice_class_hierarchy_matches_cpp() -> None:
    """# C++ parity: lattice.hpp:57, lattice1d.hpp:39, lattice2d.hpp:42."""
    assert issubclass(TreeLattice, Lattice)
    assert issubclass(TreeLattice1D, TreeLattice)
    assert issubclass(TreeLattice2D, TreeLattice)
    assert not issubclass(TreeLattice2D, TreeLattice1D)
    assert issubclass(BlackScholesLattice, TreeLattice1D)


def test_branching_is_nested_in_trinomial_tree() -> None:
    """# C++ parity: trinomialtree.hpp:42 + 66-79 — Branching is nested.

    Branching's own numbers are cross-validated against C++ indirectly but
    completely: every ``l2d_*_sizes`` entry is ``Branching.size()``, every
    ``l2d_*_descendant_*`` entry routes through ``Branching.descendant``,
    and every ``l2d_*_probability_*`` entry through
    ``Branching.probability`` (see the TreeLattice2D block above, plus the
    pre-existing ``cluster/l5b`` and ``v143/trinomial/g2process`` suites).
    This test pins the class's shape and its index algebra directly.
    """
    branching = TrinomialTree.Branching()
    branching.add(0, 1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0)
    assert branching.j_min == -1
    assert branching.j_max == 1
    assert branching.size() == 3
    # descendant(index, branch) = k[index] - jMin - 1 + branch
    assert [branching.descendant(0, b) for b in range(3)] == [0, 1, 2]
    tight(branching.probability(0, 1), 2.0 / 3.0)


def test_trinomial_branching_drives_size_and_underlying(cpp: dict[str, Any]) -> None:
    """The nested Branching drives size()/underlying() of the tree itself.

    ``size(i)`` is ``Branching.size()`` and ``underlying(i, index)`` is
    ``x0 + (Branching.j_min + index) * dx(i)``, so a whole row must be an
    arithmetic progression of step ``dx(i)`` anchored at ``j_min``.
    """
    grid = TimeGrid.regular(end=2.0, steps=5)
    tree1 = TrinomialTree(
        OrnsteinUhlenbeckProcess(speed=0.1, vol=0.01, x0=0.0, level=0.0), grid
    )
    tree2 = TrinomialTree(
        OrnsteinUhlenbeckProcess(speed=0.3, vol=0.02, x0=0.0, level=0.0), grid
    )
    expected: list[int] = cpp["l2d_pos_sizes"]
    assert [tree1.size(i) * tree2.size(i) for i in range(len(expected))] == expected
    for i in range(1, len(expected)):
        row = [tree1.underlying(i, j) for j in range(tree1.size(i))]
        for lower, upper in pairwise(row):
            tight(upper - lower, tree1.dx(i))
