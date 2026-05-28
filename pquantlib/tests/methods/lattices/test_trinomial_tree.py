"""Tests for the TrinomialTree concrete tree.

Cross-validates against C++ probe reference values captured in
``migration-harness/references/cluster/l5b.json``.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from pquantlib.methods.lattices.tree import Tree
from pquantlib.methods.lattices.trinomial_tree import TrinomialTree
from pquantlib.processes.ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess
from pquantlib.testing import tolerance
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.time.time_grid import TimeGrid


@pytest.fixture(scope="module")
def cluster_refs() -> dict[str, Any]:
    return load_reference("cluster/l5b")


def _make_tree() -> TrinomialTree:
    """Match the C++ probe fixture: OU(a=0.1, sigma=0.01), grid(2.0, 5)."""
    process = OrnsteinUhlenbeckProcess(speed=0.1, vol=0.01, x0=0.0, level=0.0)
    grid = TimeGrid.regular(end=2.0, steps=5)
    return TrinomialTree(process, grid)


def test_trinomial_branches_is_3() -> None:
    tree = _make_tree()
    assert tree.branches == 3
    assert isinstance(tree, Tree)


def test_trinomial_columns_equals_grid_size() -> None:
    tree = _make_tree()
    # TimeGrid(end=2.0, steps=5) has 6 grid points; columns() inherits
    # from Tree and stores the time-grid size.
    assert tree.columns() == 6


def test_trinomial_dx_matches_probe(cluster_refs: dict[str, Any]) -> None:
    expected: dict[str, Any] = cluster_refs["trinomial_tree"]
    expected_dx: list[float] = cast(list[float], expected["dx"])
    tree = _make_tree()
    # TIGHT: dx[i] = sigma * sqrt(3 * dt) — closed-form, deterministic.
    for i, exp in enumerate(expected_dx):
        tolerance.tight(tree.dx(i), float(exp))


def test_trinomial_sizes_match_probe(cluster_refs: dict[str, Any]) -> None:
    expected: dict[str, Any] = cluster_refs["trinomial_tree"]
    expected_sizes: list[int] = cast(list[int], expected["sizes"])
    tree = _make_tree()
    # EXACT: sizes are integer node counts derived from branching bounds.
    for i, exp in enumerate(expected_sizes):
        assert tree.size(i) == int(exp), f"size mismatch at slice {i}"


def test_trinomial_underlying_matches_probe(
    cluster_refs: dict[str, Any],
) -> None:
    expected: dict[str, Any] = cluster_refs["trinomial_tree"]
    expected_underlying: list[list[float]] = cast(
        list[list[float]], expected["underlying"]
    )
    tree = _make_tree()
    # TIGHT: x0 + (j_min + index) * dx is a chain of floating-point
    # additions; ~ulps-level precision is realistic.
    for i, slice_expected in enumerate(expected_underlying):
        for j, exp_val in enumerate(slice_expected):
            tolerance.tight(tree.underlying(i, j), float(exp_val))


def test_trinomial_prob_slice0_matches_probe(
    cluster_refs: dict[str, Any],
) -> None:
    expected: dict[str, Any] = cluster_refs["trinomial_tree"]
    expected_probs: list[float] = cast(list[float], expected["prob_slice0"])
    tree = _make_tree()
    # The first branching's probabilities at the root (single node).
    for b, exp_p in enumerate(expected_probs):
        tolerance.tight(tree.probability(0, 0, b), float(exp_p))


def test_trinomial_desc_slice0_matches_probe(
    cluster_refs: dict[str, Any],
) -> None:
    expected: dict[str, Any] = cluster_refs["trinomial_tree"]
    expected_desc: list[int] = cast(list[int], expected["desc_slice0"])
    tree = _make_tree()
    for b, exp_d in enumerate(expected_desc):
        assert tree.descendant(0, 0, b) == int(exp_d)


def test_trinomial_probs_sum_to_one() -> None:
    tree = _make_tree()
    # At every node the three-branch probabilities should sum to 1.0
    # by construction (variance-matching constraint).
    for i in range(tree.columns() - 1):
        for j in range(tree.size(i)):
            p_sum = sum(tree.probability(i, j, b) for b in range(3))
            tolerance.tight(p_sum, 1.0)


def test_trinomial_root_is_x0() -> None:
    tree = _make_tree()
    # Root underlying = x0 = 0.
    tolerance.exact(tree.underlying(0, 0), 0.0)


def test_trinomial_rejects_zero_steps() -> None:
    process = OrnsteinUhlenbeckProcess(speed=0.1, vol=0.01, x0=0.0, level=0.0)
    # A TimeGrid with a single point (regular requires steps > 0).
    # TimeGrid.regular asserts end > 0, so we construct a one-point
    # grid directly via the raw ctor.
    grid = TimeGrid([0.0], [0.0])
    with pytest.raises(Exception, match="null time steps"):
        TrinomialTree(process, grid)


def test_trinomial_is_positive_floor() -> None:
    # When ``is_positive`` is on, the lowest branch is clamped so that
    # ``x0 + (k - 1) * dx[i+1] > 0`` for every parent. The OU process
    # centred at 0 has both signs, so by floor-shifting the centre we
    # should observe larger underlying values relative to the unforced
    # tree.
    process = OrnsteinUhlenbeckProcess(speed=0.1, vol=0.01, x0=0.0, level=0.0)
    grid = TimeGrid.regular(end=2.0, steps=5)
    pos_tree = TrinomialTree(process, grid, is_positive=True)
    # Min underlying at the last slice should be > 0.
    last = grid.size() - 1
    min_val = min(pos_tree.underlying(last, j) for j in range(pos_tree.size(last)))
    assert min_val > 0.0
