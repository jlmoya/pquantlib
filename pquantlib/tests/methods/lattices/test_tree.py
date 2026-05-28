"""Tests for the Tree[T] abstract base class.

These tests exercise the abstract contract via a minimal concrete
"toy" trinomial tree — there is no C++ probe involved (the C++ Tree
template is just a CRTP scaffold). The toy tree replicates the
recombining-tree structure used by concrete trinomial trees we will
port later (TrinomialTree in L5-B).
"""

from __future__ import annotations

import pytest

from pquantlib.methods.lattices.tree import Tree
from pquantlib.testing import tolerance


class _ToyTrinomialTree(Tree[float]):
    """Toy 3-branch recombining tree for the abstract-contract tests.

    Layout: at time-slice ``i`` there are ``2*i + 1`` nodes (the
    classical recombining trinomial layout). ``descendant(i, j, b)``
    maps to ``j + b`` so branches 0/1/2 step down/level/up.
    """

    branches: int = 3

    def __init__(self, n_columns: int) -> None:
        super().__init__(columns=n_columns)

    def size(self, i: int) -> int:
        return 2 * i + 1

    def underlying(self, i: int, index: int) -> float:
        # Mid-node at value 0; trinomial spacing of 1.0 per step.
        return float(index - i)

    def descendant(self, i: int, index: int, branch: int) -> int:
        del i
        return index + branch  # branch in {0, 1, 2} -> down/level/up

    def probability(self, i: int, index: int, branch: int) -> float:
        # Uniform 1/3 — toy.
        del i, index, branch
        return 1.0 / 3.0


def test_columns_records_constructor_arg() -> None:
    t = _ToyTrinomialTree(n_columns=10)
    assert t.columns() == 10


def test_size_layout_is_recombining() -> None:
    t = _ToyTrinomialTree(n_columns=5)
    assert t.size(0) == 1
    assert t.size(1) == 3
    assert t.size(2) == 5
    assert t.size(3) == 7


def test_underlying_is_centered() -> None:
    t = _ToyTrinomialTree(n_columns=5)
    # At i=2 there are 5 nodes (index 0..4); underlying is -2..+2.
    assert t.underlying(2, 0) == -2.0
    assert t.underlying(2, 2) == 0.0
    assert t.underlying(2, 4) == 2.0


def test_descendant_branches_map_correctly() -> None:
    t = _ToyTrinomialTree(n_columns=3)
    # From node (1, 1) (the middle node at i=1) branches 0/1/2 should
    # reach nodes 1/2/3 at i=2.
    assert t.descendant(1, 1, 0) == 1
    assert t.descendant(1, 1, 1) == 2
    assert t.descendant(1, 1, 2) == 3


def test_probability_sums_to_one() -> None:
    t = _ToyTrinomialTree(n_columns=3)
    total = sum(t.probability(0, 0, b) for b in range(3))
    # 1/3 + 1/3 + 1/3 is exactly representable up to the same
    # rounding regardless of order; LOOSE matches a ULP of slack.
    tolerance.loose(total, 1.0)


def test_branches_class_attribute() -> None:
    # Concrete trees declare ``branches`` (binomial=2, trinomial=3, etc.).
    assert _ToyTrinomialTree.branches == 3


def test_tree_is_abstract() -> None:
    # The base ``Tree`` cannot be instantiated; all four methods are
    # abstract. pyright cannot statically verify abstract instantiation,
    # so ``type: ignore`` is used. We can't subclass and forget the
    # contract either (see test_subclass_must_implement_all_methods).
    with pytest.raises(TypeError):
        Tree[float](columns=1)  # type: ignore[abstract]


def test_subclass_must_implement_all_methods() -> None:
    # Partial subclass missing ``probability`` must not instantiate.
    class _Bad(Tree[float]):
        branches: int = 1

        def size(self, i: int) -> int:
            return 1

        def underlying(self, i: int, index: int) -> float:
            return 0.0

        def descendant(self, i: int, index: int, branch: int) -> int:
            return 0

        # probability missing on purpose

    with pytest.raises(TypeError):
        _Bad(columns=1)  # type: ignore[abstract]
