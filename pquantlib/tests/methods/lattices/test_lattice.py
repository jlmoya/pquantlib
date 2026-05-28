"""Tests for the Lattice abstract base class.

A minimal concrete ``_NoopLattice`` lets us exercise the constructor,
the ``time_grid()`` accessor, the ``columns()`` inheritance from
``Tree``, and the abstract-method contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from pquantlib.math.array import Array
from pquantlib.methods.lattices.lattice import Lattice
from pquantlib.methods.lattices.tree import Tree
from pquantlib.time.time_grid import TimeGrid


class _NoopLattice(Lattice):
    """Minimal lattice — every operation is a no-op.

    Useful only as a fixture for the abstract-contract tests.
    """

    branches: int = 2

    def size(self, i: int) -> int:
        return i + 1

    def underlying(self, i: int, index: int) -> float:
        return float(index - i / 2.0)

    def descendant(self, i: int, index: int, branch: int) -> int:
        del i
        return index + branch  # branch 0 = down, 1 = up (binomial)

    def probability(self, i: int, index: int, branch: int) -> float:
        del i, index, branch
        return 0.5

    def initialize(self, asset: object, t: float) -> None:
        del asset, t

    def rollback(self, asset: object, to_t: float) -> None:
        del asset, to_t

    def partial_rollback(self, asset: object, to_t: float) -> None:
        del asset, to_t

    def present_value(self, asset: object) -> float:
        del asset
        return 0.0

    def grid(self, t: float) -> Array:
        del t
        return np.zeros(1, dtype=np.float64)


def _make_grid() -> TimeGrid:
    return TimeGrid.regular(end=1.0, steps=4)


def test_constructor_stores_time_grid() -> None:
    tg = _make_grid()
    lat = _NoopLattice(time_grid=tg)
    assert lat.time_grid() is tg


def test_columns_equals_time_grid_size() -> None:
    tg = _make_grid()
    lat = _NoopLattice(time_grid=tg)
    # ``columns`` is inherited from Tree; the Lattice constructor
    # passes ``time_grid.size()`` through.
    assert lat.columns() == tg.size()


def test_inherits_tree_contract() -> None:
    lat = _NoopLattice(time_grid=_make_grid())
    # Tree-side accessors must work.
    assert lat.size(2) == 3
    assert lat.probability(0, 0, 1) == 0.5
    assert lat.descendant(0, 0, 1) == 1


def test_lattice_is_abstract() -> None:
    # The base ``Lattice`` cannot be instantiated.
    with pytest.raises(TypeError):
        Lattice(time_grid=_make_grid())  # type: ignore[abstract]


def test_isinstance_tree() -> None:
    lat = _NoopLattice(time_grid=_make_grid())
    # Class hierarchy: Lattice -> Tree[float].
    assert isinstance(lat, Tree)


def test_present_value_returns_float() -> None:
    lat = _NoopLattice(time_grid=_make_grid())
    # No-op stub returns 0.0; the typing/contract is what we're
    # exercising.
    pv = lat.present_value(asset=object())
    assert isinstance(pv, float)
    assert pv == 0.0


def test_grid_returns_array() -> None:
    lat = _NoopLattice(time_grid=_make_grid())
    g = lat.grid(0.5)
    assert g.ndim == 1
    assert g.dtype == np.float64
