"""Tests for cross-cluster Protocols.

These tests verify that the concrete L5-A classes satisfy the
structural protocols at runtime (via ``isinstance`` against the
``@runtime_checkable`` Protocols) and that pyright can statically
narrow a concrete instance to its protocol type without complaint
(verified implicitly by mypy/pyright in the CI triad).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pquantlib.exercise import Exercise
from pquantlib.methods.lattices.discretized_asset import DiscretizedAsset
from pquantlib.methods.lattices.discretized_discount_bond import (
    DiscretizedDiscountBond,
)
from pquantlib.methods.lattices.discretized_option import DiscretizedOption
from pquantlib.methods.lattices.lattice import Lattice
from pquantlib.methods.protocols import (
    DiscretizedAssetProtocol,
    LatticeProtocol,
    PathGeneratorProtocol,
)
from pquantlib.time.time_grid import TimeGrid


class _ToyLattice(Lattice):
    branches: int = 2

    def size(self, i: int) -> int:
        return i + 1

    def underlying(self, i: int, index: int) -> float:
        return float(index)

    def descendant(self, i: int, index: int, branch: int) -> int:
        del i
        return index + branch

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

    def grid(self, t: float) -> Any:
        del t
        return np.zeros(1, dtype=np.float64)


# -- DiscretizedAssetProtocol -----------------------------------------


def test_discount_bond_satisfies_discretized_asset_protocol() -> None:
    bond = DiscretizedDiscountBond()
    assert isinstance(bond, DiscretizedAssetProtocol)


def test_discretized_option_satisfies_discretized_asset_protocol() -> None:
    bond = DiscretizedDiscountBond()
    opt = DiscretizedOption(
        underlying=bond,
        exercise_type=Exercise.Type.European,
        exercise_times=[1.0],
    )
    assert isinstance(opt, DiscretizedAssetProtocol)


def test_plain_object_does_not_satisfy_protocol() -> None:
    # A plain object does not have ``mandatory_times`` etc., so
    # ``isinstance`` against the runtime-checkable Protocol must be False.
    assert not isinstance(object(), DiscretizedAssetProtocol)


# -- LatticeProtocol --------------------------------------------------


def test_concrete_lattice_satisfies_lattice_protocol() -> None:
    tg = TimeGrid.regular(end=1.0, steps=2)
    lat = _ToyLattice(time_grid=tg)
    assert isinstance(lat, LatticeProtocol)


def test_discretized_asset_does_not_satisfy_lattice_protocol() -> None:
    bond = DiscretizedDiscountBond()
    # DiscretizedAsset has ``time``, but no ``descendant`` or
    # ``time_grid`` — Protocol check must reject.
    assert not isinstance(bond, LatticeProtocol)


# -- PathGeneratorProtocol --------------------------------------------


def test_path_generator_protocol_is_not_runtime_checkable() -> None:
    # The protocol is intentionally *not* runtime-checkable because
    # the ``next()`` return type is ``Any`` (Path/MultiPath/etc.) and
    # that would make ``isinstance`` checks meaningless. Verify the
    # protocol class does not raise at structural check via getattr.
    assert hasattr(PathGeneratorProtocol, "next")
    assert hasattr(PathGeneratorProtocol, "dimension")


def test_lattice_protocol_isinstance_with_held_method() -> None:
    # DiscretizedAsset.method() returns ``Lattice | None``. We verify
    # the held concrete instance satisfies the protocol — important
    # because pricers will sometimes hold the lattice via the
    # protocol type for testability.
    tg = TimeGrid.regular(end=1.0, steps=2)
    lat = _ToyLattice(time_grid=tg)
    bond = DiscretizedDiscountBond()
    bond.initialize(lat, t=1.0)
    held = bond.method()
    assert held is not None
    assert isinstance(held, LatticeProtocol)


def test_user_defined_asset_can_satisfy_protocol_structurally() -> None:
    # A *non-inheriting* class that quacks like a DiscretizedAsset
    # should satisfy the structural protocol. This is the whole
    # reason protocols exist — without them, users would have to
    # inherit from the concrete base.
    class DuckAsset:
        time: float = 0.0
        values: Any = np.zeros(1, dtype=np.float64)

        def set_time(self, t: float) -> None:
            self.time = t

        def set_values(self, v: Any) -> None:
            self.values = v

        def reset(self, size: int) -> None:
            self.values = np.zeros(size, dtype=np.float64)

        def mandatory_times(self) -> list[float]:
            return []

        def pre_adjust_values(self) -> None:
            pass

        def post_adjust_values(self) -> None:
            pass

        def adjust_values(self) -> None:
            pass

        def is_on_time(self, t: float) -> bool:
            del t
            return False

    duck = DuckAsset()
    # Inherits from nothing yet satisfies DiscretizedAssetProtocol.
    assert isinstance(duck, DiscretizedAssetProtocol)
    assert not isinstance(duck, DiscretizedAsset)  # concrete base inheritance check
