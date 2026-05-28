"""Tests for the DiscretizedAsset hierarchy.

No C++ probe — these are scaffolding classes whose contract is
fully specified by the lattice protocol. We exercise:

- DiscretizedAsset abstract base via a toy concrete subclass.
- DiscretizedDiscountBond (concrete, trivial).
- DiscretizedOption (concrete; covers all three exercise styles +
  the underlying-rollback handshake).

Toy ``_RecordingLattice`` captures the sequence of ``initialize`` /
``rollback`` / ``partial_rollback`` / ``present_value`` calls so we
can assert that DiscretizedOption.postAdjustValuesImpl drives the
underlying correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.exercise import Exercise
from pquantlib.math.array import Array
from pquantlib.methods.lattices.discretized_asset import DiscretizedAsset
from pquantlib.methods.lattices.discretized_discount_bond import (
    DiscretizedDiscountBond,
)
from pquantlib.methods.lattices.discretized_option import DiscretizedOption
from pquantlib.methods.lattices.lattice import Lattice
from pquantlib.testing import tolerance
from pquantlib.time.time_grid import TimeGrid


class _RecordingLattice(Lattice):
    """Toy Lattice that records its delegate calls for inspection."""

    branches: int = 2

    def __init__(self, time_grid: TimeGrid) -> None:
        super().__init__(time_grid=time_grid)
        self.events: list[tuple[str, float]] = []

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
        # Cast to DiscretizedAsset to call its low-level interface.
        assert isinstance(asset, DiscretizedAsset)
        self.events.append(("initialize", t))
        asset.set_time(t)
        asset.reset(self.size(self.time_grid().index(t)))

    def rollback(self, asset: object, to_t: float) -> None:
        assert isinstance(asset, DiscretizedAsset)
        self.events.append(("rollback", to_t))
        self.partial_rollback(asset, to_t)
        asset.adjust_values()

    def partial_rollback(self, asset: object, to_t: float) -> None:
        assert isinstance(asset, DiscretizedAsset)
        self.events.append(("partial_rollback", to_t))
        asset.set_time(to_t)

    def present_value(self, asset: object) -> float:
        assert isinstance(asset, DiscretizedAsset)
        self.events.append(("present_value", asset.time))
        # PV = mean of values (toy).
        return float(np.mean(asset.values)) if asset.values.size > 0 else 0.0

    def grid(self, t: float) -> Array:
        del t
        return np.zeros(1, dtype=np.float64)


class _ToyAsset(DiscretizedAsset):
    """Minimal DiscretizedAsset for exercising the base contract."""

    def __init__(self) -> None:
        super().__init__()
        self.pre_adjust_calls: int = 0
        self.post_adjust_calls: int = 0

    def reset(self, size: int) -> None:
        self._values = np.full(size, 2.5, dtype=np.float64)

    def mandatory_times(self) -> list[float]:
        return [1.0]

    def _pre_adjust_values_impl(self) -> None:
        self.pre_adjust_calls += 1

    def _post_adjust_values_impl(self) -> None:
        self.post_adjust_calls += 1


# -- DiscretizedAsset --------------------------------------------------


def test_initialize_sets_method_and_delegates() -> None:
    tg = TimeGrid.regular(end=1.0, steps=2)
    lat = _RecordingLattice(time_grid=tg)
    asset = _ToyAsset()
    asset.initialize(lat, t=1.0)
    # The recording lattice resets the asset to size = tg.index(1.0) + 1 = 3.
    assert asset.method() is lat
    assert asset.values.shape == (3,)
    assert asset.time == 1.0
    assert lat.events == [("initialize", 1.0)]


def test_rollback_and_partial_rollback_delegate() -> None:
    tg = TimeGrid.regular(end=1.0, steps=4)
    lat = _RecordingLattice(time_grid=tg)
    asset = _ToyAsset()
    asset.initialize(lat, t=1.0)
    asset.rollback(0.5)
    asset.partial_rollback(0.25)
    kinds = [e[0] for e in lat.events]
    # initialize -> rollback (which calls partial_rollback) -> partial_rollback
    assert "initialize" in kinds
    assert kinds.count("partial_rollback") == 2  # one nested inside rollback
    assert kinds.count("rollback") == 1


def test_present_value_delegates() -> None:
    tg = TimeGrid.regular(end=1.0, steps=2)
    lat = _RecordingLattice(time_grid=tg)
    asset = _ToyAsset()
    asset.initialize(lat, t=1.0)
    pv = asset.present_value()
    tolerance.tight(pv, 2.5)  # toy PV = mean of values = 2.5
    assert lat.events[-1] == ("present_value", 1.0)


def test_present_value_without_method_raises() -> None:
    asset = _ToyAsset()
    with pytest.raises(RuntimeError, match="method is not set"):
        asset.present_value()


def test_pre_post_adjust_guarded_by_time_cache() -> None:
    # The same time should not re-invoke the impl; a different time
    # should.
    asset = _ToyAsset()
    asset.set_time(0.5)
    asset.pre_adjust_values()
    asset.pre_adjust_values()  # same time — guarded out
    asset.set_time(0.25)
    asset.pre_adjust_values()
    assert asset.pre_adjust_calls == 2

    asset.post_adjust_values()
    asset.post_adjust_values()
    assert asset.post_adjust_calls == 1  # only at t=0.25


def test_adjust_values_runs_both_impls() -> None:
    asset = _ToyAsset()
    asset.set_time(0.5)
    asset.adjust_values()
    assert asset.pre_adjust_calls == 1
    assert asset.post_adjust_calls == 1


def test_is_on_time_uses_grid_snapping() -> None:
    tg = TimeGrid.regular(end=1.0, steps=4)  # times 0, 0.25, 0.5, 0.75, 1.0
    lat = _RecordingLattice(time_grid=tg)
    asset = _ToyAsset()
    asset.initialize(lat, t=0.5)
    # The asset is currently at 0.5; ``is_on_time`` checks whether
    # ``grid[grid.index(t)]`` snaps to ``time`` (the held time).
    # At t = 0.5 we're on the grid -> True.
    assert asset.is_on_time(0.5)
    # At t = 0.25 we're on the grid but not at the asset's current
    # time -> False.
    assert not asset.is_on_time(0.25)


# -- DiscretizedDiscountBond ------------------------------------------


def test_discount_bond_reset_is_ones() -> None:
    bond = DiscretizedDiscountBond()
    bond.reset(size=5)
    assert bond.values.shape == (5,)
    assert np.all(bond.values == 1.0)


def test_discount_bond_mandatory_times_empty() -> None:
    bond = DiscretizedDiscountBond()
    assert bond.mandatory_times() == []


# -- DiscretizedOption ------------------------------------------------


def _make_initialized_pair(lat: _RecordingLattice, t: float) -> tuple[
    DiscretizedDiscountBond,
    DiscretizedOption,
]:
    """Helper: build a discount-bond underlying + option, initialize both."""
    bond = DiscretizedDiscountBond()
    bond.initialize(lat, t=t)
    opt = DiscretizedOption(
        underlying=bond,
        exercise_type=Exercise.Type.European,
        exercise_times=[t],
    )
    opt.initialize(lat, t=t)
    return bond, opt


def test_option_reset_requires_shared_method() -> None:
    tg = TimeGrid.regular(end=1.0, steps=2)
    lat_a = _RecordingLattice(time_grid=tg)
    lat_b = _RecordingLattice(time_grid=tg)
    bond = DiscretizedDiscountBond()
    bond.initialize(lat_a, t=1.0)
    opt = DiscretizedOption(
        underlying=bond,
        exercise_type=Exercise.Type.European,
        exercise_times=[1.0],
    )
    # The option's method is lat_b but the bond's is lat_a — reset
    # should refuse.
    with pytest.raises(LibraryException, match="different methods"):
        opt.initialize(lat_b, t=1.0)


def test_option_mandatory_times_unions_underlying_and_exercise() -> None:
    tg = TimeGrid.regular(end=2.0, steps=4)
    lat = _RecordingLattice(time_grid=tg)
    bond = DiscretizedDiscountBond()
    bond.initialize(lat, t=2.0)
    opt = DiscretizedOption(
        underlying=bond,
        exercise_type=Exercise.Type.Bermudan,
        # Mix positive + negative exercise times — negatives should drop.
        exercise_times=[0.5, 1.0, 1.5, -1.0],
    )
    # Underlying (bond) contributes nothing; only the positive exercise
    # times survive.
    mt = opt.mandatory_times()
    assert mt == [0.5, 1.0, 1.5]


def test_option_european_applies_exercise_at_on_time() -> None:
    tg = TimeGrid.regular(end=1.0, steps=2)  # times 0, 0.5, 1.0
    lat = _RecordingLattice(time_grid=tg)
    bond, opt = _make_initialized_pair(lat, t=1.0)
    del bond  # only needed for the initialize handshake side-effect
    # The initialize path runs reset (zero-fill) then adjust_values
    # which triggers the European exercise at t=1.0 (the only exercise
    # date). After exercise: max(bond=1, opt=0) = 1 in every node.
    assert np.all(opt.values >= 1.0 - 1e-12)
    # The recording lattice should have seen a partial_rollback driven
    # by the option's rollback of the underlying inside the exercise.
    kinds = [e[0] for e in lat.events]
    assert "partial_rollback" in kinds


def test_option_bermudan_skips_when_not_on_exercise_time() -> None:
    tg = TimeGrid.regular(end=1.0, steps=2)  # times 0, 0.5, 1.0
    lat = _RecordingLattice(time_grid=tg)
    bond = DiscretizedDiscountBond()
    bond.initialize(lat, t=1.0)
    # Exercise only at 0.5 — the asset is at 1.0 so the filter
    # ``is_on_time(0.5)`` is False (different grid index from the
    # asset's current time). No exercise happens.
    opt = DiscretizedOption(
        underlying=bond,
        exercise_type=Exercise.Type.Bermudan,
        exercise_times=[0.5],
    )
    opt.initialize(lat, t=1.0)
    # No exercise -> values from reset (zeros) remain unchanged.
    assert np.all(opt.values == 0.0)


def test_option_american_applies_in_window() -> None:
    tg = TimeGrid.regular(end=1.0, steps=2)
    lat = _RecordingLattice(time_grid=tg)
    bond, opt = _make_initialized_pair(lat, t=1.0)
    del opt
    # Re-create with American — exercise window [0.0, 1.0] is active
    # at t=1.0 (the time the asset is currently at).
    opt2 = DiscretizedOption(
        underlying=bond,
        exercise_type=Exercise.Type.American,
        exercise_times=[0.0, 1.0],
    )
    opt2.initialize(lat, t=1.0)
    # Initialize's reset+adjust_values has already applied exercise
    # because t=1.0 is in the window.
    # max(underlying=1, current=0) = 1 across the slice.
    assert np.all(opt2.values >= 1.0 - 1e-12)


def test_option_american_outside_window_skips_exercise() -> None:
    tg = TimeGrid.regular(end=1.0, steps=2)
    lat = _RecordingLattice(time_grid=tg)
    bond = DiscretizedDiscountBond()
    bond.initialize(lat, t=1.0)
    # Window [0.0, 0.4] — current time 1.0 is outside; no exercise.
    opt = DiscretizedOption(
        underlying=bond,
        exercise_type=Exercise.Type.American,
        exercise_times=[0.0, 0.4],
    )
    opt.initialize(lat, t=1.0)
    # No exercise -> values from reset stay zero.
    assert np.all(opt.values == 0.0)


def test_option_accessors_round_trip() -> None:
    tg = TimeGrid.regular(end=1.0, steps=2)
    lat = _RecordingLattice(time_grid=tg)
    bond = DiscretizedDiscountBond()
    bond.initialize(lat, t=1.0)
    opt = DiscretizedOption(
        underlying=bond,
        exercise_type=Exercise.Type.European,
        exercise_times=[0.5, 1.0],
    )
    assert opt.underlying is bond
    assert opt.exercise_type == Exercise.Type.European
    assert opt.exercise_times == [0.5, 1.0]
    # Defensive-copy property: mutating the returned list doesn't
    # affect internal state.
    et = opt.exercise_times
    et.append(2.0)
    assert opt.exercise_times == [0.5, 1.0]
