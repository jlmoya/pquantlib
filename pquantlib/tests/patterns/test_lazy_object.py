"""Tests for pquantlib.patterns.lazy_object."""

from __future__ import annotations

from pquantlib.patterns.lazy_object import LazyObject


class _Computer(LazyObject):
    def __init__(self) -> None:
        super().__init__()
        self.compute_count: int = 0
        self.result: int = -1

    def _perform_calculations(self) -> None:
        self.compute_count += 1
        self.result = 42


def test_calculate_runs_perform_calculations_once() -> None:
    c = _Computer()
    c.calculate()
    assert c.compute_count == 1
    assert c.result == 42

    # Second call: no recompute.
    c.calculate()
    assert c.compute_count == 1


def test_update_invalidates_cache_so_next_calculate_recomputes() -> None:
    c = _Computer()
    c.calculate()
    assert c.compute_count == 1

    c.update()
    c.calculate()
    assert c.compute_count == 2


def test_update_propagates_to_observers() -> None:
    class _Counter:
        def __init__(self) -> None:
            self.count: int = 0

        def update(self) -> None:
            self.count += 1

    c = _Computer()
    listener = _Counter()
    c.register_with(listener)

    c.update()
    assert listener.count == 1
