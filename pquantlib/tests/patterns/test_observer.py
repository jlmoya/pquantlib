"""Tests for pquantlib.patterns.observer (Observable + Observer Protocol)."""

from __future__ import annotations

import gc

from pquantlib.patterns.observer import Observable, Observer


class _Counter:
    """Simple Observer that tracks how many times update() was called."""

    def __init__(self) -> None:
        self.count: int = 0

    def update(self) -> None:
        self.count += 1


def test_observer_protocol_is_satisfied_by_a_class_with_update_method() -> None:
    obs = _Counter()
    assert isinstance(obs, Observer)


def test_observable_notifies_all_registered_observers() -> None:
    subject = Observable()
    a, b, c = _Counter(), _Counter(), _Counter()
    subject.register_with(a)
    subject.register_with(b)
    subject.register_with(c)

    subject.notify_observers()

    assert a.count == 1
    assert b.count == 1
    assert c.count == 1


def test_observable_unregister_stops_notifications() -> None:
    subject = Observable()
    a = _Counter()
    subject.register_with(a)
    subject.unregister_with(a)

    subject.notify_observers()

    assert a.count == 0


def test_observable_unregister_idempotent() -> None:
    subject = Observable()
    a = _Counter()
    # Unregistering a never-registered observer is a no-op (no exception).
    subject.unregister_with(a)


def test_observable_weakref_drops_dead_observers() -> None:
    """Observers held only by Observable get GC'd, mirroring C++ weak signals."""
    subject = Observable()
    a = _Counter()
    subject.register_with(a)

    del a
    gc.collect()

    # No live observers; notify should be a no-op.
    subject.notify_observers()


def test_observable_double_register_is_idempotent() -> None:
    subject = Observable()
    a = _Counter()
    subject.register_with(a)
    subject.register_with(a)  # duplicate
    subject.notify_observers()
    # WeakSet ensures a is registered once, so count is 1 not 2.
    assert a.count == 1
