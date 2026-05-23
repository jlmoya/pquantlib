"""Observer / Observable pattern.

# C++ parity: ql/patterns/observable.hpp (v1.42.1) — boost::signals2-backed.

Python uses ``weakref.WeakSet`` for observer storage so that an
Observable does not keep its observers alive (mirrors the C++
weak-binding that prevents cycles in long-lived term-structure graphs).
"""

from __future__ import annotations

import weakref
from typing import Protocol, runtime_checkable


@runtime_checkable
class Observer(Protocol):
    """Anything with an ``update()`` method is an Observer."""

    def update(self) -> None: ...


class Observable:
    """Subject that notifies registered observers on demand."""

    def __init__(self) -> None:
        self._observers: weakref.WeakSet[Observer] = weakref.WeakSet()

    def register_with(self, observer: Observer) -> None:
        self._observers.add(observer)

    def unregister_with(self, observer: Observer) -> None:
        self._observers.discard(observer)

    def notify_observers(self) -> None:
        # Snapshot the set because update() may mutate observer registration.
        for obs in list(self._observers):
            obs.update()
