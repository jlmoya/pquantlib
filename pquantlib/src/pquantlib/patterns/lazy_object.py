"""Lazy-evaluation mixin.

# C++ parity: ql/patterns/lazyobject.hpp (v1.42.1).

Subclass and implement ``_perform_calculations``. Call ``calculate()``
to trigger evaluation; the result is cached until ``update()`` invalidates
it (and propagates the notification to the LazyObject's own observers).
"""

from __future__ import annotations

from abc import abstractmethod

from pquantlib.patterns.observer import Observable


class LazyObject(Observable):
    """Cached-calculation base class with Observer-driven invalidation."""

    def __init__(self) -> None:
        super().__init__()
        self._calculated: bool = False

    @abstractmethod
    def _perform_calculations(self) -> None: ...

    def calculate(self) -> None:
        """Run ``_perform_calculations`` exactly once until ``update`` invalidates."""
        if not self._calculated:
            self._perform_calculations()
            self._calculated = True

    def update(self) -> None:
        """Invalidate the cache and notify downstream observers."""
        self._calculated = False
        self.notify_observers()
