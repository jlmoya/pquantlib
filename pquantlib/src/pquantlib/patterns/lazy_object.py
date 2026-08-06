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
        """Run ``_perform_calculations`` exactly once until ``update`` invalidates.

        # C++ parity: ``LazyObject::calculate`` sets ``calculated_=true``
        # BEFORE calling ``performCalculations`` to prevent infinite
        # recursion in bootstrap-style flows (e.g. ``Forward::forwardValue``
        # calls ``calculate()`` from inside ``performCalculations``).
        # The flag is rolled back if the calculation raises.
        """
        if not self._calculated:
            self._calculated = True
            try:
                self._perform_calculations()
            except BaseException:
                self._calculated = False
                raise

    def recalculate(self) -> None:
        """Force a recalculation and notify observers.

        # C++ parity: ``LazyObject::recalculate`` (lazyobject.hpp:139-152).
        # C++ additionally saves and restores ``frozen_`` around the call and
        # re-raises after notifying; this port has no ``freeze()`` /
        # ``unfreeze()`` (the ``frozen_`` flag is not modelled), so only the
        # invalidate-calculate-notify part has an analogue. The C++ order is
        # preserved: observers are notified after the calculation, and also
        # when it throws.
        """
        self._calculated = False
        try:
            self.calculate()
        except BaseException:
            self.notify_observers()
            raise
        self.notify_observers()

    def update(self) -> None:
        """Invalidate the cache and notify downstream observers."""
        self._calculated = False
        self.notify_observers()
