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

    def is_calculated(self) -> bool:
        """True while the cached result is valid.

        # C++ parity: ``LazyObject::isCalculated`` (lazyobject.hpp:44, 272).
        """
        return self._calculated

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

    def update(self) -> None:
        """Invalidate the cache and notify downstream observers."""
        self._calculated = False
        self.notify_observers()

    def deep_update(self) -> None:
        """Force an update of this object *and* of anything it aggregates.

        # C++ parity: ``Observer::deepUpdate`` (ql/patterns/observable.hpp:155,
        # 267-269) — the default is plain ``update()``. C++ declares it on
        # ``Observer``; here ``Observer`` is a structural Protocol, and a
        # method with a body on a Protocol becomes a *required* member for
        # every structural implementer, so it lives on this concrete base
        # instead. Classes that aggregate other lazy objects
        # (``CompositeInstrument``) override it to cascade first.
        """
        self.update()
