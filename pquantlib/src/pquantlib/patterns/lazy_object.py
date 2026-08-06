"""Lazy-evaluation mixin.

# C++ parity: ql/patterns/lazyobject.hpp @ v1.43.

Subclass and implement ``_perform_calculations``. Call ``calculate()``
to trigger evaluation; the result is cached until ``update()`` invalidates
it (and propagates the notification to the LazyObject's own observers).

The full C++ state machine is reproduced: ``calculated_`` / ``frozen_`` /
``failed_`` / ``alwaysForward_`` plus the ``updating_`` re-entrancy guard,
the ``Defaults`` per-session singleton, and ``UpdateChecker``. Defaults match
C++ compiled WITHOUT ``QL_FASTER_LAZY_OBJECTS`` (``forwardsAllNotifications_
= true``, ql/patterns/lazyobject.hpp:176-180), which is upstream's own default.
"""

from __future__ import annotations

from abc import abstractmethod
from types import TracebackType

from pquantlib.patterns.observer import Observable
from pquantlib.patterns.singleton import Singleton


class LazyObjectDefaults(Singleton):
    """Per-session default for how lazy objects forward notifications.

    # C++ parity: ``class LazyObject::Defaults : public Singleton<...>``
    # (ql/patterns/lazyobject.hpp:146-181). Flattened to module scope, as this
    # port does for nested types; a run-time change does NOT affect lazy
    # objects already created, same as C++.
    """

    def __init__(self) -> None:
        if getattr(self, "_lazy_defaults_initialized", False):
            return
        # C++ default without QL_FASTER_LAZY_OBJECTS (lazyobject.hpp:176-180).
        self._forwards_all_notifications: bool = True
        self._lazy_defaults_initialized: bool = True

    def forward_first_notification_only(self) -> None:
        """# C++ parity: ``Defaults::forwardFirstNotificationOnly`` (lazyobject.hpp:158)."""
        self._forwards_all_notifications = False

    def always_forward_notifications(self) -> None:
        """# C++ parity: ``Defaults::alwaysForwardNotifications`` (lazyobject.hpp:166)."""
        self._forwards_all_notifications = True

    def forwards_all_notifications(self) -> bool:
        """# C++ parity: ``Defaults::forwardsAllNotifications`` (lazyobject.hpp:170)."""
        return self._forwards_all_notifications


class UpdateChecker:
    """Scope guard setting ``_updating`` for the duration of ``update()``.

    # C++ parity: private nested ``LazyObject::UpdateChecker``
    # (ql/patterns/lazyobject.hpp:134-142) — an RAII guard whose destructor
    # clears the flag on both the normal and the exceptional exit. Python's
    # deterministic equivalent of a stack-scoped destructor is the context
    # manager, which is how ``LazyObject.update`` uses it.
    """

    __slots__ = ("_subject",)

    def __init__(self, subject: LazyObject) -> None:
        self._subject = subject

    def __enter__(self) -> UpdateChecker:
        # C++ UpdateChecker is a friend by nesting; ``_set_updating`` is the
        # Python stand-in for that privileged access.
        self._subject._set_updating(True)  # pyright: ignore[reportPrivateUsage]
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._subject._set_updating(False)  # pyright: ignore[reportPrivateUsage]


class LazyObject(Observable):
    """Cached-calculation base class with Observer-driven invalidation.

    The four state flags are CLASS-level defaults, mirroring the C++ in-class
    initialisers ``mutable bool calculated_ = false, frozen_ = false,
    failed_ = false`` and ``bool updating_ = false``
    (ql/patterns/lazyobject.hpp:130-133). Several models in this port inherit
    from two bases with their own ``__init__`` (e.g. Gsr -> Gaussian1dModel +
    CalibratedModel) and do not route through ``LazyObject.__init__``; class
    defaults keep those correct instead of raising AttributeError.
    """

    _calculated: bool = False
    _frozen: bool = False
    _failed: bool = False
    _updating: bool = False
    # C++ seeds alwaysForward_ from Defaults, whose own default is true when
    # the library is built without QL_FASTER_LAZY_OBJECTS (lazyobject.hpp:176-180).
    _always_forward: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._calculated = False
        self._frozen = False
        self._failed = False
        self._updating = False
        # C++ parity: ``LazyObject::LazyObject()`` (lazyobject.hpp:186-187)
        # seeds alwaysForward_ from the Defaults singleton AT CONSTRUCTION.
        self._always_forward = LazyObjectDefaults().forwards_all_notifications()

    @abstractmethod
    def _perform_calculations(self) -> None: ...

    def _set_updating(self, updating: bool) -> None:
        """Set the re-entrancy flag. Called only by :class:`UpdateChecker`.

        # C++ parity: ``LazyObject::updating_`` is private and mutated by the
        # nested friend ``UpdateChecker`` (ql/patterns/lazyobject.hpp:134-142).
        """
        self._updating = updating

    def is_calculated(self) -> bool:
        """True while the cached result is valid.

        # C++ parity: ``LazyObject::isCalculated`` (lazyobject.hpp:274-276).
        """
        return self._calculated

    def set_calculated(self, c: bool) -> None:
        """# C++ parity: ``LazyObject::setCalculated`` (lazyobject.hpp:278-280)."""
        self._calculated = c

    def calculate(self) -> None:
        """Run ``_perform_calculations`` exactly once until ``update`` invalidates.

        # C++ parity: ``LazyObject::calculate`` (lazyobject.hpp:257-272) — sets
        # ``calculated_=true`` BEFORE calling ``performCalculations`` to prevent
        # infinite recursion in bootstrap-style flows, rolls it back and records
        # ``failed_`` if the calculation raises, and is a no-op while frozen.
        """
        if not self._calculated and not self._frozen:
            self._calculated = True
            try:
                self._perform_calculations()
                self._failed = False
            except BaseException:
                self._calculated = False
                self._failed = True
                raise

    def update(self) -> None:
        """Invalidate the cache and notify downstream observers.

        # C++ parity: ``LazyObject::update`` (lazyobject.hpp:190-219) —
        # re-entrant calls return immediately (breaking notification cycles),
        # notification is forwarded only when the object had a result to
        # invalidate (or ``_always_forward`` is set, which is the default),
        # and frozen objects never notify.
        """
        if self._updating:
            return
        with UpdateChecker(self):
            if self._calculated or self._failed or self._always_forward:
                # set to false early: (1) to prevent infinite recursion,
                # (2) otherwise non-lazy observers would be served obsolete
                # data because of _calculated being still true.
                self._calculated = False
                self._failed = False
                if not self._frozen:
                    self.notify_observers()

    def recalculate(self) -> None:
        """Force recalculation, then notify.

        # C++ parity: ``LazyObject::recalculate`` (lazyobject.hpp:221-233) —
        # the frozen flag is restored and observers are notified on BOTH the
        # normal and the exceptional path.
        """
        was_frozen = self._frozen
        self._calculated = False
        self._frozen = False
        self._failed = False
        try:
            self.calculate()
        except BaseException:
            self._frozen = was_frozen
            self.notify_observers()
            raise
        self._frozen = was_frozen
        self.notify_observers()

    def freeze(self) -> None:
        """Pin the cached results across argument changes.

        # C++ parity: ``LazyObject::freeze`` (lazyobject.hpp:235-237).
        """
        self._frozen = True

    def unfreeze(self) -> None:
        """Re-enable recalculation, notifying once if we were frozen.

        # C++ parity: ``LazyObject::unfreeze`` (lazyobject.hpp:239-246).
        """
        if self._frozen:
            self._frozen = False
            self.notify_observers()

    def forward_first_notification_only(self) -> None:
        """# C++ parity: ``LazyObject::forwardFirstNotificationOnly`` (lazyobject.hpp:248-250)."""
        self._always_forward = False

    def always_forward_notifications(self) -> None:
        """# C++ parity: ``LazyObject::alwaysForwardNotifications`` (lazyobject.hpp:252-254)."""
        self._always_forward = True

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


__all__ = ["LazyObject", "LazyObjectDefaults", "UpdateChecker"]
