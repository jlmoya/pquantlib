"""Global library settings (singleton + observable).

# C++ parity: ql/settings.{hpp,cpp} @ v1.43 — class Settings + SavedSettings.

The C++ ``Settings`` singleton carries a handful of boolean flags that
affect library-wide behavior (e.g. enforcement of business-day conventions
during schedule generation, payment-date inclusion semantics) AND a
mutable ``evaluationDate`` whose changes propagate through the Observer
network to every term structure, index fixing, etc. registered with it.

PQuantLib mirrors both responsibilities on this one class:

- The ``Singleton`` base provides the one-instance-per-class semantics.
- The ``Observable`` base provides the observer plumbing.

The two are combined via simple multi-inheritance. The ``_initialized``
flag in ``__init__`` protects against re-initializing the Observable
state on every ``ObservableSettings()`` call (the Singleton metaclass
caches the instance, but the metaclass still invokes ``__init__`` on
the cached instance — guarding here is idiomatic and cheap).

The ``evaluation_date`` field follows C++ semantics: ``None`` means
"use today" (read by :meth:`evaluation_date_or_today`); setting it to
a concrete ``Date`` pins the evaluation date until set back to None
or replaced. Mutating it triggers ``notify_observers()`` so registered
term structures invalidate their cached reference dates.
"""

from __future__ import annotations

from pquantlib.patterns.observer import Observable
from pquantlib.patterns.singleton import Singleton
from pquantlib.time.date import Date


class ObservableSettings(Singleton, Observable):
    """Library-wide mutable flags + global evaluation date.

    # C++ parity: the mutable state of ``class Settings`` (ql/settings.hpp:112-116)
    # is exactly four fields::
    #
    #     DateProxy evaluationDate_;
    #     bool includeReferenceDateEvents_ = false;
    #     ext::optional<bool> includeTodaysCashFlows_;
    #     bool enforcesTodaysHistoricFixings_ = false;
    #
    # ``include_todays_cash_flows`` is ``bool | None`` because the C++ field is
    # an ``ext::optional<bool>``: unset means "no override", which is NOT the
    # same as ``False``. (It was previously spelled ``include_today_in_payments:
    # bool = False`` — wrong name and wrong type, and never read by anything.)
    #
    # ``enforces_business_day_convention`` has no C++ counterpart in v1.43; it
    # is a jquantlib-era carry-over kept only because an existing test asserts
    # its presence. It is deliberately NOT part of the SavedSettings snapshot,
    # which mirrors the C++ four fields.
    """

    enforces_business_day_convention: bool = True
    include_reference_date_events: bool = False
    include_todays_cash_flows: bool | None = None
    enforces_todays_historic_fixings: bool = False

    def __init__(self) -> None:
        if getattr(self, "_observable_settings_initialized", False):
            # Re-instantiation through the Singleton metaclass: the cached
            # instance is returned, but ``__init__`` is still invoked. Skip
            # to preserve Observable state.
            return
        Observable.__init__(self)
        self._evaluation_date: Date | None = None
        self._observable_settings_initialized: bool = True

    # --- evaluation_date property -----------------------------------------

    @property
    def evaluation_date(self) -> Date | None:
        """Current pinned evaluation date, or ``None`` for "today".

        # C++ parity: ``Settings::instance().evaluationDate()`` returns
        # an ``ObservableValue<Date>``. The C++ getter returns today
        # implicitly if never set; the Python port preserves the
        # distinction (None vs Today) and exposes a separate
        # :meth:`evaluation_date_or_today` for the "resolve to a real
        # date" code path.
        """
        return self._evaluation_date

    @evaluation_date.setter
    def evaluation_date(self, d: Date | None) -> None:
        """Pin (or unpin, via ``None``) the global evaluation date.

        Notifies all registered observers — TermStructures in moving
        mode, RelativeDateBootstrapHelpers, floating SmileSections, etc.
        — so that their derived dates re-snap on next access.
        """
        if d == self._evaluation_date:
            # No-op (mirrors C++ ObservableValue<Date> behavior: assign
            # and notify only on actual change).
            return
        self._evaluation_date = d
        self.notify_observers()

    def evaluation_date_or_today(self) -> Date:
        """Resolve the evaluation date: pinned date if set, else today.

        Typical pattern for code that needs an effective "as-of" date:

            today = ObservableSettings().evaluation_date_or_today()

        # C++ parity: ``Settings::instance().evaluationDate()`` semantics
        # when the underlying ``ObservableValue<Date>`` has its lazy-init
        # default applied.
        """
        if self._evaluation_date is None:
            return Date.todays_date()
        return self._evaluation_date

    def anchor_evaluation_date(self) -> None:
        """Pin the evaluation date to today if it is not already pinned.

        # C++ parity: ``Settings::anchorEvaluationDate`` (ql/settings.cpp:38-43) —
        # a no-op when a date is already set.
        """
        if self._evaluation_date is None:
            self.evaluation_date = Date.todays_date()

    def reset_evaluation_date(self) -> None:
        """Un-pin the evaluation date, letting it track today again.

        # C++ parity: ``Settings::resetEvaluationDate`` (ql/settings.cpp:45-47).
        """
        self.evaluation_date = None


class SavedSettings:
    """Snapshot of :class:`ObservableSettings`, restored on exit.

    # C++ parity: ``class SavedSettings`` (ql/settings.hpp:120, ql/settings.cpp:49-67).
    #
    # C++ uses RAII: the constructor snapshots the four mutable ``Settings``
    # fields and the destructor puts them back. Python's deterministic analogue
    # of a stack-scoped destructor is the context manager, so this is both a
    # context manager AND callable via explicit :meth:`restore` for callers that
    # want the C++ object-lifetime shape.
    #
    # Note the C++ constructor reads ``Settings::instance().evaluationDate()``
    # through ``DateProxy::operator Date()``, which RESOLVES an unset date to
    # ``Date::todaysDate()``. Restoring therefore leaves the evaluation date
    # anchored even if it was unset on entry — reproduced here deliberately via
    # :meth:`ObservableSettings.evaluation_date_or_today`.

    Usage::

        with SavedSettings():
            ObservableSettings().evaluation_date = Date.from_ymd(1, Month.March, 2025)
            ...
        # every field is back to what it was
    """

    __slots__ = (
        "_enforces_todays_historic_fixings",
        "_evaluation_date",
        "_include_reference_date_events",
        "_include_todays_cash_flows",
    )

    def __init__(self) -> None:
        settings = ObservableSettings()
        self._evaluation_date: Date = settings.evaluation_date_or_today()
        self._include_reference_date_events: bool = settings.include_reference_date_events
        self._include_todays_cash_flows: bool | None = settings.include_todays_cash_flows
        self._enforces_todays_historic_fixings: bool = settings.enforces_todays_historic_fixings

    def restore(self) -> None:
        """Put every snapshotted field back.

        # C++ parity: ``SavedSettings::~SavedSettings`` (ql/settings.cpp:55-67).
        # The evaluation date is only re-assigned when it actually differs, so
        # no spurious observer notification is emitted — same guard as C++.
        """
        settings = ObservableSettings()
        if settings.evaluation_date != self._evaluation_date:
            settings.evaluation_date = self._evaluation_date
        settings.include_reference_date_events = self._include_reference_date_events
        settings.include_todays_cash_flows = self._include_todays_cash_flows
        settings.enforces_todays_historic_fixings = self._enforces_todays_historic_fixings

    def __enter__(self) -> SavedSettings:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.restore()


__all__ = ["ObservableSettings", "SavedSettings"]
