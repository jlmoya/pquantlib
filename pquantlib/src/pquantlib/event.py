"""Event — the dated, observable base of everything that happens on a date.

# C++ parity: ql/event.{hpp,cpp} @ v1.43.

``Event`` is the base of ``CashFlow`` (ql/cashflow.hpp) and of
``Callability`` (ql/instruments/callabilityschedule.hpp): a thing with a
``date()`` that can be asked whether it has already happened.

# C++ parity divergence — ``accept``:
# C++ ``Event::accept(AcyclicVisitor&)`` does a ``dynamic_cast`` to
# ``Visitor<Event>*`` and QL_FAILs when the visitor does not handle events.
# This port's visitor is a structural ``Protocol`` (pquantlib/patterns/visitor.py),
# so the "does this visitor handle me" question is answered by whether the
# object satisfies the protocol — checked here with ``isinstance`` against the
# runtime-checkable ``Visitor``, which is the direct analogue of the C++
# ``dynamic_cast``-and-fail.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pquantlib import qassert
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.patterns.observer import Observable
from pquantlib.patterns.visitor import Visitor
from pquantlib.time.date import Date

_NULL_DATE: Date = Date()


class Event(Observable, ABC):
    """Base class for anything that occurs on a date.

    # C++ parity: ``class Event : public virtual Observable`` (ql/event.hpp:40).
    """

    @abstractmethod
    def date(self) -> Date:
        """The date at which the event occurs.

        # C++ parity: ``virtual Date date() const = 0`` (ql/event.hpp:47).
        """

    def has_occurred(
        self,
        ref_date: Date | None = None,
        include_ref_date: bool | None = None,
    ) -> bool:
        """Whether the event has already occurred as of ``ref_date``.

        # C++ parity: ``Event::hasOccurred`` (ql/event.cpp:27-38), reproduced
        # branch for branch::
        #
        #     Date refDate = d != Date() ? d : Settings::instance().evaluationDate();
        #     bool inc = includeRefDate ? *includeRefDate
        #                               : Settings::instance().includeReferenceDateEvents();
        #     return inc ? date() < refDate : date() <= refDate;
        #
        # ``ref_date=None`` and the null ``Date()`` are both the C++ "not
        # supplied" sentinel and fall back to the global evaluation date;
        # ``include_ref_date=None`` is the C++ ``ext::nullopt`` and falls back
        # to ``ObservableSettings().include_reference_date_events``.
        """
        settings = ObservableSettings()
        if ref_date is None or ref_date == _NULL_DATE:
            resolved_ref = settings.evaluation_date_or_today()
        else:
            resolved_ref = ref_date
        include = (
            include_ref_date
            if include_ref_date is not None
            else settings.include_reference_date_events
        )
        if include:
            return self.date() < resolved_ref
        return self.date() <= resolved_ref

    def accept(self, visitor: object) -> None:
        """Dispatch to a visitor.

        # C++ parity: ``Event::accept(AcyclicVisitor&)`` (ql/event.cpp:40-46) —
        # C++ takes the degenerate ``AcyclicVisitor&`` base and dynamic_casts to
        # ``Visitor<Event>*``, QL_FAILing when that fails. The Python parameter is
        # therefore typed ``object`` (the analogue of the untyped base) and the
        # isinstance check against the runtime_checkable ``Visitor`` Protocol IS
        # the dynamic_cast.
        """
        if not isinstance(visitor, Visitor):
            qassert.fail("not an event visitor")
        visitor.visit(self)


__all__ = ["Event"]
