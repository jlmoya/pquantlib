"""Callability + CallabilitySchedule — call/put schedule for callable bonds.

# C++ parity: ql/instruments/callabilityschedule.hpp (v1.42.1).

Data-carrier port: holds price + type + date for each call/put event.
Embedded-option pricing is deferred (no callable-bond engine in L3-B).

# C++ parity divergence — Visitor:
# the C++ Callability inherits from Event and is Visitor-dispatchable. In
# the Python port we omit the visitor.accept machinery at the
# Callability level — no consumer in L3-B needs visitor dispatch, and the
# wider visitor pattern is a deferred L2-D/L3 carve-out.
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib.instruments.bond import BondPrice
from pquantlib.time.date import Date

_NULL_DATE: Date = Date()


class CallabilityType(IntEnum):
    """C++ parity: ``Callability::Type`` enum (callabilityschedule.hpp:42)."""

    Call = 0
    Put = 1


class Callability:
    """A single call/put right at a fixed price and date.

    # C++ parity: ``class Callability : public Event``.
    """

    def __init__(self, price: BondPrice, callability_type: CallabilityType, date: Date) -> None:
        self._price: BondPrice = price
        self._type: CallabilityType = callability_type
        self._date: Date = date

    def price(self) -> BondPrice:
        return self._price

    def type(self) -> CallabilityType:
        return self._type

    def date(self) -> Date:
        return self._date

    def has_occurred(
        self,
        ref_date: Date | None = None,
        include_ref_date: bool | None = None,
    ) -> bool:
        """Whether this callability event has occurred at ``ref_date``.

        # C++ parity: ``Event::hasOccurred`` (ql/event.cpp:28-39) — the
        # ``Callability : public Event`` inherits it. Same semantics as
        # :meth:`CashFlow.has_occurred`: None/null ``ref_date`` -> False;
        # date < ref_date -> True; date > ref_date -> False; on equality
        # returns ``not include_ref_date`` (default False).
        """
        if ref_date is None or ref_date == _NULL_DATE:
            return False
        if ref_date < self._date:
            return False
        if self._date < ref_date:
            return True
        include = include_ref_date if include_ref_date is not None else False
        return not include


# In the C++ source ``CallabilitySchedule`` is a ``typedef
# std::vector<ext::shared_ptr<Callability>>`` (callabilityschedule.hpp:73).
# Python's equivalent is a list — we export both a Python-idiomatic
# type alias and a concrete class (for ``isinstance`` checks).
CallabilitySchedule = list[Callability]


__all__ = ["Callability", "CallabilitySchedule", "CallabilityType"]
