"""PaymentTerm — payment-term flyweight (e.g. "Pricing end + 5 days").

# C++ parity: ql/experimental/commodities/paymentterm.hpp +
#             paymentterm.cpp (v1.42.1).

Flyweight keyed on ``name`` (static ``paymentTerms_`` map in C++; a
module-level ``_payment_terms`` dict here). A payment term resolves a
payment date from a reference date via
``calendar.adjust(date + offset_days)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date


class EventType(IntEnum):
    """Anchor event for the payment offset (parity with ``PaymentTerm::EventType``)."""

    TRADE_DATE = 0
    PRICING_DATE = 1


@dataclass
class _Data:
    """Flyweight payload (parity with C++ ``PaymentTerm::Data``)."""

    name: str
    event_type: EventType
    offset_days: int
    calendar: Calendar


# Module-level flyweight registry (parity with C++ static paymentTerms_ map,
# keyed on name).
_payment_terms: dict[str, _Data] = {}


class PaymentTerm:
    """Payment-term flyweight (name + anchor event + offset days + calendar)."""

    # Nested enum alias for the C++ idiom ``PaymentTerm.EventType.TRADE_DATE``.
    EventType = EventType

    def __init__(
        self,
        name: str | None = None,
        event_type: EventType | None = None,
        offset_days: int | None = None,
        calendar: Calendar | None = None,
    ) -> None:
        if name is None:
            self._data: _Data | None = None
            return
        assert event_type is not None
        assert offset_days is not None
        assert calendar is not None
        existing = _payment_terms.get(name)
        if existing is not None:
            self._data = existing
        else:
            data = _Data(name, event_type, offset_days, calendar)
            _payment_terms[name] = data
            self._data = data

    # ---- inspectors ----

    @property
    def name(self) -> str:
        """Name, e.g. ``"Pricing end + 5 days"``."""
        assert self._data is not None
        return self._data.name

    @property
    def event_type(self) -> EventType:
        assert self._data is not None
        return self._data.event_type

    @property
    def offset_days(self) -> int:
        assert self._data is not None
        return self._data.offset_days

    @property
    def calendar(self) -> Calendar:
        assert self._data is not None
        return self._data.calendar

    def empty(self) -> bool:
        return self._data is None

    def get_payment_date(self, date: Date) -> Date:
        """Resolve a payment date from ``date`` (parity with ``getPaymentDate``)."""
        assert self._data is not None
        return self._data.calendar.adjust(date + self._data.offset_days)

    # ---- comparison (parity with C++ name-based relational operators) ----

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PaymentTerm):
            return NotImplemented
        if self.empty() and other.empty():
            return True
        if self.empty() or other.empty():
            return False
        return self.name == other.name

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __hash__(self) -> int:
        return hash(self.name) if self._data is not None else hash(None)

    def __str__(self) -> str:
        return self.name if self._data is not None else "null payment term type"

    def __repr__(self) -> str:
        return f"PaymentTerm({self.__str__()!r})"
