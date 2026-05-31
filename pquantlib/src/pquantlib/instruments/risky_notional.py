"""Notional-loss tracking for cat bonds.

# C++ parity: ql/experimental/catbonds/riskynotional.{hpp,cpp} (v1.42.1).

A ``NotionalPath`` records the surviving fraction of the original notional
after a sequence of loss-triggered reductions (each pinned to a payment
date).  A ``NotionalRisk`` maps a scenario of (event_date, loss) pairs onto
such a path:

- ``DigitalNotionalRisk`` — any single event with loss >= threshold wipes
  the notional to zero.
- ``ProportionalNotionalRisk`` — cumulative losses between an attachment
  and an exhaustion point linearly erode the notional.

An ``EventPaymentOffset`` maps an event date to the date the loss is
actually applied (``NoOffset`` is the identity).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.time.date import Date

if TYPE_CHECKING:
    from collections.abc import Sequence

_NULL_DATE: Date = Date()


class EventPaymentOffset(ABC):
    """Maps an event date to the date its loss is applied.

    # C++ parity: ``class EventPaymentOffset`` (riskynotional.hpp:36-40).
    """

    @abstractmethod
    def payment_date(self, event_date: Date) -> Date:
        ...


class NoOffset(EventPaymentOffset):
    """Identity offset — loss applied on the event date.

    # C++ parity: ``class NoOffset`` (riskynotional.hpp:42-45).
    """

    def payment_date(self, event_date: Date) -> Date:
        return event_date


class NotionalPath:
    """Surviving-notional schedule after loss reductions.

    # C++ parity: ``class NotionalPath`` (riskynotional.{hpp,cpp}).
    """

    def __init__(self) -> None:
        # Full notional at the beginning (pinned to the null date).
        self._notional_rate: list[tuple[Date, float]] = [(_NULL_DATE, 1.0)]

    def notional_rate(self, date: Date) -> float:
        """Surviving fraction of the original notional on ``date``.

        # C++ parity: riskynotional.cpp:30-36.  Returns the rate in force
        # on ``date`` (the last reduction at or before it).
        """
        i = 0
        while i < len(self._notional_rate) and self._notional_rate[i][0] <= date:
            i += 1
        return self._notional_rate[i - 1][1]

    def reset(self) -> None:
        # C++ parity: riskynotional.cpp:38-40.
        del self._notional_rate[1:]

    def add_reduction(self, date: Date, new_rate: float) -> None:
        # C++ parity: riskynotional.cpp:42-44.
        self._notional_rate.append((date, new_rate))

    def loss(self) -> float:
        # C++ parity: riskynotional.cpp:46-48 — 1 minus the final rate.
        return 1.0 - self._notional_rate[-1][1]


class NotionalRisk(ABC):
    """Maps a loss scenario onto a notional path.

    # C++ parity: ``class NotionalRisk`` (riskynotional.hpp:63-74).
    """

    def __init__(self, payment_offset: EventPaymentOffset) -> None:
        self._payment_offset: EventPaymentOffset = payment_offset

    @abstractmethod
    def update_path(self, events: Sequence[tuple[Date, float]], path: NotionalPath) -> None:
        """Reset ``path`` and populate it from ``events``.

        # C++ parity: ``NotionalRisk::updatePath`` (pure virtual).
        """


class DigitalNotionalRisk(NotionalRisk):
    """Single-event threshold wipe.

    # C++ parity: ``class DigitalNotionalRisk`` (riskynotional.{hpp,cpp}).
    """

    def __init__(self, payment_offset: EventPaymentOffset, threshold: float) -> None:
        super().__init__(payment_offset)
        self._threshold: float = threshold

    def update_path(self, events: Sequence[tuple[Date, float]], path: NotionalPath) -> None:
        # C++ parity: riskynotional.cpp:50-58.
        path.reset()
        for event_date, event_loss in events:
            if event_loss >= self._threshold:
                path.add_reduction(self._payment_offset.payment_date(event_date), 0.0)


class ProportionalNotionalRisk(NotionalRisk):
    """Attachment/exhaustion linear notional erosion.

    # C++ parity: ``class ProportionalNotionalRisk`` (riskynotional.hpp:90-119).
    """

    def __init__(
        self, payment_offset: EventPaymentOffset, attachement: float, exhaustion: float
    ) -> None:
        super().__init__(payment_offset)
        qassert.require(
            attachement < exhaustion,
            "exhaustion level needs to be greater than attachement",
        )
        self._attachement: float = attachement
        self._exhaustion: float = exhaustion

    def update_path(self, events: Sequence[tuple[Date, float]], path: NotionalPath) -> None:
        # C++ parity: riskynotional.hpp:101-114.
        path.reset()
        losses = 0.0
        previous_notional = 1.0
        for event_date, event_loss in events:
            losses += event_loss
            if losses > self._attachement and previous_notional > 0:
                previous_notional = max(
                    0.0, (self._exhaustion - losses) / (self._exhaustion - self._attachement)
                )
                path.add_reduction(
                    self._payment_offset.payment_date(event_date), previous_notional
                )


__all__ = [
    "DigitalNotionalRisk",
    "EventPaymentOffset",
    "NoOffset",
    "NotionalPath",
    "NotionalRisk",
    "ProportionalNotionalRisk",
]
