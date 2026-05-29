"""Issuer — credit-name with probability curves and event history.

# C++ parity: ql/experimental/credit/issuer.{hpp,cpp} (v1.42.1).

The C++ class is a vector of (DefaultProbKey -> DefaultProbabilityTermStructure)
pairs (vector preferred over map for performance) plus a set of past
DefaultEvents ordered by date. The Python port keeps the same data
shape but exposes it as a list of pairs + a sorted list for
``DefaultEventSet`` (we don't need C++'s ``set<shared_ptr<>>`` with
``earlier_than`` comparator — Python sorts on key).

# C++ parity divergence: the C++ ``DefaultEventSet`` is a sorted set
# keyed on shared-pointer identity ordered by date. Python collapses
# this to a list-of-events kept sorted by date via the constructor's
# implicit sort. Equality remains via DefaultEvent.__eq__.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.currencies.currency import Currency
from pquantlib.experimental.credit.default_event import DefaultEvent
from pquantlib.experimental.credit.default_probability_key import DefaultProbKey
from pquantlib.experimental.credit.default_type import DefaultType, Seniority
from pquantlib.termstructures.credit.default_probability_term_structure import (
    DefaultProbabilityTermStructure,
)
from pquantlib.time.date import Date

# Type alias for the C++ DefaultEventSet (a date-sorted container of
# DefaultEvent instances). Python uses a plain list kept sorted on insert.
DefaultEventSet = list[DefaultEvent]
# Type alias for the (key, curve) pair stored on Issuer. Maps to C++
# Issuer::key_curve_pair = std::pair<DefaultProbKey, Handle<...>>.
KeyCurvePair = tuple[DefaultProbKey, DefaultProbabilityTermStructure]


def _between(
    event: DefaultEvent,
    start: Date,
    end: Date,
    include_ref_date: bool = False,
) -> bool:
    """C++ parity: anonymous ``between`` helper at issuer.cpp:27-33."""
    return not event.has_occurred(start, include_ref_date) and event.has_occurred(
        end, include_ref_date
    )


class Issuer:
    """A credit-issuer aggregate of default-probability curves + event history.

    Fields:
      - ``probabilities``: list of (DefaultProbKey, DefaultProbabilityTermStructure)
        pairs. Lookups are linear (vector-preferred-over-map as in C++).
      - ``events``: list of past ``DefaultEvent`` instances, sorted by date.

    A second constructor variant (``from_components``) corresponds to the
    C++ overload that takes separate vectors of (event_types, currencies,
    seniorities, curves).
    """

    __slots__ = ("_events", "_probabilities")

    def __init__(
        self,
        probabilities: list[KeyCurvePair] | None = None,
        events: DefaultEventSet | None = None,
    ) -> None:
        self._probabilities: list[KeyCurvePair] = (
            list(probabilities) if probabilities is not None else []
        )
        # Keep events sorted by date (mirrors C++ DefaultEventSet ordering).
        self._events: DefaultEventSet = (
            sorted(events) if events is not None else []
        )

    @classmethod
    def from_components(
        cls,
        event_types: list[list[DefaultType]],
        currencies: list[Currency],
        seniorities: list[Seniority],
        curves: list[DefaultProbabilityTermStructure],
        events: DefaultEventSet | None = None,
    ) -> Issuer:
        """Construct from parallel component lists.

        # C++ parity: issuer.cpp:41-57 — Issuer(vector<vector<DefaultType>>,
        # vector<Currency>, vector<Seniority>, vector<curves>, events).
        Requires all four component lists to have the same length.
        """
        qassert.require(
            len(event_types) == len(curves) == len(currencies) == len(seniorities),
            "Incompatible size of Issuer parameters.",
        )
        probabilities: list[KeyCurvePair] = []
        for i, evs in enumerate(event_types):
            key = DefaultProbKey(
                event_types=tuple(evs),
                currency=currencies[i],
                seniority=seniorities[i],
            )
            probabilities.append((key, curves[i]))
        return cls(probabilities=probabilities, events=events)

    # ----- inspectors ----------------------------------------------------------

    def probabilities(self) -> list[KeyCurvePair]:
        """Return the registered (key, curve) pairs."""
        return list(self._probabilities)

    def default_probability(
        self, key: DefaultProbKey
    ) -> DefaultProbabilityTermStructure:
        """Return the default-probability curve for ``key``.

        # C++ parity: issuer.cpp:59-65 — linear search, QL_FAIL on miss.
        """
        for k, curve in self._probabilities:
            if k == key:
                return curve
        qassert.fail("Probability curve not available.")

    def events(self) -> DefaultEventSet:
        """Return the date-sorted event history."""
        return list(self._events)

    # ----- utilities -----------------------------------------------------------

    def defaulted_between(
        self,
        start: Date,
        end: Date,
        contract_key: DefaultProbKey,
        include_ref_date: bool = False,
    ) -> DefaultEvent | None:
        """Return the first event in (start, end] matching ``contract_key``, or None.

        # C++ parity: issuer.cpp:67-80 — first-match wins (set is ordered
        # by date so this is the earliest). Python iterates the
        # date-sorted list.
        """
        for event in self._events:
            if event.matches_default_key(contract_key) and _between(
                event, start, end, include_ref_date
            ):
                return event
        return None

    def defaults_between(
        self,
        start: Date,
        end: Date,
        contract_key: DefaultProbKey,
        include_ref_date: bool = False,
    ) -> list[DefaultEvent]:
        """Return every event in (start, end] matching ``contract_key``.

        # C++ parity: issuer.cpp:83-97.
        """
        return [
            event
            for event in self._events
            if event.matches_default_key(contract_key)
            and _between(event, start, end, include_ref_date)
        ]
