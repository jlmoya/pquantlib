"""CorrelationTermStructure — abstract correlation curve / surface base.

# C++ parity: ql/experimental/credit/correlationstructure.{hpp,cpp} (v1.42.1).

A correlation term structure is a TermStructure that returns scalar / matrix
correlation values indexed by time and (optionally) loss level. Compared
with a volatility surface, the correlation range is intrinsic ([-1, +1]) and
there is no strike reference; the abstract base is otherwise the same.

The Python port also adds a ``CompoundCorrelationStructure`` aggregator
that wraps a list of child correlation structures. The C++ surface does
not expose this aggregator directly but the constituent correlation
structures (used in the base-correlation framework) often need to be
composed; the Python port exposes it directly for downstream consumption.
"""

from __future__ import annotations

from abc import abstractmethod

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.termstructures.term_structure import TermStructure
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.period import Period


class CorrelationTermStructure(TermStructure):
    """Abstract correlation term structure.

    The C++ version exposes three constructors (default / fixed reference /
    settlement-days); the Python port keeps that surface but routes
    everything through the unified ``TermStructure`` constructor.

    # C++ parity: correlationstructure.hpp:40-72.
    """

    __slots__ = ("_bdc",)

    def __init__(
        self,
        *,
        bdc: BusinessDayConvention,
        calendar: Calendar | None = None,
        day_counter: DayCounter | None = None,
        reference_date: Date | None = None,
        settlement_days: int | None = None,
    ) -> None:
        super().__init__(
            reference_date=reference_date,
            calendar=calendar,
            day_counter=day_counter,
            settlement_days=settlement_days,
        )
        self._bdc = bdc

    def business_day_convention(self) -> BusinessDayConvention:
        return self._bdc

    def date_from_tenor(self, p: Period) -> Date:
        """Advance the reference date by ``p`` under the stored bdc."""
        return self.calendar().advance_period(self.reference_date(), p, self._bdc)

    @abstractmethod
    def correlation_size(self) -> int:
        """Size of the squared correlation matrix (1 for scalar curves)."""


class CompoundCorrelationStructure(CorrelationTermStructure):
    """Aggregator that wraps a list of child correlation structures.

    # C++ parity divergence: the C++ surface does not expose a direct
    # ``CompoundCorrelationStructure`` aggregator — composition is done
    # ad hoc inside specific base-correlation engines. The Python port
    # exposes the aggregator as a separate class so downstream loss-model
    # / engine code can compose curves uniformly.

    All children must share the same reference date + day counter + calendar
    (validated at construction). The aggregator's ``correlation_size`` is
    the maximum of the children's sizes.
    """

    __slots__ = ("_structures",)

    def __init__(self, structures: list[CorrelationTermStructure]) -> None:
        qassert.require(
            len(structures) > 0,
            "CompoundCorrelationStructure requires at least one child",
        )
        first = structures[0]
        super().__init__(
            bdc=first.business_day_convention(),
            calendar=first.calendar(),
            day_counter=first.day_counter(),
            reference_date=first.reference_date(),
        )
        # Validate calendar / day-counter consistency
        for s in structures[1:]:
            qassert.require(
                s.calendar() == first.calendar(),
                "child correlation structures have inconsistent calendars",
            )
            qassert.require(
                s.day_counter() == first.day_counter(),
                "child correlation structures have inconsistent day counters",
            )
            qassert.require(
                s.reference_date() == first.reference_date(),
                "child correlation structures have inconsistent reference dates",
            )
        self._structures = list(structures)
        for s in self._structures:
            s.register_with(self)

    def update(self) -> None:
        """Propagate child invalidations."""
        super().update()

    def structures(self) -> list[CorrelationTermStructure]:
        """Return a copy of the child structures."""
        return list(self._structures)

    def correlation_size(self) -> int:
        # Size = max of children's matrix sizes.
        return max(s.correlation_size() for s in self._structures)

    def max_date(self) -> Date:
        # Compound is alive only over the intersection of children's lifespans.
        # # C++ parity divergence: there is no C++ aggregator analogue —
        # the Python implementation uses the safest behavior (min of children
        # max-dates) so that downstream callers cannot accidentally query
        # past a child's max.
        return min(s.max_date() for s in self._structures)
