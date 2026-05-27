"""Shared mock YieldTermStructure satisfying YieldTermStructureProtocol.

L2-C IborIndex / rate-helper tests need a concrete curve, but L2-B hasn't
landed on this branch yet. We supply a simple flat-forward mock implemented
inline so the tests don't depend on L2-B.

The flat-forward formula matches C++ ``FlatForward`` with ``Continuous``
compounding:  ``discount(t) = exp(-r * t)``  for ``t`` derived from
``day_counter.year_fraction(reference_date, target_date)``.

Annual-compounded variant available for parity completeness:
   discount(t) = (1 + r)^(-t).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.time.date import Date


@dataclass(frozen=True, slots=True)
class FlatForwardMock:
    """Mock flat-forward curve. Continuous compounding, day-counted on ref + target."""

    reference: Date
    rate: float
    dc: DayCounter

    def reference_date(self) -> Date:
        return self.reference

    def max_date(self) -> Date:
        return Date.max_date()

    def day_counter(self) -> DayCounter:
        return self.dc

    def discount(self, t: float | Date, extrapolate: bool = False) -> float:
        del extrapolate
        t = t if not isinstance(t, Date) else self.dc.year_fraction(self.reference, t)
        return exp(-self.rate * t)

    def zero_rate(self, arg: float | Date, extrapolate: bool = False) -> float:
        del extrapolate, arg
        return self.rate

    def forward_rate(
        self,
        t1: float | Date,
        t2: float | Date,
        extrapolate: bool = False,
    ) -> float:
        del t1, t2, extrapolate
        return self.rate
