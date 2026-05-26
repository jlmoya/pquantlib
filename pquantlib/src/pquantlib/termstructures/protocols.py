"""Cross-cluster typing Protocols for Phase 2 L2-B/C/D/E parallelism.

# C++ parity: no direct analogue — C++ uses template parameters + concrete
   abstract base classes for the same role. Python's structural typing lets
   parallel cluster work reference each other's not-yet-merged concretes
   via these Protocols without import cycles.

How this is used:
- L2-D's IborCoupon takes ``IborIndexProtocol`` in its constructor.
- L2-C's concrete IborIndex (subclass of pquantlib.indexes.index.Index)
  structurally satisfies IborIndexProtocol.
- When both clusters merge into ``main``, runtime ``isinstance``
  conformance is automatic — no glue code required.

Pyright + ruff treat ``@runtime_checkable`` Protocols as duck-typing
contracts; they're verified at type-check time and can also be used
with isinstance() at runtime (cheap, since each method is just an
attribute lookup).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.period import Period


@runtime_checkable
class YieldTermStructureProtocol(Protocol):
    """Minimum API for a yield curve.

    L2-D's IborCoupon, OvernightIndexedCoupon, and CashFlows aggregator
    take this Protocol; L2-B's FlatForward / Interpolated*Curve /
    PiecewiseYieldCurve satisfy it.
    """

    def reference_date(self) -> Date: ...
    def max_date(self) -> Date: ...
    def day_counter(self) -> DayCounter: ...
    def discount(self, t: float | Date, extrapolate: bool = False) -> float: ...
    def zero_rate(self, t: float | Date, extrapolate: bool = False) -> float: ...
    def forward_rate(
        self,
        t1: float | Date,
        t2: float | Date,
        extrapolate: bool = False,
    ) -> float: ...


@runtime_checkable
class IborIndexProtocol(Protocol):
    """Minimum API for an IBOR-style index (Euribor, USDLibor, etc.).

    L2-D's IborCoupon takes this; L2-C's IborIndex concretes satisfy it.
    """

    def name(self) -> str: ...
    def tenor(self) -> Period: ...
    def fixing_days(self) -> int: ...
    def currency(self) -> object: ...  # avoid currency-module import cycle
    def fixing_calendar(self) -> Calendar: ...
    def day_counter(self) -> DayCounter: ...
    def business_day_convention(self) -> BusinessDayConvention: ...
    def end_of_month(self) -> bool: ...
    def fixing(self, fixing_date: Date, forecast_todays_fixing: bool = False) -> float: ...


@runtime_checkable
class OvernightIndexProtocol(Protocol):
    """Minimum API for an overnight rate index (Eonia, Sofr, Sonia, etc.).

    L2-D's OvernightIndexedCoupon takes this; L2-C's OvernightIndex
    concretes satisfy it.
    """

    def name(self) -> str: ...
    def currency(self) -> object: ...
    def fixing_calendar(self) -> Calendar: ...
    def day_counter(self) -> DayCounter: ...
    def fixing(self, fixing_date: Date, forecast_todays_fixing: bool = False) -> float: ...


@runtime_checkable
class SwapIndexProtocol(Protocol):
    """Minimum API for a swap rate index (EuriborSwapIsdaFixA, etc.)."""

    def name(self) -> str: ...
    def tenor(self) -> Period: ...
    def fixing_days(self) -> int: ...
    def currency(self) -> object: ...
    def fixed_leg_tenor(self) -> Period: ...
    def fixed_leg_convention(self) -> BusinessDayConvention: ...
    def fixed_leg_day_counter(self) -> DayCounter: ...
    def ibor_index(self) -> IborIndexProtocol: ...
