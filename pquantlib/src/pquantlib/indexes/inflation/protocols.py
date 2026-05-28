"""Cross-cluster typing Protocols for the Phase 7 inflation cluster.

# C++ parity: no direct analogue — C++ uses template parameters + concrete
   abstract base classes for the same role. Python's structural typing lets
   L7-B (curves), L7-C (cashflows), and L7-D (instruments + engines) cross-
   reference each other's not-yet-merged concretes via these Protocols
   without import cycles.

How this is used:

- L7-C's inflation cashflows take ``InflationIndexProtocol`` in their
  constructor; L7-A's ``InflationIndex`` concretes (EUHICP / FRHICP / ...)
  structurally satisfy it.
- L7-D's inflation cap/floor engines take ``InflationTermStructureProtocol``;
  L7-B's piecewise/interpolated curves satisfy it.
- When the clusters merge into ``main``, runtime ``isinstance`` conformance
  is automatic — no glue code required.

Pyright + ruff treat ``@runtime_checkable`` Protocols as duck-typing
contracts; they're verified at type-check time and can also be used with
``isinstance()`` at runtime (cheap, since each method is just an attribute
lookup).

Pattern source: pquantlib.termstructures.protocols (Phase 2 L2-B/C/D/E).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pquantlib.indexes.inflation.region import Region
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period


@runtime_checkable
class InflationIndexProtocol(Protocol):
    """Minimum API for an inflation-rate index (EUHICP, FRHICP, UKRPI, ...).

    L7-C's inflation cashflows take this; L7-A's InflationIndex concretes
    satisfy it. The currency accessor is typed as ``object`` to break the
    currency-module import cycle (mirrors the IborIndexProtocol pattern).
    """

    def name(self) -> str: ...
    def family_name(self) -> str: ...
    def region(self) -> Region: ...
    def frequency(self) -> Frequency: ...
    def availability_lag(self) -> Period: ...
    def interpolated(self) -> bool: ...
    def revised(self) -> bool: ...
    def currency(self) -> object: ...
    def fixing(self, fixing_date: Date, forecast_todays_fixing: bool = False) -> float: ...


@runtime_checkable
class InflationTermStructureProtocol(Protocol):
    """Minimum API for an inflation term structure (zero or YoY).

    L7-D's inflation cap/floor engines take this; L7-B's piecewise /
    interpolated curves satisfy it. The ``nominal_term_structure`` slot
    is Protocol-typed to keep the curve seam decoupled from concrete
    yield-curve choices.
    """

    def reference_date(self) -> Date: ...
    def max_date(self) -> Date: ...
    def base_date(self) -> Date: ...
    def frequency(self) -> Frequency: ...
    def observation_lag(self) -> Period | None: ...
    def nominal_term_structure(self) -> YieldTermStructureProtocol | None: ...
    def base_rate(self) -> float: ...
