"""YoYOptionletStripper — interface for YoY-cap stripping.

# C++ parity: ql/experimental/inflation/yoyoptionletstripper.hpp (v1.42.1)
   — abstract ``YoYOptionletStripper``.

Strippers return K-slices of the YoY optionlet volatility surface at a
given date; :meth:`initialize` performs the per-K stripping.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pquantlib.experimental.inflation.yoy_cap_floor_term_price_surface import (
    YoYCapFloorTermPriceSurface,
)
from pquantlib.pricingengines.inflation.yoy_inflation_capfloor_engine import (
    YoYInflationCapFloorEngine,
)
from pquantlib.time.date import Date


class YoYOptionletStripper(ABC):
    """Interface for inflation-cap stripping from price surfaces."""

    @abstractmethod
    def initialize(
        self,
        surface: YoYCapFloorTermPriceSurface,
        pricer: YoYInflationCapFloorEngine,
        slope: float,
    ) -> None:
        """Strip the YoY caplet vols along each K from a price surface."""

    @abstractmethod
    def min_strike(self) -> float: ...

    @abstractmethod
    def max_strike(self) -> float: ...

    @abstractmethod
    def strikes(self) -> list[float]: ...

    @abstractmethod
    def slice(self, d: Date) -> tuple[list[float], list[float]]:
        """Return ``(strikes, vols)`` of the surface at date ``d``."""
