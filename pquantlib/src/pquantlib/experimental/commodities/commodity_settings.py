"""CommoditySettings — global commodity-domain settings (singleton).

# C++ parity: ql/experimental/commodities/commoditysettings.hpp +
#             commoditysettings.cpp (v1.42.1).

Holds the process-global "cash" currency and unit of measure used by
commodity pricing. Defaults match C++: ``USDCurrency()`` and
``BarrelUnitOfMeasure()``.

The C++ class is a ``Singleton`` exposing mutable ``currency()`` /
``unitOfMeasure()`` reference accessors. PQuantLib exposes read/write
properties on the singleton; an ``instance()`` classmethod mirrors the
C++ ``::instance()`` idiom.
"""

from __future__ import annotations

from pquantlib.currencies.america import USDCurrency
from pquantlib.currencies.currency import Currency
from pquantlib.experimental.commodities.petroleum_units_of_measure import (
    BarrelUnitOfMeasure,
)
from pquantlib.experimental.commodities.unit_of_measure import UnitOfMeasure
from pquantlib.patterns.singleton import Singleton


class CommoditySettings(Singleton):
    """Global repository for commodity run-time settings (singleton)."""

    def __init__(self) -> None:
        super().__init__()
        self._currency: Currency = USDCurrency()
        self._unit_of_measure: UnitOfMeasure = BarrelUnitOfMeasure()

    @classmethod
    def instance(cls) -> CommoditySettings:
        """Return the singleton instance (parity with C++ ``::instance()``)."""
        return cls()

    @property
    def currency(self) -> Currency:
        return self._currency

    @currency.setter
    def currency(self, value: Currency) -> None:
        self._currency = value

    @property
    def unit_of_measure(self) -> UnitOfMeasure:
        return self._unit_of_measure

    @unit_of_measure.setter
    def unit_of_measure(self, value: UnitOfMeasure) -> None:
        self._unit_of_measure = value
