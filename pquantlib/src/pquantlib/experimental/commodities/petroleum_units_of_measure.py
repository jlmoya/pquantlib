"""Petroleum units of measure (Barrel, Metric Tonne, Gallon, Litre, ...).

# C++ parity: ql/experimental/commodities/petroleumunitsofmeasure.hpp
#             (v1.42.1).

Each concrete is a thin ``UnitOfMeasure`` subclass that installs a
registered flyweight ``_Data`` carrying the unit's name/code/type and a
triangulation unit (most petroleum volumes triangulate through Barrel,
matching the C++ subclass constructors).
"""

from __future__ import annotations

from pquantlib.experimental.commodities.unit_of_measure import (
    UnitOfMeasure,
    UnitType,
)


class BarrelUnitOfMeasure(UnitOfMeasure):
    """Barrels (BBL), a volume unit. The triangulation hub for petroleum."""

    def __init__(self) -> None:
        self._assign_flyweight("Barrels", "BBL", UnitType.VOLUME)


class MTUnitOfMeasure(UnitOfMeasure):
    """Metric Tonnes (MT), a mass unit."""

    def __init__(self) -> None:
        self._assign_flyweight("Metric Tonnes", "MT", UnitType.MASS)


class MBUnitOfMeasure(UnitOfMeasure):
    """1000 Barrels (MB), a volume unit triangulating through Barrel."""

    def __init__(self) -> None:
        self._assign_flyweight(
            "1000 Barrels", "MB", UnitType.VOLUME, BarrelUnitOfMeasure()
        )


class GallonUnitOfMeasure(UnitOfMeasure):
    """US Gallons (GAL), a volume unit triangulating through Barrel."""

    def __init__(self) -> None:
        self._assign_flyweight(
            "US Gallons", "GAL", UnitType.VOLUME, BarrelUnitOfMeasure()
        )


class LitreUnitOfMeasure(UnitOfMeasure):
    """Litres (l), a volume unit triangulating through Barrel."""

    def __init__(self) -> None:
        self._assign_flyweight("Litres", "l", UnitType.VOLUME, BarrelUnitOfMeasure())


class KilolitreUnitOfMeasure(UnitOfMeasure):
    """Kilolitres (kl), a volume unit triangulating through Barrel."""

    def __init__(self) -> None:
        self._assign_flyweight(
            "Kilolitres", "kl", UnitType.VOLUME, BarrelUnitOfMeasure()
        )


class TokyoKilolitreUnitOfMeasure(UnitOfMeasure):
    """Tokyo Kilolitres (KL_tk), a volume unit triangulating through Barrel."""

    def __init__(self) -> None:
        self._assign_flyweight(
            "Tokyo Kilolitres", "KL_tk", UnitType.VOLUME, BarrelUnitOfMeasure()
        )
