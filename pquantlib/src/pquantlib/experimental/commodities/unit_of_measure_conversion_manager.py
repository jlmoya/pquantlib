"""UnitOfMeasureConversionManager — singleton conversion-factor registry.

# C++ parity: ql/experimental/commodities/unitofmeasureconversionmanager.hpp +
#             unitofmeasureconversionmanager.cpp (v1.42.1).

A process-global repository of :class:`UnitOfMeasureConversion` factors. On
construction it seeds the known petroleum conversion factors (Barrel /
Gallon / MB / Litre / Kilolitre). ``lookup`` first tries a direct match,
then triangulation through a unit's ``triangulation_unit_of_measure``, and
finally a depth-first ``smart`` chain search.

The C++ class is a ``Singleton`` exposing ``::instance()``. PQuantLib's
:class:`~pquantlib.patterns.singleton.Singleton` returns the cached instance
when called (``UnitOfMeasureConversionManager()``); we additionally provide
an ``instance()`` classmethod so client code can mirror the C++ idiom.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.experimental.commodities.commodity_type import (
    CommodityType,
    NullCommodityType,
)
from pquantlib.experimental.commodities.petroleum_units_of_measure import (
    BarrelUnitOfMeasure,
    GallonUnitOfMeasure,
    KilolitreUnitOfMeasure,
    LitreUnitOfMeasure,
    MBUnitOfMeasure,
    TokyoKilolitreUnitOfMeasure,
)
from pquantlib.experimental.commodities.unit_of_measure import UnitOfMeasure
from pquantlib.experimental.commodities.unit_of_measure_conversion import (
    UnitOfMeasureConversion,
)
from pquantlib.patterns.singleton import Singleton


def _matches_cc(c1: UnitOfMeasureConversion, c2: UnitOfMeasureConversion) -> bool:
    return c1.commodity_type == c2.commodity_type and (
        (c1.source == c2.source and c1.target == c2.target)
        or (c1.source == c2.target and c1.target == c2.source)
    )


def _matches_st(
    c: UnitOfMeasureConversion,
    commodity_type: CommodityType,
    source: UnitOfMeasure,
    target: UnitOfMeasure,
) -> bool:
    return c.commodity_type == commodity_type and (
        (c.source == source and c.target == target)
        or (c.source == target and c.target == source)
    )


def _matches_s(
    c: UnitOfMeasureConversion,
    commodity_type: CommodityType,
    source: UnitOfMeasure,
) -> bool:
    return c.commodity_type == commodity_type and source in (c.source, c.target)


class UnitOfMeasureConversionManager(Singleton):
    """Singleton repository of conversion factors between units of measure."""

    def __init__(self) -> None:
        super().__init__()
        self._data: list[UnitOfMeasureConversion] = []
        self._add_known_conversion_factors()

    @classmethod
    def instance(cls) -> UnitOfMeasureConversionManager:
        """Return the singleton instance (parity with C++ ``::instance()``)."""
        return cls()

    # ---- mutation ----

    def add(self, c: UnitOfMeasureConversion) -> None:
        """Register ``c``, replacing any existing matching conversion."""
        for i, existing in enumerate(self._data):
            if _matches_cc(existing, c):
                del self._data[i]
                break
        self._data.append(c)

    def clear(self) -> None:
        """Drop all conversions and re-seed the known petroleum factors."""
        self._data.clear()
        self._add_known_conversion_factors()

    # ---- lookup ----

    def lookup(
        self,
        commodity_type: CommodityType,
        source: UnitOfMeasure,
        target: UnitOfMeasure,
        type: UnitOfMeasureConversion.Type = UnitOfMeasureConversion.Type.DERIVED,
    ) -> UnitOfMeasureConversion:
        """Find a conversion from ``source`` to ``target`` (parity with ``lookup``)."""
        if type == UnitOfMeasureConversion.Type.DIRECT:
            return self._direct_lookup(commodity_type, source, target)
        if not source.triangulation_unit_of_measure.empty():
            link = source.triangulation_unit_of_measure
            if link == target:
                return self._direct_lookup(commodity_type, source, link)
            return UnitOfMeasureConversion.chain(
                self._direct_lookup(commodity_type, source, link),
                self.lookup(commodity_type, link, target),
            )
        if not target.triangulation_unit_of_measure.empty():
            link = target.triangulation_unit_of_measure
            if source == link:
                return self._direct_lookup(commodity_type, link, target)
            return UnitOfMeasureConversion.chain(
                self.lookup(commodity_type, source, link),
                self._direct_lookup(commodity_type, link, target),
            )
        return self._smart_lookup(commodity_type, source, target)

    # ---- internals ----

    def _add_known_conversion_factors(self) -> None:
        nct = NullCommodityType()
        self.add(
            UnitOfMeasureConversion(nct, MBUnitOfMeasure(), BarrelUnitOfMeasure(), 1000)
        )
        self.add(
            UnitOfMeasureConversion(
                nct, BarrelUnitOfMeasure(), GallonUnitOfMeasure(), 42
            )
        )
        self.add(
            UnitOfMeasureConversion(
                nct, GallonUnitOfMeasure(), MBUnitOfMeasure(), 1000 * 42
            )
        )
        self.add(
            UnitOfMeasureConversion(
                nct, LitreUnitOfMeasure(), GallonUnitOfMeasure(), 3.78541
            )
        )
        self.add(
            UnitOfMeasureConversion(
                nct, BarrelUnitOfMeasure(), LitreUnitOfMeasure(), 158.987
            )
        )
        self.add(
            UnitOfMeasureConversion(
                nct, KilolitreUnitOfMeasure(), BarrelUnitOfMeasure(), 6.28981
            )
        )
        self.add(
            UnitOfMeasureConversion(
                nct, TokyoKilolitreUnitOfMeasure(), BarrelUnitOfMeasure(), 6.28981
            )
        )

    def _direct_lookup(
        self,
        commodity_type: CommodityType,
        source: UnitOfMeasure,
        target: UnitOfMeasure,
    ) -> UnitOfMeasureConversion:
        for c in self._data:
            if _matches_st(c, commodity_type, source, target):
                return c
        qassert.fail(
            f"no direct conversion available from {commodity_type.code} "
            f"{source.code} to {target.code}"
        )

    def _smart_lookup(
        self,
        commodity_type: CommodityType,
        source: UnitOfMeasure,
        target: UnitOfMeasure,
        forbidden: list[str] | None = None,
    ) -> UnitOfMeasureConversion:
        forbidden = list(forbidden) if forbidden is not None else []
        try:
            return self._direct_lookup(commodity_type, source, target)
        except LibraryException:
            pass  # no direct conversion; fall through to chain search

        # Forbid the source code to avoid cycles in the depth-first search.
        forbidden.append(source.code)

        for c in self._data:
            if _matches_s(c, commodity_type, source):
                other = c.target if source == c.source else c.source
                if other.code not in forbidden:
                    try:
                        tail = self._smart_lookup(
                            commodity_type, other, target, forbidden
                        )
                        return UnitOfMeasureConversion.chain(c, tail)
                    except LibraryException:
                        pass  # discard this branch, try the next

        qassert.fail(
            f"no conversion available for {commodity_type.code} from "
            f"{source.code} to {target.code}"
        )
