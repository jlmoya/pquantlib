"""UnitOfMeasureConversion — a single UOM conversion factor (+ chaining).

# C++ parity: ql/experimental/commodities/unitofmeasureconversion.hpp +
#             unitofmeasureconversion.cpp (v1.42.1).

A ``Direct`` conversion stores a factor with the convention that one unit of
``source`` is worth ``factor`` units of ``target``. A ``Derived`` conversion
chains two conversions (built by :meth:`UnitOfMeasureConversion.chain`).
``convert`` applies the factor to a :class:`Quantity` (forward or inverse).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from pquantlib import qassert
from pquantlib.experimental.commodities.commodity_type import CommodityType
from pquantlib.experimental.commodities.quantity import Quantity
from pquantlib.experimental.commodities.unit_of_measure import UnitOfMeasure


class Type(IntEnum):
    """Conversion provenance (parity with C++ ``UnitOfMeasureConversion::Type``)."""

    DIRECT = 0  # given directly by the user
    DERIVED = 1  # derived from a chain of other conversions


@dataclass
class _Data:
    """PIMPL payload (parity with C++ ``UnitOfMeasureConversion::Data``)."""

    commodity_type: CommodityType
    source: UnitOfMeasure
    target: UnitOfMeasure
    conversion_factor: float
    type: Type
    code: str
    # For Derived conversions: the (r1, r2) pair chained together.
    chain_first: UnitOfMeasureConversion | None = None
    chain_second: UnitOfMeasureConversion | None = None


class UnitOfMeasureConversion:
    """A conversion factor between two units of measure for a commodity type."""

    # Nested enum alias for the C++ idiom ``UnitOfMeasureConversion.Type.DIRECT``.
    Type = Type

    def __init__(
        self,
        commodity_type: CommodityType | None = None,
        source: UnitOfMeasure | None = None,
        target: UnitOfMeasure | None = None,
        conversion_factor: float | None = None,
    ) -> None:
        if commodity_type is None:
            # default ctor -> empty placeholder
            self._data: _Data | None = None
            return
        assert source is not None
        assert target is not None
        assert conversion_factor is not None
        # code = commodityType.name + source.code + target.code  (parity)
        code = commodity_type.name + source.code + target.code
        self._data = _Data(
            commodity_type,
            source,
            target,
            conversion_factor,
            Type.DIRECT,
            code,
        )

    # ---- inspectors ----

    @property
    def commodity_type(self) -> CommodityType:
        assert self._data is not None
        return self._data.commodity_type

    @property
    def source(self) -> UnitOfMeasure:
        assert self._data is not None
        return self._data.source

    @property
    def target(self) -> UnitOfMeasure:
        assert self._data is not None
        return self._data.target

    @property
    def conversion_factor(self) -> float:
        assert self._data is not None
        return self._data.conversion_factor

    @property
    def type(self) -> Type:
        assert self._data is not None
        return self._data.type

    @property
    def code(self) -> str:
        assert self._data is not None
        return self._data.code

    def empty(self) -> bool:
        return self._data is None

    # ---- utilities ----

    def convert(self, quantity: Quantity) -> Quantity:
        """Apply the conversion factor to ``quantity`` (parity with ``convert``)."""
        assert self._data is not None
        data = self._data
        if data.type == Type.DIRECT:
            if quantity.unit_of_measure == data.source:
                return Quantity(
                    quantity.commodity_type,
                    data.target,
                    quantity.amount * data.conversion_factor,
                )
            if quantity.unit_of_measure == data.target:
                return Quantity(
                    quantity.commodity_type,
                    data.source,
                    quantity.amount / data.conversion_factor,
                )
            qassert.fail("direct conversion not applicable")
        elif data.type == Type.DERIVED:
            first = data.chain_first
            second = data.chain_second
            assert first is not None
            assert second is not None
            if quantity.unit_of_measure in (first.source, first.target):
                return second.convert(first.convert(quantity))
            if quantity.unit_of_measure in (second.source, second.target):
                return first.convert(second.convert(quantity))
            qassert.fail("derived conversion factor not applicable")
        qassert.fail("unknown conversion-factor type")

    @staticmethod
    def chain(
        r1: UnitOfMeasureConversion,
        r2: UnitOfMeasureConversion,
    ) -> UnitOfMeasureConversion:
        """Chain two conversions into a Derived conversion (parity with ``chain``)."""
        assert r1._data is not None
        assert r2._data is not None
        result = UnitOfMeasureConversion()
        result._data = _Data(
            commodity_type=r1._data.commodity_type,
            source=UnitOfMeasure(),
            target=UnitOfMeasure(),
            conversion_factor=0.0,
            type=Type.DERIVED,
            code="",
            chain_first=r1,
            chain_second=r2,
        )
        d = result._data
        d1 = r1._data
        d2 = r2._data
        if d1.source == d2.source:
            d.source = d1.target
            d.target = d2.target
            d.conversion_factor = d2.conversion_factor / d1.conversion_factor
        elif d1.source == d2.target:
            d.source = d1.target
            d.target = d2.source
            d.conversion_factor = 1.0 / (d1.conversion_factor * d2.conversion_factor)
        elif d1.target == d2.source:
            d.source = d1.source
            d.target = d2.target
            d.conversion_factor = d1.conversion_factor * d2.conversion_factor
        elif d1.target == d2.target:
            d.source = d1.source
            d.target = d2.source
            d.conversion_factor = d1.conversion_factor / d2.conversion_factor
        else:
            qassert.fail("conversion factors not chainable")
        return result
