"""Quantity — amount of a commodity in a given unit of measure.

# C++ parity: ql/experimental/commodities/quantity.hpp + quantity.cpp
#             (v1.42.1).

Arithmetic (+, -, *, /) and comparisons combine quantities; when the two
operands carry different units of measure the behavior is governed by the
class-level ``conversion_type`` / ``base_unit_of_measure`` settings, exactly
mirroring the C++ ``static`` members ``Quantity::conversionType`` /
``Quantity::baseUnitOfMeasure``. Cross-UOM conversions are routed through
``UnitOfMeasureConversionManager`` (imported lazily to avoid an import cycle).
"""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum

from pquantlib import qassert
from pquantlib.experimental.commodities.commodity_type import CommodityType
from pquantlib.experimental.commodities.unit_of_measure import UnitOfMeasure
from pquantlib.math.closeness import close as _close_real
from pquantlib.math.closeness import close_enough as _close_enough_real

_DEFAULT_N = 42


class ConversionType(IntEnum):
    """How to combine quantities in different units of measure.

    Parity with C++ ``Quantity::ConversionType``.
    """

    NO_CONVERSION = 0
    BASE_UNIT_OF_MEASURE_CONVERSION = 1
    AUTOMATED_CONVERSION = 2


def _convert_to(m: Quantity, target: UnitOfMeasure) -> Quantity:
    """Convert ``m`` into ``target`` via the conversion manager (then round)."""
    if m.unit_of_measure != target:
        # Lazy import: manager -> conversion -> quantity would cycle otherwise.
        from pquantlib.experimental.commodities.unit_of_measure_conversion_manager import (  # noqa: PLC0415
            UnitOfMeasureConversionManager,
        )

        rate = UnitOfMeasureConversionManager.instance().lookup(
            m.commodity_type, m.unit_of_measure, target
        )
        return rate.convert(m).rounded()
    return m


def _convert_to_base(m: Quantity) -> Quantity:
    qassert.require(
        not Quantity.base_unit_of_measure.empty(),
        "no base unitOfMeasure set",
    )
    return _convert_to(m, Quantity.base_unit_of_measure)


class Quantity:
    """Amount of a commodity (commodity type + unit of measure + amount)."""

    # C++ parity: ``static ConversionType conversionType`` /
    # ``static UnitOfMeasure baseUnitOfMeasure`` — process-global mutable
    # settings shared by all arithmetic.
    conversion_type: ConversionType = ConversionType.NO_CONVERSION
    base_unit_of_measure: UnitOfMeasure = UnitOfMeasure()

    # Re-export the enum as a nested name for the C++ idiom
    # ``Quantity.ConversionType.NO_CONVERSION``.
    ConversionType = ConversionType

    def __init__(
        self,
        commodity_type: CommodityType | None = None,
        unit_of_measure: UnitOfMeasure | None = None,
        amount: float = 0.0,
    ) -> None:
        self._commodity_type: CommodityType = (
            commodity_type if commodity_type is not None else CommodityType()
        )
        self._unit_of_measure: UnitOfMeasure = (
            unit_of_measure if unit_of_measure is not None else UnitOfMeasure()
        )
        self._amount: float = amount

    # ---- inspectors ----

    @property
    def commodity_type(self) -> CommodityType:
        return self._commodity_type

    @property
    def unit_of_measure(self) -> UnitOfMeasure:
        return self._unit_of_measure

    @property
    def amount(self) -> float:
        return self._amount

    def rounded(self) -> Quantity:
        """Apply the unit-of-measure rounding policy to the amount."""
        return Quantity(
            self._commodity_type,
            self._unit_of_measure,
            self._unit_of_measure.rounding(self._amount),
        )

    # ---- unary + in-place arithmetic ----

    def __pos__(self) -> Quantity:
        return Quantity(self._commodity_type, self._unit_of_measure, self._amount)

    def __neg__(self) -> Quantity:
        return Quantity(self._commodity_type, self._unit_of_measure, -self._amount)

    def __iadd__(self, other: Quantity) -> Quantity:
        if self._unit_of_measure == other._unit_of_measure:
            self._amount += other._amount
        elif Quantity.conversion_type == ConversionType.BASE_UNIT_OF_MEASURE_CONVERSION:
            base_self = _convert_to_base(self)
            base_other = _convert_to_base(other)
            self._adopt(base_self)
            self._amount += base_other._amount
        elif Quantity.conversion_type == ConversionType.AUTOMATED_CONVERSION:
            tmp = _convert_to(other, self._unit_of_measure)
            self._amount += tmp._amount
        else:
            qassert.fail("unitOfMeasure mismatch and no conversion specified")
        return self

    def __isub__(self, other: Quantity) -> Quantity:
        if self._unit_of_measure == other._unit_of_measure:
            self._amount -= other._amount
        elif Quantity.conversion_type == ConversionType.BASE_UNIT_OF_MEASURE_CONVERSION:
            base_self = _convert_to_base(self)
            base_other = _convert_to_base(other)
            self._adopt(base_self)
            self._amount -= base_other._amount
        elif Quantity.conversion_type == ConversionType.AUTOMATED_CONVERSION:
            tmp = _convert_to(other, self._unit_of_measure)
            self._amount -= tmp._amount
        else:
            qassert.fail("unitOfMeasure mismatch and no conversion specified")
        return self

    def __imul__(self, x: float) -> Quantity:
        self._amount *= x
        return self

    def __itruediv__(self, x: float) -> Quantity:
        self._amount /= x
        return self

    def _adopt(self, other: Quantity) -> None:
        """Replace this quantity's commodity type / UOM / amount in place."""
        self._commodity_type = other._commodity_type
        self._unit_of_measure = other._unit_of_measure
        self._amount = other._amount

    # ---- binary arithmetic (return new objects) ----

    def __add__(self, other: Quantity) -> Quantity:
        tmp = Quantity(self._commodity_type, self._unit_of_measure, self._amount)
        tmp += other
        return tmp

    def __sub__(self, other: Quantity) -> Quantity:
        tmp = Quantity(self._commodity_type, self._unit_of_measure, self._amount)
        tmp -= other
        return tmp

    def __mul__(self, x: float) -> Quantity:
        tmp = Quantity(self._commodity_type, self._unit_of_measure, self._amount)
        tmp *= x
        return tmp

    def __rmul__(self, x: float) -> Quantity:
        return self.__mul__(x)

    def __truediv__(self, other: Quantity | float) -> Quantity | float:
        """``q / scalar`` -> Quantity; ``q / q`` -> dimensionless Real ratio."""
        if isinstance(other, Quantity):
            return _ratio(self, other)
        tmp = Quantity(self._commodity_type, self._unit_of_measure, self._amount)
        tmp /= other
        return tmp

    # ---- comparisons (parity with the C++ relational free functions) ----

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quantity):
            return NotImplemented
        return _apply_bool(self, other, lambda x, y: x == y)

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __lt__(self, other: Quantity) -> bool:
        return _apply_bool(self, other, lambda x, y: x < y)

    def __le__(self, other: Quantity) -> bool:
        return _apply_bool(self, other, lambda x, y: x <= y)

    def __gt__(self, other: Quantity) -> bool:
        return _apply_bool(other, self, lambda x, y: x < y)

    def __ge__(self, other: Quantity) -> bool:
        return _apply_bool(other, self, lambda x, y: x <= y)

    def __hash__(self) -> int:
        return hash((self._unit_of_measure, self._amount))

    def __str__(self) -> str:
        return f"{self._commodity_type.code} {self._amount} {self._unit_of_measure.code}"

    def __repr__(self) -> str:
        return f"Quantity({self.__str__()!r})"


def _ratio(m1: Quantity, m2: Quantity) -> float:
    """Dimensionless ratio of two quantities (parity with ``operator/``)."""
    if m1.unit_of_measure == m2.unit_of_measure:
        return m1.amount / m2.amount
    if Quantity.conversion_type == ConversionType.BASE_UNIT_OF_MEASURE_CONVERSION:
        return _ratio(_convert_to_base(m1), _convert_to_base(m2))
    if Quantity.conversion_type == ConversionType.AUTOMATED_CONVERSION:
        return _ratio(m1, _convert_to(m2, m1.unit_of_measure))
    qassert.fail("unitOfMeasure mismatch and no conversion specified")


def _apply_bool(
    m1: Quantity,
    m2: Quantity,
    f: Callable[[float, float], bool],
) -> bool:
    """Shared dispatch for the comparison operators (parity with quantity.cpp)."""
    if m1.unit_of_measure == m2.unit_of_measure:
        return f(m1.amount, m2.amount)
    if Quantity.conversion_type == ConversionType.BASE_UNIT_OF_MEASURE_CONVERSION:
        return f(_convert_to_base(m1).amount, _convert_to_base(m2).amount)
    if Quantity.conversion_type == ConversionType.AUTOMATED_CONVERSION:
        return f(m1.amount, _convert_to(m2, m1.unit_of_measure).amount)
    qassert.fail("unitOfMeasure mismatch and no conversion specified")


def close(m1: Quantity, m2: Quantity, n: int = _DEFAULT_N) -> bool:
    """Floating-point closeness of two quantities (parity with ``close``)."""
    if m1.unit_of_measure == m2.unit_of_measure:
        return _close_real(m1.amount, m2.amount, n)
    if Quantity.conversion_type == ConversionType.BASE_UNIT_OF_MEASURE_CONVERSION:
        return close(_convert_to_base(m1), _convert_to_base(m2), n)
    if Quantity.conversion_type == ConversionType.AUTOMATED_CONVERSION:
        return close(m1, _convert_to(m2, m1.unit_of_measure), n)
    qassert.fail("unitOfMeasure mismatch and no conversion specified")


def close_enough(m1: Quantity, m2: Quantity, n: int = _DEFAULT_N) -> bool:
    """Looser floating-point closeness (parity with ``close_enough``)."""
    if m1.unit_of_measure == m2.unit_of_measure:
        return _close_enough_real(m1.amount, m2.amount, n)
    if Quantity.conversion_type == ConversionType.BASE_UNIT_OF_MEASURE_CONVERSION:
        return close_enough(_convert_to_base(m1), _convert_to_base(m2), n)
    if Quantity.conversion_type == ConversionType.AUTOMATED_CONVERSION:
        return close_enough(m1, _convert_to(m2, m1.unit_of_measure), n)
    qassert.fail("unitOfMeasure mismatch and no conversion specified")
