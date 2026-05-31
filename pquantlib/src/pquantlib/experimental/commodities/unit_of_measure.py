"""UnitOfMeasure — commodity unit-of-measure specification.

# C++ parity: ql/experimental/commodities/unitofmeasure.hpp +
#             unitofmeasure.cpp (v1.42.1).

The C++ class uses a ``shared_ptr<Data>`` PIMPL plus a static
``unitsOfMeasure_`` map keyed on ``name`` to give flyweight semantics:
two ``UnitOfMeasure`` objects built with the same ``name`` share the same
``Data`` instance. PQuantLib reproduces this with a module-level
``_units_of_measure`` registry dict and a small ``_Data`` dataclass; a
``UnitOfMeasure`` holds a reference to its ``_Data`` (or ``None`` for the
default-constructed "empty" instance).

Equality mirrors C++ ``operator==``: code-based (``c1.code() == c2.code()``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from pquantlib import qassert
from pquantlib.math.rounding import Rounding


class UnitType(IntEnum):
    """Kind of unit of measure (parity with C++ ``UnitOfMeasure::Type``)."""

    MASS = 0
    VOLUME = 1
    ENERGY = 2
    QUANTITY = 3


@dataclass
class _Data:
    """Flyweight payload shared by all UnitOfMeasure instances with the same name.

    Mirrors C++ ``UnitOfMeasure::Data``.
    """

    name: str
    code: str
    unit_type: UnitType
    triangulation_unit_of_measure: UnitOfMeasure
    rounding: Rounding = field(default_factory=Rounding)


# Module-level flyweight registry (parity with C++ static unitsOfMeasure_ map,
# keyed on name).
_units_of_measure: dict[str, _Data] = {}


class UnitOfMeasure:
    """%Unit of measure specification (Barrel, Metric Tonne, Gallon, ...).

    Default construction yields an "empty" placeholder (parity with C++
    ``UnitOfMeasure() = default``); such an instance must be reassigned to
    a valid unit before use.
    """

    # C++ parity: ``UnitOfMeasure::Type`` is a nested enum; expose it as a
    # class attribute so ``UnitOfMeasure.Type.MASS`` reads like the C++ idiom.
    Type = UnitType

    def __init__(
        self,
        name: str | None = None,
        code: str | None = None,
        unit_type: UnitType | None = None,
    ) -> None:
        if name is None:
            # default ctor -> empty placeholder
            self._data: _Data | None = None
            return
        qassert.require(code is not None, "UnitOfMeasure: code required")
        qassert.require(unit_type is not None, "UnitOfMeasure: unit_type required")
        assert code is not None
        assert unit_type is not None
        existing = _units_of_measure.get(name)
        if existing is not None:
            self._data = existing
        else:
            # Default triangulation is the empty UnitOfMeasure (parity with
            # C++ Data's default-constructed triangulationUnitOfMeasure).
            data = _Data(name, code, unit_type, UnitOfMeasure())
            _units_of_measure[name] = data
            self._data = data

    # ---- inspectors ----

    @property
    def name(self) -> str:
        """Name, e.g. ``"Barrels"``."""
        assert self._data is not None
        return self._data.name

    @property
    def code(self) -> str:
        """Code, e.g. ``"BBL"``, ``"MT"``."""
        assert self._data is not None
        return self._data.code

    @property
    def unit_type(self) -> UnitType:
        """Unit type (mass, volume, ...)."""
        assert self._data is not None
        return self._data.unit_type

    @property
    def rounding(self) -> Rounding:
        """Per-unit rounding policy."""
        assert self._data is not None
        return self._data.rounding

    @property
    def triangulation_unit_of_measure(self) -> UnitOfMeasure:
        """Unit used for triangulation when a direct conversion is unavailable."""
        assert self._data is not None
        return self._data.triangulation_unit_of_measure

    def empty(self) -> bool:
        """True if this is the default-constructed placeholder."""
        return self._data is None

    # ---- comparison (parity with C++ relational operators) ----

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UnitOfMeasure):
            return NotImplemented
        if self.empty() and other.empty():
            return True
        if self.empty() or other.empty():
            return False
        return self.code == other.code

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __hash__(self) -> int:
        return hash(self.code) if self._data is not None else hash(None)

    def __str__(self) -> str:
        return self.code if self._data is not None else "null unit of measure"

    def _assign_flyweight(
        self,
        name: str,
        code: str,
        unit_type: UnitType,
        triangulation: UnitOfMeasure | None = None,
    ) -> None:
        """Install (and register) this instance's flyweight ``_Data``.

        Used by the petroleum-UOM concretes. Mirrors the C++ subclass ctors
        that install a static ``Data`` carrying a triangulation unit (e.g.
        Litre triangulates through Barrel).
        """
        existing = _units_of_measure.get(name)
        if existing is not None:
            self._data = existing
            return
        tri = triangulation if triangulation is not None else UnitOfMeasure()
        data = _Data(name, code, unit_type, tri)
        _units_of_measure[name] = data
        self._data = data

    def __repr__(self) -> str:
        return f"UnitOfMeasure({self.__str__()!r})"


class LotUnitOfMeasure(UnitOfMeasure):
    """The "Lot" quantity unit (parity with C++ ``LotUnitOfMeasure``)."""

    def __init__(self) -> None:
        self._assign_flyweight("Lot", "Lot", UnitType.QUANTITY)
