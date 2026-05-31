"""CommodityType — commodity-type flyweight (Heating Oil, WTI, ...).

# C++ parity: ql/experimental/commodities/commoditytype.hpp +
#             commoditytype.cpp (v1.42.1).

Flyweight: a static ``commodityTypes_`` map (keyed on ``code``) shares one
``Data`` instance across all ``CommodityType`` objects with the same code.
PQuantLib reproduces this with a module-level ``_commodity_types`` dict.

# C++ parity quirk: in v1.42.1 the *header* declares the constructor
# ``CommodityType(code, name)`` but the *.cpp* defines it as
# ``CommodityType(name, code)`` and keys the registry on ``code`` (the 2nd
# positional arg). Because C++ binds arguments by position, the first
# positional becomes ``name`` and the second becomes ``code`` at runtime
# (verified by the W7-B probe: ``CommodityType("HO", "Heating Oil")`` yields
# ``name()=="HO"``, ``code()=="Heating Oil"``). We replicate the *runtime*
# behavior: ``CommodityType(name, code)`` with the registry keyed on ``code``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _Data:
    """Flyweight payload (parity with C++ ``CommodityType::Data``)."""

    name: str
    code: str


# Module-level flyweight registry (parity with C++ static commodityTypes_ map,
# keyed on code).
_commodity_types: dict[str, _Data] = {}


class CommodityType:
    """Commodity type (e.g. name ``"Heating Oil"``, code ``"HO"``).

    Default construction yields an "empty" placeholder (parity with C++
    ``CommodityType() = default``).

    See the module docstring for the C++ ``(name, code)`` argument-order
    quirk that this class faithfully reproduces.
    """

    def __init__(self, name: str | None = None, code: str | None = None) -> None:
        if name is None and code is None:
            self._data: _Data | None = None
            return
        # Both must be present for a non-empty type.
        assert name is not None
        assert code is not None
        existing = _commodity_types.get(code)
        if existing is not None:
            self._data = existing
        else:
            data = _Data(name, code)
            _commodity_types[code] = data
            self._data = data

    @property
    def code(self) -> str:
        """Commodity code, e.g. ``"HO"`` (see ctor argument-order note)."""
        assert self._data is not None
        return self._data.code

    @property
    def name(self) -> str:
        """Name, e.g. ``"Heating Oil"`` (see ctor argument-order note)."""
        assert self._data is not None
        return self._data.name

    def empty(self) -> bool:
        """True if this is the default-constructed placeholder."""
        return self._data is None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CommodityType):
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
        return self.code if self._data is not None else "null commodity type"

    def __repr__(self) -> str:
        return f"CommodityType({self.__str__()!r})"


class NullCommodityType(CommodityType):
    """The ``<NULL>`` commodity type (parity with C++ ``NullCommodityType``)."""

    def __init__(self) -> None:
        super().__init__("<NULL>", "<NULL>")
