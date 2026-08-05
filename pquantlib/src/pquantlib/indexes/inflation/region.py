"""Region — geographical area, used for inflation indexes.

# C++ parity: ql/indexes/region.{hpp,cpp} (v1.43) — class ``Region`` holds a
  ``(name, code)`` pair behind ``name()`` / ``code()`` accessors, with one
  subclass per region and equality defined on ``name()`` alone.

``CustomRegion`` takes a user-supplied ``(name, code)``, so the region set is
open — which is why this is a class hierarchy rather than an enumeration.
``GenericRegion`` comes from ``ql/experimental/inflation/genericindexes.hpp``
and backs the generic test indexes that drive YoY optionlet stripping.
"""

from __future__ import annotations


class Region:
    """Geographical/economic region of an inflation index.

    # C++ parity: ``Region`` in ql/indexes/region.hpp. The C++ base has a
    # protected default constructor and subclasses fill in ``data_``; Python
    # takes the pair directly and each subclass supplies it.
    """

    __slots__ = ("_code", "_name")

    def __init__(self, name: str, code: str) -> None:
        self._name = name
        self._code = code

    def name(self) -> str:
        """Mirror C++ ``Region::name()``."""
        return self._name

    def code(self) -> str:
        """Mirror C++ ``Region::code()``."""
        return self._code

    def __eq__(self, other: object) -> bool:
        # C++ parity: operator== compares name() only, not code().
        if not isinstance(other, Region):
            return NotImplemented
        return self._name == other._name

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __hash__(self) -> int:
        return hash(self._name)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self._name!r}, code={self._code!r})"


class CustomRegion(Region):
    """User-defined region. # C++ parity: ``CustomRegion`` in region.hpp."""

    __slots__ = ()

    def __init__(self, name: str, code: str) -> None:
        super().__init__(name, code)


class AustraliaRegion(Region):
    """Australia. # C++ parity: ``AustraliaRegion`` in region.cpp."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("Australia", "AU")


class EURegion(Region):
    """European Union. # C++ parity: ``EURegion`` in region.cpp."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("EU", "EU")


class FranceRegion(Region):
    """France. # C++ parity: ``FranceRegion`` in region.cpp."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("France", "FR")


class UKRegion(Region):
    """United Kingdom. # C++ parity: ``UKRegion`` in region.cpp."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("UK", "UK")


class USRegion(Region):
    """United States. # C++ parity: ``USRegion`` in region.cpp."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("USA", "US")


class ZARegion(Region):
    """South Africa. # C++ parity: ``ZARegion`` in region.cpp."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("South Africa", "ZA")


class GenericRegion(Region):
    """Placeholder region for the generic test indexes.

    # C++ parity: ``GenericRegion`` in
    # ql/experimental/inflation/genericindexes.hpp.
    """

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("Generic", "GENERIC")
