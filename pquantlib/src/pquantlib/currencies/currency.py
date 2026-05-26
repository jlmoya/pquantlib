"""Currency descriptor — ISO 4217 unit of account.

# C++ parity: ql/currency.hpp + ql/currency.cpp (v1.42.1).

The C++ class uses a ``shared_ptr<Data>`` PIMPL to allow cheap copying
of a "default-constructed empty" currency. PQuantLib collapses that into
a plain frozen dataclass, treating "empty currency" as "the all-empty
default Currency()" (name == "" and code == "").

Equality semantics mirror C++ ``operator==``: two non-empty currencies
are equal iff their ``name`` matches; two empty currencies are equal;
empty vs non-empty are unequal. We override ``__eq__`` and ``__hash__``
explicitly because the default dataclass-generated comparators include
every field (including ``rounding``), which would diverge from C++.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pquantlib.math.rounding import Rounding


@dataclass(frozen=True, slots=True)
class Currency:
    """ISO 4217 currency descriptor.

    Attributes mirror C++ ``Currency::Data``:
      - ``name``: full English name (e.g. ``"U.S. dollar"``).
      - ``code``: 3-letter ISO code (e.g. ``"USD"``).
      - ``numeric_code``: ISO numeric code (e.g. ``840``).
      - ``symbol``: short symbol (e.g. ``"$"``).
      - ``fraction_symbol``: cents/pence symbol if any (e.g. ``"c"``).
      - ``fractions_per_unit``: e.g. 100 for cents (JPY uses 100 too,
        per ISO 4217 — historical sen).
      - ``rounding``: per-currency rounding policy.

    Default ctor produces an "empty" currency (parity with C++
    ``Currency() = default``).
    """

    name: str = ""
    code: str = ""
    numeric_code: int = 0
    symbol: str = ""
    fraction_symbol: str = ""
    fractions_per_unit: int = 100
    rounding: Rounding = field(default_factory=Rounding)

    def empty(self) -> bool:
        """True if this currency was default-constructed."""
        return self.name == "" and self.code == ""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Currency):
            return NotImplemented
        # Mirrors C++ ``operator==``: name-based equality (or empty-empty).
        if self.empty() and other.empty():
            return True
        if self.empty() or other.empty():
            return False
        return self.name == other.name

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __hash__(self) -> int:
        # Hash on name to be consistent with __eq__.
        return hash(self.name)

    def __str__(self) -> str:
        return self.code if self.code else "(null currency)"
