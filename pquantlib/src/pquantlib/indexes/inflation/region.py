"""Region — geographical area, used for inflation indexes.

# C++ parity: ql/indexes/region.{hpp,cpp} (v1.42.1) — class Region holds
   ``(name, code)`` strings, with one C++ subclass per region (EURegion,
   FranceRegion, UKRegion, USRegion, ...).

Python divergence: rather than one class per region, we expose an
``IntEnum`` whose members carry a ``(name, code)`` payload via a frozen
helper class and accessor methods. This:

- preserves the C++ ``name() / code()`` accessors verbatim, so
  ``InflationIndex.name()`` reproduces ``"<region.name()> <familyName>"``
  exactly (e.g. ``"EU HICP"`` from EUHICP + ``Region.Europe``);
- is hashable and orderable for free (IntEnum), so ``Region`` can land in
  registries / dataclasses without extra glue;
- allows L7-A region-concrete subclasses to write
  ``Region.Europe`` instead of ``EURegion()``.

The ``CustomRegion`` C++ constructor (user-supplied name + code) is
deferred to Phase 8+ — it has zero callers in the must-port surface for
Phase 7.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final


class Region(IntEnum):
    """Enum tagging the geographical/economic region of an inflation index.

    # C++ parity: each member corresponds to one of the QuantLib subclasses
    # in ``ql/indexes/region.hpp``. The ``name`` and ``code`` payloads come
    # from ``ql/indexes/region.cpp`` and are exposed via ``region_name()`` /
    # ``region_code()`` methods that map name → name() / code → code().
    """

    Europe = 1
    France = 2
    UnitedKingdom = 3
    UnitedStates = 4

    def region_name(self) -> str:
        """Return the C++-equivalent ``Region::name()`` string."""
        return _NAMES[self]

    def region_code(self) -> str:
        """Return the C++-equivalent ``Region::code()`` string."""
        return _CODES[self]


# C++ parity: matches the static Data initializers in ql/indexes/region.cpp.
_NAMES: Final[dict[Region, str]] = {
    Region.Europe: "EU",
    Region.France: "France",
    Region.UnitedKingdom: "UK",
    Region.UnitedStates: "USA",
}

_CODES: Final[dict[Region, str]] = {
    Region.Europe: "EU",
    Region.France: "FR",
    Region.UnitedKingdom: "UK",
    Region.UnitedStates: "US",
}
