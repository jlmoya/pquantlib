"""VolatilityCube — generic interest-rate (optionlet/swaption) vol cube base.

# C++ parity: ql/experimental/volatility/volcube.{hpp,cpp} (v1.42.1).

A thin aggregation container holding a list of
:class:`InterestRateVolSurface` (the smile surfaces, one per index
tenor) plus a list of :class:`AbcdAtmVolCurve` (the ATM curves). It is
the experimental generic base for a vol cube (expiry x tenor x
strike-spread) — distinct from the core ``SwaptionVolatilityCube``.

Divergences from C++:

* **No ``Handle`` wrapper.** PQuantLib has no ``Handle`` indirection;
  the surfaces / curves are held directly. Observability, where needed,
  is the caller's responsibility.
* **``minIndexTenor`` / ``maxIndexTenor`` are declared-but-undefined in
  C++ v1.42.1** (no out-of-line definition exists — linking against
  them fails). PQuantLib therefore does not provide them; the
  ``surfaces`` / ``curves`` inspectors are the usable surface of the
  class. The "sort increasing index tenor" comment in the C++ ctor is
  likewise a no-op placeholder.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.experimental.volatility.interest_rate_vol_surface import (
    InterestRateVolSurface,
)

if TYPE_CHECKING:
    from pquantlib.experimental.volatility.abcd_atm_vol_curve import AbcdAtmVolCurve


class VolatilityCube:
    """Generic vol-cube aggregation of IR vol surfaces + ATM curves."""

    def __init__(
        self,
        surfaces: Sequence[InterestRateVolSurface],
        curves: Sequence[AbcdAtmVolCurve],
    ) -> None:
        self._surfaces: list[InterestRateVolSurface] = list(surfaces)
        self._curves: list[AbcdAtmVolCurve] = list(curves)
        qassert.require(len(self._surfaces) > 1, "at least 2 surfaces are needed")

        ref_date = self._surfaces[0].reference_date()
        for surface in self._surfaces:
            qassert.require(
                surface.reference_date() == ref_date, "different reference dates"
            )
        for curve in self._curves:
            qassert.require(
                curve.reference_date() == ref_date, "different reference dates"
            )
        # C++: "sort increasing index tenor" — a no-op placeholder.

    def surfaces(self) -> list[InterestRateVolSurface]:
        return self._surfaces

    def curves(self) -> list[AbcdAtmVolCurve]:
        return self._curves


__all__ = ["VolatilityCube"]
