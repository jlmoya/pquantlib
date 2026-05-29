"""YoYInflationTraits — bootstrap traits for piecewise YoY inflation curves.

# C++ parity: ql/termstructures/inflation/inflationtraits.hpp (v1.42.1) —
   ``YoYInflationTraits`` (struct of static methods).

Differences from ``ZeroInflationTraits``:

- ``initial_value`` returns ``ts.base_rate()`` (the user-supplied base YoY
  rate at curve construction), not ``AVG_INFLATION`` — # C++ parity:
  inflationtraits.hpp:130.
- ``update_guess`` writes **only** ``data[i]`` — does NOT propagate the
  just-solved value to ``data[0]``. The YoY curve has a user-supplied base
  rate set at construction; the first-pillar Brent solve must not overwrite
  it — # C++ parity: inflationtraits.hpp:184.

Other constants and methods mirror the zero-side traits exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib.termstructures.inflation.zero_inflation_traits import (
    AVG_INFLATION,
    MAX_INFLATION,
    MAX_ITERATIONS,
)
from pquantlib.time.date import Date

if TYPE_CHECKING:
    from pquantlib.termstructures.inflation.yoy_inflation_term_structure import (
        YoYInflationTermStructure,
    )


class YoYInflationTraits:
    """Bootstrap traits for ``PiecewiseYoYInflationCurve``."""

    # ---- pillar 0 (base date) ----------------------------------------

    def initial_date(self, ts: YoYInflationTermStructure) -> Date:
        """Start of curve data — the curve's base date.

        # C++ parity: ``YoYInflationTraits::initialDate`` (inflationtraits.hpp:123).
        """
        return ts.base_date()

    def initial_value(self, ts: YoYInflationTermStructure) -> float:
        """Value at base date — user-supplied base YoY rate.

        # C++ parity: ``YoYInflationTraits::initialValue`` (inflationtraits.hpp:130).
        # Differs from ``ZeroInflationTraits``: YoY exposes a base rate via
        # the constructor; we read it back via ``ts.base_rate()``.
        """
        return ts.base_rate()

    # ---- iterative guess + bounds (identical to zero traits) ---------

    def guess(
        self,
        i: int,
        data: list[float],
        valid_data: bool,
    ) -> float:
        """Per-iteration guess. # C++ parity: inflationtraits.hpp:140."""
        if valid_data:
            return data[i]
        return AVG_INFLATION

    def min_value_after(
        self,
        i: int,
        data: list[float],
        valid_data: bool,
    ) -> float:
        """Lower bound. # C++ parity: inflationtraits.hpp:150."""
        del i
        if valid_data:
            r = min(data)
            return r * 2.0 if r < 0.0 else r / 2.0
        return -MAX_INFLATION

    def max_value_after(
        self,
        i: int,
        data: list[float],
        valid_data: bool,
    ) -> float:
        """Upper bound. # C++ parity: inflationtraits.hpp:167."""
        del i
        if valid_data:
            r = max(data)
            return r / 2.0 if r < 0.0 else r * 2.0
        return MAX_INFLATION

    # ---- propagation ------------------------------------------------

    def update_guess(self, data: list[float], level: float, i: int) -> None:
        """Install the just-solved value at pillar ``i``.

        # C++ parity: ``YoYInflationTraits::updateGuess`` (inflationtraits.hpp:184).
        # KEY DIFFERENCE from zero traits: does NOT propagate to ``data[0]``.
        # YoY curves carry a user-supplied base rate that must be preserved.
        """
        data[i] = level

    def max_iterations(self) -> int:
        """Convergence loop cap. # C++ parity: inflationtraits.hpp:191."""
        return MAX_ITERATIONS
