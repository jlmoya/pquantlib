"""YoYInflationTraits — bootstrap traits for piecewise YoY inflation curves.

# C++ parity: ql/termstructures/inflation/inflationtraits.hpp (v1.43) —
   ``YoYInflationTraits``.

Differences from ``ZeroInflationTraits``:

- ``initial_value`` returns ``ts.base_rate()`` (the user-supplied base YoY
  rate at curve construction), not ``AVG_INFLATION`` — # C++ parity:
  inflationtraits.hpp:128-131.
- ``update_guess`` writes **only** ``data[i]`` — it does NOT propagate the
  just-solved value to ``data[0]``. The YoY curve has a user-supplied base
  rate set at construction; the first-pillar Brent solve must not overwrite
  it — # C++ parity: inflationtraits.hpp:174-178.

Everything else — including the curve-taking ``guess`` / ``min_value_after``
/ ``max_value_after`` signatures and the identity transforms — mirrors the
zero-side traits exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

        # C++ parity: ``YoYInflationTraits::initialDate``
        # (inflationtraits.hpp:123-126).
        """
        return ts.base_date()

    def initial_value(self, ts: YoYInflationTermStructure) -> float:
        """Value at base date — user-supplied base YoY rate.

        # C++ parity: ``YoYInflationTraits::initialValue``
        # (inflationtraits.hpp:128-131). Differs from
        # ``ZeroInflationTraits``: YoY exposes a base rate via the
        # constructor; we read it back via ``ts.base_rate()``.
        """
        return ts.base_rate()

    # ---- iterative guess + bounds (identical to zero traits) ---------

    def guess(self, i: int, c: Any, valid_data: bool, first_alive_helper: int) -> float:
        """Per-iteration guess. # C++ parity: inflationtraits.hpp:134-142."""
        del first_alive_helper
        if valid_data:
            return c.data()[i]
        return AVG_INFLATION

    def min_value_after(
        self, i: int, c: Any, valid_data: bool, first_alive_helper: int
    ) -> float:
        """Lower bound. # C++ parity: inflationtraits.hpp:147-158."""
        del i, first_alive_helper
        if valid_data:
            r = min(c.data())
            return r * 2.0 if r < 0.0 else r / 2.0
        return -MAX_INFLATION

    def max_value_after(
        self, i: int, c: Any, valid_data: bool, first_alive_helper: int
    ) -> float:
        """Upper bound. # C++ parity: inflationtraits.hpp:159-172."""
        del i, first_alive_helper
        if valid_data:
            r = max(c.data())
            return r / 2.0 if r < 0.0 else r * 2.0
        return MAX_INFLATION

    # ---- unconstrained-optimisation transforms -----------------------

    def transform_direct(self, x: float, i: int, c: Any) -> float:
        """# C++ parity: inflationtraits.hpp:181-184 — identity."""
        del i, c
        return x

    def transform_inverse(self, x: float, i: int, c: Any) -> float:
        """# C++ parity: inflationtraits.hpp:185-188 — identity."""
        del i, c
        return x

    # ---- propagation ------------------------------------------------

    def update_guess(self, data: list[float], level: float, i: int) -> None:
        """Install the just-solved value at pillar ``i``.

        # C++ parity: ``YoYInflationTraits::updateGuess``
        # (inflationtraits.hpp:174-178). KEY DIFFERENCE from the zero
        # traits: does NOT propagate to ``data[0]``. YoY curves carry a
        # user-supplied base rate that must be preserved.
        """
        data[i] = level

    def max_iterations(self) -> int:
        """Convergence loop cap. # C++ parity: inflationtraits.hpp:190 — 40."""
        return MAX_ITERATIONS
