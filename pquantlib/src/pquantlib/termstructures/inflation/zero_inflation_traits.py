"""ZeroInflationTraits — bootstrap traits for piecewise zero-inflation curves.

# C++ parity: ql/termstructures/inflation/inflationtraits.hpp (v1.42.1) —
   ``ZeroInflationTraits`` (struct of static methods + ``detail::avgInflation``
   / ``detail::maxInflation`` constants).

The C++ struct uses static methods + a curve template parameter; Python
uses an instance with `(i, ts_or_data, valid_data)` methods because:

- Python type erasure already removes the static-template advantage.
- ``IterativeBootstrap`` instantiates the traits once per piecewise curve and
  threads it through the bootstrap loop; instance methods read more naturally.
- The companion ``YoYInflationTraits`` overrides ``initial_value`` (reads
  ``ts.base_rate()``) and ``update_guess`` (does NOT propagate to ``data[0]``);
  instance methods make this difference explicit.

Constants:

- ``AVG_INFLATION = 0.02`` — `detail::avgInflation` (C++).
- ``MAX_INFLATION = 0.5`` — `detail::maxInflation` (C++).
- ``MAX_ITERATIONS = 40``.

# C++ parity: `ZeroInflationTraits::guess` / `minValueAfter` / `maxValueAfter`
# at inflationtraits.hpp:55-115 — same signatures, Python uses lists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from pquantlib.time.date import Date

if TYPE_CHECKING:
    from pquantlib.termstructures.inflation.zero_inflation_term_structure import (
        ZeroInflationTermStructure,
    )


AVG_INFLATION: Final[float] = 0.02
MAX_INFLATION: Final[float] = 0.5
MAX_ITERATIONS: Final[int] = 40


class ZeroInflationTraits:
    """Bootstrap traits for ``PiecewiseZeroInflationCurve``.

    Threaded through ``IterativeBootstrap[ZeroInflationTermStructure,
    ZeroInflationTraits]`` and the piecewise curve.
    """

    # ---- pillar 0 (base date) ----------------------------------------

    def initial_date(self, ts: ZeroInflationTermStructure) -> Date:
        """Start of curve data — the curve's base date.

        # C++ parity: ``ZeroInflationTraits::initialDate`` (inflationtraits.hpp:55).
        """
        return ts.base_date()

    def initial_value(self, ts: ZeroInflationTermStructure) -> float:
        """Value at base date — seeded with ``AVG_INFLATION``, overwritten by bootstrap.

        # C++ parity: ``ZeroInflationTraits::initialValue`` (inflationtraits.hpp:62).
        # The zero variant has *no* user-supplied base rate; it is solved
        # alongside the other pillars. The YoY variant returns ``ts.baseRate()``.
        """
        del ts
        return AVG_INFLATION

    # ---- iterative guess + bounds ------------------------------------

    def guess(
        self,
        i: int,
        data: list[float],
        valid_data: bool,
    ) -> float:
        """Per-iteration guess for pillar ``i``.

        # C++ parity: ``ZeroInflationTraits::guess`` (inflationtraits.hpp:72).
        # ``valid_data`` means a previous bootstrap iteration solved through —
        # reuse its data slot; otherwise return the average-inflation constant.
        """
        if valid_data:
            return data[i]
        return AVG_INFLATION

    def min_value_after(
        self,
        i: int,
        data: list[float],
        valid_data: bool,
    ) -> float:
        """Lower bound for the Brent bracket of pillar ``i``.

        # C++ parity: ``ZeroInflationTraits::minValueAfter`` (inflationtraits.hpp:82).
        # If we have a valid prior pass, halve the minimum non-base rate (or
        # double it if negative); else use ``-MAX_INFLATION``.
        """
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
        """Upper bound for the Brent bracket of pillar ``i``.

        # C++ parity: ``ZeroInflationTraits::maxValueAfter`` (inflationtraits.hpp:99).
        """
        del i
        if valid_data:
            r = max(data)
            return r / 2.0 if r < 0.0 else r * 2.0
        return MAX_INFLATION

    # ---- propagation of the just-solved value ------------------------

    def update_guess(self, data: list[float], level: float, i: int) -> None:
        """Install the just-solved value ``level`` at pillar ``i``.

        # C++ parity: ``ZeroInflationTraits::updateGuess`` (inflationtraits.hpp:108).
        # When the FIRST pillar (``i == 1``) is solved, the value is also
        # propagated to ``data[0]`` — this sets the curve's effective base
        # rate (zero inflation has no user-supplied base rate, so the first
        # solve also pins the base).
        """
        data[i] = level
        if i == 1:
            data[0] = level

    # ---- convergence loop cap ----------------------------------------

    def max_iterations(self) -> int:
        """Upper bound for the convergence loop.

        # C++ parity: ``ZeroInflationTraits::maxIterations`` (inflationtraits.hpp:115).
        """
        return MAX_ITERATIONS
