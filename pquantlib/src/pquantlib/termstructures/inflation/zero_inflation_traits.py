"""ZeroInflationTraits — bootstrap traits for piecewise zero-inflation curves.

# C++ parity: ql/termstructures/inflation/inflationtraits.hpp (v1.43) —
   ``ZeroInflationTraits`` (static members + ``detail::avgInflation`` /
   ``detail::maxInflation`` constants).

The C++ class templates every guess/bound member on the curve type and
passes ``const C* c``; the Python port does the same, taking the curve
rather than a bare data list. It also carries the trailing
``firstAliveHelper`` argument — ignored here, as in C++, but part of the
protocol ``IterativeBootstrap`` calls through.

Python uses instance methods (rather than C++'s statics) because the
companion ``YoYInflationTraits`` overrides ``initial_value`` (reads
``ts.base_rate()``) and ``update_guess`` (does NOT propagate to
``data[0]``); instance methods make that difference explicit.

Constants:

- ``AVG_INFLATION = 0.02`` — ``detail::avgInflation`` (inflationtraits.hpp:36).
- ``MAX_INFLATION = 0.5`` — ``detail::maxInflation`` (inflationtraits.hpp:37).
- ``MAX_ITERATIONS = 40``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

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

        # C++ parity: ``ZeroInflationTraits::initialDate``
        # (inflationtraits.hpp:45-48).
        """
        return ts.base_date()

    def initial_value(self, ts: ZeroInflationTermStructure) -> float:
        """Value at base date — seeded with ``AVG_INFLATION``, overwritten by bootstrap.

        # C++ parity: ``ZeroInflationTraits::initialValue``
        # (inflationtraits.hpp:49-54). The zero variant has *no*
        # user-supplied base rate; it is solved alongside the other
        # pillars. The YoY variant returns ``ts.baseRate()``.
        """
        del ts
        return AVG_INFLATION

    # ---- iterative guess + bounds ------------------------------------

    def guess(self, i: int, c: Any, valid_data: bool, first_alive_helper: int) -> float:
        """Per-iteration guess for pillar ``i``.

        # C++ parity: ``ZeroInflationTraits::guess``
        # (inflationtraits.hpp:56-64). ``valid_data`` means a previous
        # bootstrap pass solved through — reuse its data slot; otherwise
        # return the average-inflation constant (no extrapolation here,
        # unlike the yield traits).
        """
        del first_alive_helper
        if valid_data:
            return c.data()[i]
        return AVG_INFLATION

    def min_value_after(
        self, i: int, c: Any, valid_data: bool, first_alive_helper: int
    ) -> float:
        """Lower bound for the Brent bracket of pillar ``i``.

        # C++ parity: ``ZeroInflationTraits::minValueAfter``
        # (inflationtraits.hpp:69-80). If a valid prior pass exists, halve
        # the minimum rate (or double it if negative); else
        # ``-MAX_INFLATION``.
        """
        del i, first_alive_helper
        if valid_data:
            r = min(c.data())
            return r * 2.0 if r < 0.0 else r / 2.0
        return -MAX_INFLATION

    def max_value_after(
        self, i: int, c: Any, valid_data: bool, first_alive_helper: int
    ) -> float:
        """Upper bound for the Brent bracket of pillar ``i``.

        # C++ parity: ``ZeroInflationTraits::maxValueAfter``
        # (inflationtraits.hpp:81-94).
        """
        del i, first_alive_helper
        if valid_data:
            r = max(c.data())
            return r / 2.0 if r < 0.0 else r * 2.0
        return MAX_INFLATION

    # ---- unconstrained-optimisation transforms -----------------------

    def transform_direct(self, x: float, i: int, c: Any) -> float:
        """# C++ parity: inflationtraits.hpp:105-108 — identity."""
        del i, c
        return x

    def transform_inverse(self, x: float, i: int, c: Any) -> float:
        """# C++ parity: inflationtraits.hpp:109-112 — identity."""
        del i, c
        return x

    # ---- propagation of the just-solved value ------------------------

    def update_guess(self, data: list[float], level: float, i: int) -> None:
        """Install the just-solved value ``level`` at pillar ``i``.

        # C++ parity: ``ZeroInflationTraits::updateGuess``
        # (inflationtraits.hpp:96-102). When the FIRST pillar (``i == 1``)
        # is solved the value is also propagated to ``data[0]`` — this sets
        # the curve's effective base rate (zero inflation has no
        # user-supplied base rate, so the first solve also pins the base).
        """
        data[i] = level
        if i == 1:
            data[0] = level

    # ---- convergence loop cap ----------------------------------------

    def max_iterations(self) -> int:
        """Upper bound for the convergence loop.

        # C++ parity: ``ZeroInflationTraits::maxIterations``
        # (inflationtraits.hpp:114) — 40.
        """
        return MAX_ITERATIONS
