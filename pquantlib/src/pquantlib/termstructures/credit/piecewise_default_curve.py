"""PiecewiseDefaultCurve — bootstrap default-probability curve.

# C++ parity: ql/termstructures/credit/piecewisedefaultcurve.hpp (v1.42.1).

C++ ``PiecewiseDefaultCurve<Traits, Interpolator, Bootstrap>`` is the
core piecewise bootstrap default-probability curve. It chains
``InterpolatedXxxCurve<Interpolator>`` (Xxx in {SurvivalProbability,
HazardRate, DefaultDensity}) with a per-trait bootstrap iterator that
solves for one pillar at a time given a list of ``BootstrapHelper``
instances.

This class is **scaffolding only** in this stage — full bootstrap is
deferred along with ``PiecewiseYieldCurve`` per Phase 2 carve-out
(see ``docs/migration/phase2-completion.md``).

The scaffolding constructor stores the inputs (traits class, helpers,
reference date, day counter) and exposes the helpers + traits +
data accessors. Calling ``survival_probability`` / ``hazard_rate`` /
``default_density`` raises ``LibraryException`` until the
``PiecewiseYieldCurve``-style ``IterativeBootstrap`` lands.
"""

from __future__ import annotations

from collections.abc import Sequence

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.termstructures.credit.default_probability_helpers import CdsHelper
from pquantlib.termstructures.credit.default_probability_term_structure import (
    DefaultProbabilityTermStructure,
)
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date


class PiecewiseDefaultCurve(DefaultProbabilityTermStructure):
    """Scaffold piecewise default-probability curve.

    Once an iterative bootstrap is implemented, this class will route
    ``survival_probability`` / ``hazard_rate`` / ``default_density``
    queries through the underlying ``InterpolatedXxxCurve``. For now,
    those methods raise ``LibraryException`` directing callers to the
    pending follow-up.
    """

    def __init__(
        self,
        traits: type,
        reference_date: Date,
        instruments: Sequence[CdsHelper],
        day_counter: DayCounter,
        calendar: Calendar | None = None,
    ) -> None:
        super().__init__(
            reference_date=reference_date,
            calendar=calendar,
            day_counter=day_counter,
        )
        qassert.require(
            len(instruments) >= 1,
            "PiecewiseDefaultCurve: at least one instrument is required",
        )
        self._traits: type = traits
        self._instruments: list[CdsHelper] = list(instruments)
        self._bootstrapped: bool = False

    # ---- inspectors -------------------------------------------------------

    def traits(self) -> type:
        return self._traits

    def instruments(self) -> list[CdsHelper]:
        return list(self._instruments)

    def max_date(self) -> Date:
        # Until bootstrap lands, fall back to the latest helper date.
        return max(h.latest_date() for h in self._instruments)

    # ---- DefaultProbabilityTermStructure not-yet-implemented hooks -------

    def _survival_probability_impl(self, t: float) -> float:
        del t
        qassert.fail(
            "PiecewiseDefaultCurve: bootstrap is deferred — see Phase 8 carve-out. "
            "Use FlatHazardRate / InterpolatedSurvivalProbabilityCurve directly "
            "until the iterative bootstrap lands.",
        )
        return 0.0

    def _default_density_impl(self, t: float) -> float:
        del t
        qassert.fail(
            "PiecewiseDefaultCurve: bootstrap is deferred — see Phase 8 carve-out.",
        )
        return 0.0


__all__ = ["PiecewiseDefaultCurve"]
