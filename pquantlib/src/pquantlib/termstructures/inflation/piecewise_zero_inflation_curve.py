"""PiecewiseZeroInflationCurve — zero curve bootstrapped from market quotes.

# C++ parity: ql/termstructures/inflation/piecewisezeroinflationcurve.hpp
   (v1.42.1) — ``PiecewiseZeroInflationCurve<Interpolator, Bootstrap, Traits>``
   template, with default ``Bootstrap = IterativeBootstrap`` and
   ``Traits = ZeroInflationTraits``.

The curve extends :class:`InterpolatedZeroInflationCurve` with an
``IterativeBootstrap`` over a list of inflation-swap bootstrap helpers
(e.g. :class:`ZeroCouponInflationSwapHelper`). At construction the curve
holds only one node (the base-date pillar); the bootstrap is run lazily
on first access.

Closes the L7-Bb carve-out from Phase 7.

Design notes (PQuantLib-specific):

- The constructor takes ``observation_lag`` per the L8-A design spec and
  computes the curve base date as ``inflation_period(reference_date -
  observation_lag, frequency).first`` — # C++ parity with the
  ``piecewisezeroinflationcurve.hpp`` overload that derives the base date
  from the lag, used in the L7-B probe (``curveBase`` variable).
- A second constructor classmethod ``from_base_date`` accepts the base
  date directly (mirrors the alternative C++ overload).
- The ``nominal_yts`` parameter is stored on the parent class slot
  (``nominal_term_structure``) and is forwarded to helpers via
  ``set_term_structure`` if they consult it. The bootstrap itself does
  not need a nominal curve — the inflation swap fair-rate cancels equal
  discount factors on both legs.
- The bootstrap is run **eagerly at construction** rather than lazily;
  Python's lack of a clean LazyObject equivalent makes the lazy path
  more error-prone, and the eager run keeps observability simple.
- ``base_rate=None`` means the bootstrap will pin the base rate from
  data[0] (which the traits propagate via ``update_guess`` for i=1).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.inflation.inflation_index import inflation_period
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.termstructures.bootstrap.iterative_bootstrap import IterativeBootstrap
from pquantlib.termstructures.bootstrap_helper import BootstrapHelper
from pquantlib.termstructures.inflation.interpolated_zero_inflation_curve import (
    InterpolatedZeroInflationCurve,
    InterpolationFactory,
)
from pquantlib.termstructures.inflation.zero_inflation_term_structure import (
    ZeroInflationTermStructure,
)
from pquantlib.termstructures.inflation.zero_inflation_traits import (
    AVG_INFLATION,
    ZeroInflationTraits,
)
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period


class PiecewiseZeroInflationCurve(InterpolatedZeroInflationCurve):
    """Zero-inflation curve bootstrapped from a list of helpers."""

    def __init__(
        self,
        reference_date: Date,
        calendar: Calendar,
        day_counter: DayCounter,
        observation_lag: Period,
        frequency: Frequency,
        instruments: Sequence[BootstrapHelper[ZeroInflationTermStructure]],
        nominal_yts: YieldTermStructureProtocol | None = None,
        base_rate: float | None = None,
        interpolator: InterpolationFactory = LinearInterpolation,
        accuracy: float = 1.0e-12,
    ) -> None:
        qassert.require(
            len(instruments) > 0,
            "no helpers provided to piecewise inflation curve",
        )
        # C++ parity: curveBase = inflation_period(reference_date - lag, freq).first.
        base_date, _ = inflation_period(reference_date - observation_lag, frequency)

        # Seed the parent with a 2-point grid: (base_date, AVG_INFLATION),
        # (base_date + 1d, AVG_INFLATION). The bootstrap will overwrite
        # everything after we re-install the full grid below.
        seed_rate = base_rate if base_rate is not None else AVG_INFLATION
        seed_dates: list[Date] = [base_date, base_date + 1]
        seed_rates: list[float] = [seed_rate, seed_rate]
        super().__init__(
            reference_date=reference_date,
            dates=seed_dates,
            rates=seed_rates,
            frequency=frequency,
            day_counter=day_counter,
            interpolator=interpolator,
            calendar=calendar,
            observation_lag=observation_lag,
            nominal_term_structure=nominal_yts,
        )

        self._instruments: list[BootstrapHelper[ZeroInflationTermStructure]] = list(
            instruments
        )
        self._traits: ZeroInflationTraits = ZeroInflationTraits()
        self._accuracy: float = accuracy

        # Run bootstrap eagerly.  Lazy bootstrap (matching C++ LazyObject)
        # is a documented PQuantLib divergence — see module docstring.
        self._bootstrap()

    # -- IterativeBootstrap protocol ---------------------------------------

    def time_from_reference(self, d: Date) -> float:
        """C++ parity: ``TermStructure::timeFromReference``."""
        return self.day_counter().year_fraction(self.reference_date(), d)

    def set_data_at(self, i: int, level: float) -> None:
        """Install a single bootstrapped value at pillar ``i``.

        Bootstrap-internal — IterativeBootstrap calls this after each
        Brent solve.
        """
        self._data[i] = level

    def data_live(self) -> list[float]:
        """Return the live ``_data`` list (no defensive copy).

        Part of the ``BootstrapCurveProtocol`` — distinct from the public
        :meth:`data` (which returns a copy). Bootstrap-internal callers
        write into the returned list via traits ``update_guess``; the
        live aliasing is what makes the mutation visible to the curve's
        interpolation refresh.
        """
        return self._data

    def refresh_interpolation_through(self, up_to: int) -> None:
        """Rebuild interpolation over the first ``up_to + 1`` nodes.

        # C++ parity: ``ts.setInterpolation(interpolator.interpolate(
        # times.subspan(0, up_to+1), data.subspan(0, up_to+1)))``.
        Part of the ``BootstrapCurveProtocol`` consumed by
        :class:`IterativeBootstrap`.
        """
        partial_times = self._times[: up_to + 1]
        partial_data = self._data[: up_to + 1]
        self._interpolation = self._interpolator(
            np.asarray(partial_times, dtype=np.float64),
            np.asarray(partial_data, dtype=np.float64),
        )

    def bootstrap_install_grid(
        self,
        dates: list[Date],
        times: list[float],
        data: list[float],
    ) -> None:
        """Install the n+1 dates/times/data grid before the Brent loop.

        Part of the ``BootstrapCurveProtocol`` consumed by
        :class:`IterativeBootstrap`.
        """
        self._dates = list(dates)
        self._times = list(times)
        self._data = list(data)

    # -- bootstrap entry --------------------------------------------------

    def _bootstrap(self) -> None:
        """Run the iterative bootstrap.

        Wires the helpers to ``self`` and dispatches to
        :class:`IterativeBootstrap` for the algorithm.
        """
        bootstrapper: IterativeBootstrap[
            ZeroInflationTermStructure, ZeroInflationTraits
        ] = IterativeBootstrap(
            curve=self,
            instruments=self._instruments,
            traits=self._traits,
            accuracy=self._accuracy,
        )
        bootstrapper.calculate()
        # Refresh the full interpolation in case the last partial-refresh
        # left it scoped to a subset of the data.
        self.refresh_interpolation_through(len(self._data) - 1)

    # -- inspectors -------------------------------------------------------

    def instruments(self) -> list[BootstrapHelper[ZeroInflationTermStructure]]:
        """Return the helper list (defensive copy)."""
        return list(self._instruments)

    def accuracy(self) -> float:
        return self._accuracy
