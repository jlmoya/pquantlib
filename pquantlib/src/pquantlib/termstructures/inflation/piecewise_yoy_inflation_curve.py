"""PiecewiseYoYInflationCurve — YoY curve bootstrapped from market quotes.

# C++ parity: ql/termstructures/inflation/piecewiseyoyinflationcurve.hpp
   (v1.42.1) — ``PiecewiseYoYInflationCurve<Interpolator, Bootstrap, Traits>``
   template, with default ``Bootstrap = IterativeBootstrap`` and
   ``Traits = YoYInflationTraits``.

Same shape as :class:`PiecewiseZeroInflationCurve` but parameterised
with :class:`YoYInflationTraits` (which preserves the user-supplied
base YoY rate during the first-pillar Brent solve).

The constructor takes a mandatory ``base_rate`` (the YoY rate at the
curve base date) because the YoY traits do NOT auto-set ``data[0]`` —
the user must supply the YoY rate observed at curve construction.
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
from pquantlib.termstructures.inflation.interpolated_yoy_inflation_curve import (
    InterpolatedYoYInflationCurve,
    InterpolationFactory,
)
from pquantlib.termstructures.inflation.yoy_inflation_term_structure import (
    YoYInflationTermStructure,
)
from pquantlib.termstructures.inflation.yoy_inflation_traits import YoYInflationTraits
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period


class PiecewiseYoYInflationCurve(InterpolatedYoYInflationCurve):
    """YoY inflation curve bootstrapped from a list of helpers."""

    def __init__(
        self,
        reference_date: Date,
        calendar: Calendar,
        day_counter: DayCounter,
        observation_lag: Period,
        frequency: Frequency,
        base_rate: float,
        instruments: Sequence[BootstrapHelper[YoYInflationTermStructure]],
        nominal_yts: YieldTermStructureProtocol | None = None,
        interpolator: InterpolationFactory = LinearInterpolation,
        accuracy: float = 1.0e-12,
    ) -> None:
        qassert.require(
            len(instruments) > 0,
            "no helpers provided to piecewise YoY inflation curve",
        )
        # C++ parity: curveBase = inflation_period(reference - lag, freq).first.
        base_date, _ = inflation_period(reference_date - observation_lag, frequency)

        seed_dates: list[Date] = [base_date, base_date + 1]
        seed_rates: list[float] = [base_rate, base_rate]
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

        self._instruments: list[BootstrapHelper[YoYInflationTermStructure]] = list(
            instruments
        )
        self._traits: YoYInflationTraits = YoYInflationTraits()
        self._accuracy: float = accuracy
        self._bootstrap()

    # -- IterativeBootstrap protocol ---------------------------------------

    def time_from_reference(self, d: Date) -> float:
        return self.day_counter().year_fraction(self.reference_date(), d)

    def set_data_at(self, i: int, level: float) -> None:
        self._data[i] = level

    def data_live(self) -> list[float]:
        """Return the live ``_data`` list (no defensive copy).

        See :meth:`PiecewiseZeroInflationCurve.data_live` for rationale.
        """
        return self._data

    def refresh_interpolation_through(self, up_to: int) -> None:
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
        self._dates = list(dates)
        self._times = list(times)
        self._data = list(data)

    # -- bootstrap --------------------------------------------------------

    def _bootstrap(self) -> None:
        bootstrapper: IterativeBootstrap[
            YoYInflationTermStructure, YoYInflationTraits
        ] = IterativeBootstrap(
            curve=self,
            instruments=self._instruments,
            traits=self._traits,
            accuracy=self._accuracy,
        )
        bootstrapper.calculate()
        self.refresh_interpolation_through(len(self._data) - 1)

    # -- inspectors -------------------------------------------------------

    def instruments(self) -> list[BootstrapHelper[YoYInflationTermStructure]]:
        return list(self._instruments)

    def accuracy(self) -> float:
        return self._accuracy
