"""PiecewiseYoYOptionletVolatilityCurve — bootstrapped YoY vol curve.

# C++ parity: ql/experimental/inflation/piecewiseyoyoptionletvolatility.hpp
   (v1.42.1) — ``YoYInflationVolatilityTraits`` +
   ``PiecewiseYoYOptionletVolatilityCurve<Interpolator, Bootstrap, Traits>``.

A flat-smile YoY optionlet vol curve bootstrapped from
:class:`YoYOptionletHelper` instruments at a constant strike. Most of the
machinery is shared with :class:`InterpolatedYoYOptionletVolatilityCurve`;
this adds the bootstrap (the early pillars are usually pure assumption,
seeded by the base vol level).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.experimental.inflation.yoy_inflation_optionlet_volatility_structure2 import (
    InterpolatedYoYOptionletVolatilityCurve,
    Interpolation1DFactory,
)
from pquantlib.experimental.inflation.yoy_optionlet_helpers import YoYOptionletHelper
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.termstructures.bootstrap.iterative_bootstrap import IterativeBootstrap
from pquantlib.termstructures.volatility.inflation.yoy_optionlet_volatility_surface import (
    YoYOptionletVolatilitySurface,
)
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period


class YoYInflationVolatilityTraits:
    """Traits for the YoY inflation-volatility bootstrap.

    # C++ parity: ``YoYInflationVolatilityTraits``
    # (piecewiseyoyoptionletvolatility.hpp:36-96).
    """

    def initial_date(self, curve: YoYOptionletVolatilitySurface) -> Date:
        return curve.base_date()

    def initial_value(self, curve: YoYOptionletVolatilitySurface) -> float:
        # REALLY important: the base vol embodies assumptions on the early
        # (unquoted) options.
        return curve.base_level()

    def guess(self, i: int, data: Sequence[float], valid_data: bool) -> float:
        if valid_data:
            return data[i]
        if i == 1:
            return 0.005
        return 0.002

    def min_value_after(self, i: int, data: Sequence[float], valid_data: bool) -> float:
        del valid_data
        return max(0.0, data[i - 1] - 0.02)  # vol cannot be negative

    def max_value_after(self, i: int, data: Sequence[float], valid_data: bool) -> float:
        del valid_data
        return data[i - 1] + 0.02

    def update_guess(self, data: list[float], level: float, i: int) -> None:
        data[i] = level

    def max_iterations(self) -> int:
        return 25


class PiecewiseYoYOptionletVolatilityCurve(InterpolatedYoYOptionletVolatilityCurve):
    """Piecewise (bootstrapped) flat-smile YoY optionlet vol curve."""

    def __init__(
        self,
        settlement_days: int,
        calendar: Calendar,
        bdc: BusinessDayConvention,
        day_counter: DayCounter,
        lag: Period,
        frequency: Frequency,
        index_is_interpolated: bool,
        min_strike: float,
        max_strike: float,
        base_yoy_volatility: float,
        instruments: Sequence[YoYOptionletHelper],
        accuracy: float = 1.0e-12,
        interpolator: Interpolation1DFactory = LinearInterpolation,
    ) -> None:
        qassert.require(
            len(instruments) > 0,
            "no helpers provided to piecewise YoY optionlet vol curve",
        )
        # Seed the surface with no data (just the base vol level) — the
        # bootstrap fills the pillars. This mirrors the data-less
        # `InterpolatedYoYOptionletVolatilityCurve.for_bootstrap` path
        # (the C++ piecewise curve uses the protected base constructor).
        YoYOptionletVolatilitySurface.__init__(
            self,
            settlement_days=settlement_days,
            calendar=calendar,
            business_day_convention=bdc,
            day_counter=day_counter,
            observation_lag=lag,
            frequency=frequency,
            index_is_interpolated=index_is_interpolated,
        )
        self._interpolator = interpolator
        self._dates = []
        self._min_strike = min_strike
        self._max_strike = max_strike
        self._times = []
        self._data = []
        self._nodes = []
        self._interpolation = None  # type: ignore[assignment]
        self._set_base_level(base_yoy_volatility)

        self._instruments: list[YoYOptionletHelper] = list(instruments)
        self._traits: YoYInflationVolatilityTraits = YoYInflationVolatilityTraits()
        self._accuracy: float = accuracy
        self._bootstrap()

    # -- IterativeBootstrap curve protocol ---------------------------------

    def set_data_at(self, i: int, level: float) -> None:
        self._data[i] = level

    def data_live(self) -> list[float]:
        return self._data

    def refresh_interpolation_through(self, up_to: int) -> None:
        partial_times = self._times[: up_to + 1]
        partial_data = self._data[: up_to + 1]
        self._interpolation = self._interpolator(
            np.asarray(partial_times, dtype=np.float64),
            np.asarray(partial_data, dtype=np.float64),
        )

    def bootstrap_install_grid(
        self, dates: list[Date], times: list[float], data: list[float]
    ) -> None:
        self._dates = list(dates)
        self._times = list(times)
        self._data = list(data)
        self._nodes = list(zip(self._dates, self._data, strict=True))

    # -- bootstrap ---------------------------------------------------------

    def _bootstrap(self) -> None:
        bootstrapper: IterativeBootstrap[
            YoYOptionletVolatilitySurface, YoYInflationVolatilityTraits
        ] = IterativeBootstrap(
            curve=self,
            instruments=self._instruments,
            traits=self._traits,
            accuracy=self._accuracy,
        )
        bootstrapper.calculate()
        self.refresh_interpolation_through(len(self._data) - 1)
        self._nodes = list(zip(self._dates, self._data, strict=True))

    # -- inspectors --------------------------------------------------------

    def instruments(self) -> list[YoYOptionletHelper]:
        return list(self._instruments)
