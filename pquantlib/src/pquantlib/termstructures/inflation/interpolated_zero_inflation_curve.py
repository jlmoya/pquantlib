"""InterpolatedZeroInflationCurve — zero-inflation curve from known (date, rate) nodes.

# C++ parity: ql/termstructures/inflation/interpolatedzeroinflationcurve.hpp
   (v1.42.1) — ``InterpolatedZeroInflationCurve<Interpolator>`` template.

C++ specializes ``InterpolatedZeroInflationCurve<Linear>`` to
``ZeroInflationCurve``; Python takes a ``InterpolationFactory`` callable
mapping ``(xs, ys) → Interpolation`` (default ``LinearInterpolation`` —
matches the C++ ``ZeroInflationCurve`` typedef). The interpolation runs
in *time* space (year fractions from the reference date).

C++ stores the dates twice (once as raw ``dates_`` and once converted to
``times_`` on ``InterpolatedCurve<I>``); we keep the same split for parity.

Per the L7-A divergence policy on ``InflationTermStructure``:
* ``observation_lag`` and ``nominal_term_structure`` are optional opt-in
  fields. C++ deprecated ``observationLag()`` but downstream engines still
  use it, so the L7-A spec re-instates both as typed slots.
* The base date is ``dates[0]`` (matches the C++ constructor).

The C++ constructor pulls ``baseRate`` from ``rates[0]`` implicitly via
``ZeroInflationTermStructure(referenceDate, dates.at(0), ...)`` — but
``baseRate`` actually lives in the ``InflationTermStructure`` slot only
for the YoY abstract case (zero-inflation does not pin a curve-base
rate). For closest parity we forward an explicit ``base_rate=None``
(i.e. ``has_base_rate()`` returns ``False`` on the zero abstract).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.array import Array
from pquantlib.math.interpolations.interpolation import Interpolation
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.termstructures.inflation.seasonality import Seasonality
from pquantlib.termstructures.inflation.zero_inflation_term_structure import (
    ZeroInflationTermStructure,
)
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period

InterpolationFactory = Callable[[Array, Array], Interpolation]


class InterpolatedZeroInflationCurve(ZeroInflationTermStructure):
    """Zero-inflation curve from linear (or other) interpolation of zero rates.

    Inputs:
    - ``reference_date``: today's evaluation date.
    - ``dates``: anchor dates; ``dates[0]`` becomes the curve base date.
    - ``rates``: zero-coupon inflation rates at each anchor.
    - ``frequency``: inflation period frequency (Monthly / Quarterly / ...).
    - ``day_counter``: used to convert anchor dates → times.
    - ``interpolator``: factory ``(xs, ys) → Interpolation``. Default
      ``LinearInterpolation`` (C++ ``ZeroInflationCurve`` typedef).
    - ``seasonality`` / ``observation_lag`` / ``nominal_term_structure``:
      optional, forwarded to the abstract base.
    """

    def __init__(
        self,
        reference_date: Date,
        dates: Sequence[Date],
        rates: Sequence[float],
        frequency: Frequency,
        day_counter: DayCounter,
        seasonality: Seasonality | None = None,
        interpolator: InterpolationFactory = LinearInterpolation,
        calendar: Calendar | None = None,
        observation_lag: Period | None = None,
        nominal_term_structure: YieldTermStructureProtocol | None = None,
    ) -> None:
        qassert.require(len(dates) > 1, f"too few dates: {len(dates)}")
        qassert.require(
            len(rates) == len(dates),
            f"indices/dates count mismatch: {len(rates)} vs {len(dates)}",
        )
        for i in range(1, len(rates)):
            # C++ parity: rates may be < 0 but must be > -1 (>-100%).
            qassert.require(rates[i] > -1.0, "zero inflation data < -100 %")

        super().__init__(
            base_date=dates[0],
            frequency=frequency,
            day_counter=day_counter,
            observation_lag=observation_lag,
            nominal_term_structure=nominal_term_structure,
            seasonality=seasonality,
            reference_date=reference_date,
            calendar=calendar,
        )
        self._dates: list[Date] = list(dates)
        self._data: list[float] = list(rates)
        self._interpolator: InterpolationFactory = interpolator
        # C++ parity: setupTimes(dates_, referenceDate, dayCounter).
        self._times: list[float] = [
            day_counter.year_fraction(reference_date, d) for d in self._dates
        ]
        self._interpolation: Interpolation = self._interpolator(
            np.asarray(self._times, dtype=np.float64),
            np.asarray(self._data, dtype=np.float64),
        )

    # ---- InflationTermStructure interface ----------------------------

    def max_date(self) -> Date:
        """C++ parity: ``maxDate()`` returns the last anchor."""
        return self._dates[-1]

    # ---- ZeroInflationTermStructure implementation -------------------

    def _zero_rate_impl(self, t: float) -> float:
        """C++ parity: ``zeroRateImpl(Time)`` = ``interpolation_(t, true)``."""
        return self._interpolation(t, allow_extrapolation=True)

    # ---- inspectors --------------------------------------------------

    def dates(self) -> list[Date]:
        return list(self._dates)

    def times(self) -> list[float]:
        return list(self._times)

    def data(self) -> list[float]:
        return list(self._data)

    def rates(self) -> list[float]:
        """C++ parity: ``rates()`` is an alias for ``data()`` for zero curves."""
        return list(self._data)

    def nodes(self) -> list[tuple[Date, float]]:
        return list(zip(self._dates, self._data, strict=True))
