"""CapFloorTermVolCurve — 1-D ATM cap/floor vol curve.

# C++ parity: ql/termstructures/volatility/capfloor/capfloortermvolcurve.{hpp,cpp}
# (v1.42.1).

A vector of (option_tenor, vol) pillars. Vols are interpolated on the
*time* axis (year-fraction-from-reference of each tenor's advanced
date). Strike is ignored (the curve is ATM-only).

PQuantLib divergences:

- C++ inherits from ``LazyObject`` + ``CapFloorTermVolatilityStructure``
  to defer interpolation rebuild until first ``calculate()``. PQuantLib
  builds the interpolation eagerly in ``__init__`` (we don't need lazy
  recomputation at this layer — moving-mode invalidation is handled by
  the ``TermStructure`` cache, and the option dates derive from the
  reference date so a fresh interpolation would have to rebuild on
  every eval-date change anyway).
- C++ supports four constructors (Quote/vol x settlement-days/date);
  PQuantLib expects pre-resolved ``float`` vols and lets the caller
  thread Quotes if needed.
- **Interpolator opt-in (L9-A).** C++ uses a templated
  ``Interpolator = Linear`` parameter so callers can pass
  ``Cubic`` etc. PQuantLib exposes the same via an ``interpolator``
  kwarg accepting any ``type[Interpolation]`` (default
  ``LinearInterpolation`` for backward compatibility). Pass
  ``CubicNaturalSpline`` (from L9-A's
  ``pquantlib.math.interpolations.cubic_interpolation``) to match
  the C++ default of the analogous ``Cubic`` template instantiation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.interpolations.interpolation import Interpolation
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.termstructures.volatility.capfloor.cap_floor_term_volatility_structure import (
    CapFloorTermVolatilityStructure,
)
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.period import Period


class CapFloorTermVolCurve(CapFloorTermVolatilityStructure):
    """Cap/floor at-the-money term volatility curve."""

    def __init__(
        self,
        *,
        business_day_convention: BusinessDayConvention,
        option_tenors: Sequence[Period],
        vols: Sequence[float],
        calendar: Calendar,
        day_counter: DayCounter | None = None,
        reference_date: Date | None = None,
        settlement_days: int | None = None,
        interpolator: type[Interpolation] = LinearInterpolation,
    ) -> None:
        dc = day_counter if day_counter is not None else Actual365Fixed()
        super().__init__(
            business_day_convention=business_day_convention,
            reference_date=reference_date,
            calendar=calendar,
            day_counter=dc,
            settlement_days=settlement_days,
        )
        qassert.require(
            len(option_tenors) == len(vols),
            f"mismatch between tenors ({len(option_tenors)}) and vols ({len(vols)})",
        )
        qassert.require(len(option_tenors) > 0, "no option tenors given")

        self._option_tenors: list[Period] = list(option_tenors)
        self._vols: list[float] = [float(v) for v in vols]

        ref = self.reference_date()
        option_dates: list[Date] = []
        option_times: list[float] = []
        for i, p in enumerate(self._option_tenors):
            d = self.option_date_from_tenor(p)
            t = dc.year_fraction(ref, d)
            qassert.require(
                i == 0 or t > option_times[-1],
                f"option dates must be sorted unique; t[{i}]={t}",
            )
            option_dates.append(d)
            option_times.append(t)

        self._option_dates: list[Date] = option_dates
        self._option_times: list[float] = option_times
        self._interpolation: Interpolation = interpolator(
            np.asarray(option_times, dtype=np.float64),
            np.asarray(self._vols, dtype=np.float64),
        )

    # --- inspectors ------------------------------------------------------

    def option_tenors(self) -> list[Period]:
        return list(self._option_tenors)

    def option_dates(self) -> list[Date]:
        return list(self._option_dates)

    def option_times(self) -> list[float]:
        return list(self._option_times)

    # --- TermStructure interface -----------------------------------------

    def max_date(self) -> Date:
        return self._option_dates[-1]

    def min_strike(self) -> float:
        return -math.inf

    def max_strike(self) -> float:
        return math.inf

    def _volatility_impl(self, t: float, strike: float) -> float:
        _ = strike  # ATM-only
        return self._interpolation(t, allow_extrapolation=True)
