"""SwaptionVolatilityMatrix — 2-D bilinear ATM swaption vol surface.

# C++ parity: ql/termstructures/volatility/swaption/swaptionvolmatrix.{hpp,cpp}
# (v1.42.1).

A grid of (option_tenor, swap_tenor) ATM swaption vols. The matrix
is interpreted with rows = option tenors, columns = swap tenors.
C++ uses bilinear interpolation by default (with a separate
``flatExtrapolation`` switch that we expose); PQuantLib uses the
same ``BilinearInterpolation`` from L1-E.

ATM-only — the strike argument is ignored by the
``_volatility_impl`` (the C++ class does the same, modulo the
``smileSectionImpl`` carve-out which lives in the SABR cube
Phase 9 deferral).

The optional ``shifts`` matrix supports Shifted Lognormal vols.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.interpolations.bilinear import BilinearInterpolation
from pquantlib.math.matrix import Matrix
from pquantlib.termstructures.volatility.swaption.swaption_volatility_discrete import (
    SwaptionVolatilityDiscrete,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.period import Period


class SwaptionVolatilityMatrix(SwaptionVolatilityDiscrete):
    """ATM swaption-volatility matrix with bilinear interpolation."""

    def __init__(
        self,
        *,
        business_day_convention: BusinessDayConvention,
        option_tenors: Sequence[Period],
        swap_tenors: Sequence[Period],
        volatilities: Matrix | Sequence[Sequence[float]],
        calendar: Calendar,
        day_counter: DayCounter,
        reference_date: Date | None = None,
        settlement_days: int | None = None,
        flat_extrapolation: bool = False,
        volatility_type: VolatilityType = VolatilityType.ShiftedLognormal,
        shifts: Matrix | Sequence[Sequence[float]] | None = None,
    ) -> None:
        super().__init__(
            business_day_convention=business_day_convention,
            option_tenors=option_tenors,
            swap_tenors=swap_tenors,
            calendar=calendar,
            day_counter=day_counter,
            reference_date=reference_date,
            settlement_days=settlement_days,
        )
        n_opt = len(option_tenors)
        n_swap = len(swap_tenors)
        vols = np.ascontiguousarray(volatilities, dtype=np.float64)
        qassert.require(
            vols.shape == (n_opt, n_swap),
            f"vol matrix shape {vols.shape} must be (n_option_tenors={n_opt}, "
            f"n_swap_tenors={n_swap})",
        )
        if shifts is None:
            shifts_arr = np.zeros((n_opt, n_swap), dtype=np.float64)
        else:
            shifts_arr = np.ascontiguousarray(shifts, dtype=np.float64)
            qassert.require(
                shifts_arr.shape == (n_opt, n_swap),
                f"shifts matrix shape {shifts_arr.shape} must match vols",
            )

        self._volatilities: Matrix = vols.copy()
        self._shifts: Matrix = shifts_arr.copy()
        self._flat_extrapolation: bool = flat_extrapolation
        self._volatility_type: VolatilityType = volatility_type

        times = np.asarray(self._option_times, dtype=np.float64)
        lengths = np.asarray(self._swap_lengths, dtype=np.float64)
        # BilinearInterpolation expects z[y, x]. We have
        # vols[opt_idx, swap_idx] = vols[y, x] with x = swap_length,
        # y = option_time. Matches directly.
        self._vol_interp: BilinearInterpolation = BilinearInterpolation(
            lengths, times, vols
        )
        self._shift_interp: BilinearInterpolation = BilinearInterpolation(
            lengths, times, shifts_arr
        )

    # --- inspectors ------------------------------------------------------

    def volatility_type(self) -> VolatilityType:
        return self._volatility_type

    def volatilities(self) -> Matrix:
        return self._volatilities.copy()

    # --- TermStructure interface -----------------------------------------

    def max_date(self) -> Date:
        return self._option_dates[-1]

    def min_strike(self) -> float:
        return -float("inf")

    def max_strike(self) -> float:
        return float("inf")

    def max_swap_tenor(self) -> Period:
        return self._swap_tenors[-1]

    # --- impl ------------------------------------------------------------

    def _volatility_impl(
        self, option_time: float, swap_length: float, strike: float
    ) -> float:
        _ = strike  # ATM matrix — strike-independent
        # If flat-extrapolation is on, clamp inputs to the pillar grid.
        if self._flat_extrapolation:
            option_time = max(
                self._option_times[0], min(option_time, self._option_times[-1])
            )
            swap_length = max(
                self._swap_lengths[0], min(swap_length, self._swap_lengths[-1])
            )
        return self._vol_interp(swap_length, option_time, allow_extrapolation=True)

    def _shift_impl(self, option_time: float, swap_length: float) -> float:
        if self._flat_extrapolation:
            option_time = max(
                self._option_times[0], min(option_time, self._option_times[-1])
            )
            swap_length = max(
                self._swap_lengths[0], min(swap_length, self._swap_lengths[-1])
            )
        return self._shift_interp(swap_length, option_time, allow_extrapolation=True)
