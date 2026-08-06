"""FixedLocalVolSurface — local-vol surface from a fixed (t, K) matrix.

# C++ parity: ql/termstructures/volatility/equityfx/fixedlocalvolsurface.hpp +
#             fixedlocalvolsurface.cpp (v1.43).

A local-volatility surface backed by a (times, strikes) grid of fixed
local-vol values: one Linear interpolation in strike per time slice, linear
in time between slices. Used as the leverage-function carrier in the
Heston-SLV calibration pipelines (``HestonSlvFdmModel`` /
``HestonSlvMcModel``).

Two behaviours here are easy to get wrong and are worth stating, because an
earlier version of this port had both wrong:

* ``min_strike()`` / ``max_strike()`` come from the LAST time slice only
  (``strikes_.back()->front()`` / ``->back()``), not from the union of all
  slices. With per-slice strike grids — which is the whole point of the
  per-column constructor — those differ.
* The strike-extrapolation policy applies ONLY between time slices. On a time
  node C++ calls ``localVolInterpol_[idx](strike, true)`` with no clamping, so
  the strike is extrapolated LINEARLY there whatever ``lowerExtrapolation`` /
  ``upperExtrapolation`` say (fixedlocalvolsurface.cpp:139-144). It is an
  asymmetry, not a tidy rule, and it is reproduced rather than smoothed over.

``set_column`` has no C++ counterpart; it exists for ``HestonSlvMcModel``,
whose calibration fills one time slice per step.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from enum import IntEnum
from typing import cast

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.daycounters.year_fraction_to_date import year_fraction_to_date
from pquantlib.math.closeness import close_enough
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.termstructures.volatility.equity_fx.local_vol_term_structure import (
    LocalVolTermStructure,
)
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date


def _as_per_slice_strikes(
    strikes: Sequence[Sequence[float]] | Sequence[float], n_times: int
) -> list[list[float]]:
    """Accept either one shared strike grid or one grid per time slice.

    # C++ parity: the ``const std::vector<Real>&`` constructors replicate the
    # single grid across every slice; the
    # ``std::vector<shared_ptr<vector<Real>>>`` one takes them per slice.
    """
    if strikes and isinstance(strikes[0], (int, float)):
        flat = cast("Sequence[float]", strikes)
        shared = [float(k) for k in flat]
        return [list(shared) for _ in range(n_times)]
    nested = cast("Sequence[Sequence[float]]", strikes)
    return [[float(k) for k in row] for row in nested]


class FixedLocalVolSurface(LocalVolTermStructure):
    """Local-vol surface backed by a (times, strikes) grid of fixed values."""

    class Extrapolation(IntEnum):
        """Strike-extrapolation mode below / above the slice's strike grid.

        # C++ parity: ``FixedLocalVolSurface::Extrapolation``.
        """

        ConstantExtrapolation = 0
        InterpolatorDefaultExtrapolation = 1

    def __init__(
        self,
        *,
        reference_date: Date,
        times: Sequence[float],
        strikes: Sequence[Sequence[float]] | Sequence[float],
        local_vol_matrix: npt.NDArray[np.float64],
        day_counter: DayCounter,
        lower_extrapolation: Extrapolation = Extrapolation.ConstantExtrapolation,
        upper_extrapolation: Extrapolation = Extrapolation.ConstantExtrapolation,
    ) -> None:
        """Construct from explicit times + strike grid(s) + local-vol matrix.

        # C++ parity: the two time-anchored constructors,
        # fixedlocalvolsurface.hpp:47-62. ``strikes`` may be a single grid
        # (shared by every slice) or one grid per slice.

        Parameters
        ----------
        reference_date:
            Anchor date for time-from-reference computations.
        times:
            Time grid (year fractions from ``reference_date``), sorted unique.
        strikes:
            Either one strike grid shared by all slices, or ``len(times)``
            grids, one per slice.
        local_vol_matrix:
            ``(n_strikes, n_times)`` matrix; column ``j`` holds the local vols
            at time ``times[j]`` across ``strikes[j]``.
        day_counter:
            Day counter used by the parent ``TermStructure``.
        lower_extrapolation, upper_extrapolation:
            Strike-extrapolation policy BETWEEN time slices (see the module
            docstring — on a time node the interpolator always extrapolates).
        """
        super().__init__(
            reference_date=reference_date,
            calendar=NullCalendar(),
            day_counter=day_counter,
        )
        qassert.require(len(times) > 0, "empty time grid")
        self._times: list[float] = [float(t) for t in times]
        qassert.require(self._times[0] >= 0.0, "cannot have times[0] < 0")
        self._strikes: list[list[float]] = _as_per_slice_strikes(
            strikes, len(self._times)
        )
        qassert.require(
            len(self._strikes) == len(self._times), "need strikes for every time step"
        )
        self._matrix: npt.NDArray[np.float64] = np.asarray(
            local_vol_matrix, dtype=np.float64
        )
        self._lower_extrapolation: FixedLocalVolSurface.Extrapolation = (
            lower_extrapolation
        )
        self._upper_extrapolation: FixedLocalVolSurface.Extrapolation = (
            upper_extrapolation
        )
        self._max_date: Date = year_fraction_to_date(
            day_counter, reference_date, self._times[-1]
        )
        self._check_surface()
        self._interpolations: list[LinearInterpolation | None] = []
        self._set_interpolation()

    @classmethod
    def from_dates(
        cls,
        *,
        reference_date: Date,
        dates: Sequence[Date],
        strikes: Sequence[Sequence[float]] | Sequence[float],
        local_vol_matrix: npt.NDArray[np.float64],
        day_counter: DayCounter,
        lower_extrapolation: Extrapolation = Extrapolation.ConstantExtrapolation,
        upper_extrapolation: Extrapolation = Extrapolation.ConstantExtrapolation,
    ) -> FixedLocalVolSurface:
        """Construct from dates rather than times.

        # C++ parity: fixedlocalvolsurface.hpp:39-46. ``maxDate()`` is then
        # ``dates.back()`` exactly, not the round-tripped
        # ``year_fraction_to_date`` of the last time.
        """
        qassert.require(len(dates) > 0, "empty date grid")
        qassert.require(
            dates[0] >= reference_date, "cannot have dates[0] < referenceDate"
        )
        times = [day_counter.year_fraction(reference_date, d) for d in dates]
        surface = cls(
            reference_date=reference_date,
            times=times,
            strikes=strikes,
            local_vol_matrix=local_vol_matrix,
            day_counter=day_counter,
            lower_extrapolation=lower_extrapolation,
            upper_extrapolation=upper_extrapolation,
        )
        surface._max_date = dates[-1]
        return surface

    # --- construction helpers ----------------------------------------------

    def _check_surface(self) -> None:
        # C++ parity: ``FixedLocalVolSurface::checkSurface``.
        qassert.require(
            len(self._times) == self._matrix.shape[1],
            "mismatch between date vector and vol matrix colums",
        )
        n_rows = self._matrix.shape[0]
        for strike in self._strikes:
            qassert.require(
                len(strike) == n_rows,
                "mismatch between money-strike vector and vol matrix rows",
            )
        for j in range(1, len(self._times)):
            qassert.require(
                self._times[j] > self._times[j - 1], "dates must be sorted unique!"
            )
        for strike in self._strikes:
            for j in range(1, len(strike)):
                qassert.require(strike[j] >= strike[j - 1], "strikes must be sorted")

    def _set_interpolation(self) -> None:
        # C++ parity: ``setInterpolation<Linear>()`` — one interpolation per
        # time slice over that slice's strikes and the matrix's column.
        self._interpolations = [
            self._build_interpolation(j) for j in range(len(self._times))
        ]
        self.notify_observers()

    def _build_interpolation(self, j: int) -> LinearInterpolation | None:
        """Linear interpolation over slice ``j``, or None if its grid collapsed.

        A slice whose strikes are all equal (which the SLV pipelines produce
        at t = 0) has no interpolation: C++ still constructs one, but its
        slopes are ``0/0``, so every evaluation of it is a NaN. The C++ reader
        never reaches those NaNs on the guarded paths — ``localVolImpl`` short-
        circuits to the matrix's middle row — so this port stores None and
        keeps the short-circuit, rather than manufacturing NaNs to be faithful
        to a value C++ never uses. The one unguarded path raises instead; see
        ``_local_vol_impl``.
        """
        if self._strikes[j][0] >= self._strikes[j][-1]:
            return None
        return LinearInterpolation(
            np.asarray(self._strikes[j], dtype=np.float64),
            np.ascontiguousarray(self._matrix[:, j], dtype=np.float64),
        )

    # --- TermStructure overrides -------------------------------------------

    def max_date(self) -> Date:
        return self._max_date

    def max_time(self) -> float:
        return self._times[-1]

    def min_strike(self) -> float:
        # C++ parity: ``strikes_.back()->front()`` — the LAST slice.
        return self._strikes[-1][0]

    def max_strike(self) -> float:
        # C++ parity: ``strikes_.back()->back()`` — the LAST slice.
        return self._strikes[-1][-1]

    # --- local-vol impl ----------------------------------------------------

    def _slice_value(self, j: int, strike: float) -> float:
        """Slice ``j`` at ``strike``, with the degenerate-grid fallback.

        # C++ parity: the ``strikes_[j]->front() < strikes_[j]->back()``
        # guard — a slice whose strike grid has collapsed is answered with
        # the matrix's MIDDLE row rather than by interpolating.
        """
        interpolation = self._interpolations[j]
        if interpolation is not None:
            return float(interpolation(strike, allow_extrapolation=True))
        return float(self._matrix[self._matrix.shape[0] // 2, j])

    def _local_vol_impl(self, t: float, underlying_level: float) -> float:
        # C++ parity: ``FixedLocalVolSurface::localVolImpl``.
        t = min(self._times[-1], max(t, self._times[0]))
        idx = bisect_left(self._times, t)

        if close_enough(t, self._times[idx]):
            # NOTE: no strike clamping here — see the module docstring.
            return self._slice_value(idx, underlying_level)

        earlier_strike = underlying_level
        later_strike = underlying_level
        if self._lower_extrapolation == FixedLocalVolSurface.Extrapolation.ConstantExtrapolation:
            if underlying_level < self._strikes[idx - 1][0]:
                earlier_strike = self._strikes[idx - 1][0]
            if underlying_level < self._strikes[idx][0]:
                later_strike = self._strikes[idx][0]
        if self._upper_extrapolation == FixedLocalVolSurface.Extrapolation.ConstantExtrapolation:
            if underlying_level > self._strikes[idx - 1][-1]:
                earlier_strike = self._strikes[idx - 1][-1]
            if underlying_level > self._strikes[idx][-1]:
                later_strike = self._strikes[idx][-1]

        early_vol = self._slice_value(idx - 1, earlier_strike)
        # C++ asymmetry: the later leg has no degenerate-grid fallback, so on a
        # collapsed slice it evaluates a 0/0 interpolation and returns NaN.
        later_interpolation = self._interpolations[idx]
        qassert.require(
            later_interpolation is not None,
            f"time slice {idx} has a collapsed strike grid "
            f"({self._strikes[idx][0]}); C++ returns NaN here "
            "(fixedlocalvolsurface.cpp:158 evaluates the slice's Linear "
            "interpolation unguarded, and its slopes are 0/0)",
        )
        assert later_interpolation is not None
        later_vol = float(later_interpolation(later_strike, allow_extrapolation=True))
        return early_vol + (later_vol - early_vol) / (
            self._times[idx] - self._times[idx - 1]
        ) * (t - self._times[idx - 1])

    # --- helpers used by SLV calibration -----------------------------------

    def set_column(
        self, j: int, strikes_j: Sequence[float], col: npt.NDArray[np.float64]
    ) -> None:
        """Overwrite slice ``j`` with new strikes + local-vol values.

        # C++ parity: none — ``HestonSlvMcModel``'s calibration loop fills one
        # slice per time step, which C++ does by rebuilding the surface.
        """
        qassert.require(0 <= j < len(self._times), f"slice index {j} out of range")
        qassert.require(
            len(strikes_j) == self._matrix.shape[0],
            f"new strikes have {len(strikes_j)} entries; "
            f"expected {self._matrix.shape[0]}",
        )
        self._strikes[j] = [float(k) for k in strikes_j]
        self._matrix[:, j] = col
        self._interpolations[j] = self._build_interpolation(j)
        self.notify_observers()


__all__ = ["FixedLocalVolSurface"]
