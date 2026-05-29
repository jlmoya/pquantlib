"""FixedLocalVolSurface — local-vol surface from a fixed (t, K) matrix.

# C++ parity: ql/termstructures/volatility/equityfx/fixedlocalvolsurface.hpp
# (v1.42.1).

A local-volatility surface backed by a (times, strikes) grid of fixed
local-vol values plus bilinear (linear-in-strike, linear-in-time)
interpolation. Used as the leverage-function carrier in the Heston-SLV
calibration pipelines (``HestonSlvFdmModel`` / ``HestonSlvMcModel``):
the calibration writes the leverage values into the matrix and the
surface answers ``local_vol(t, K)`` queries during simulation.

L11-W1-D scope: ports the per-column (strikes can differ per time
slice) form used by the C++ SLV models. ``set_interpolation`` is a
single linear interpolator across strikes; the time dimension uses
linear interpolation between adjacent slices.

Divergences from C++:
- The C++ ``Extrapolation`` enum has ``ConstantExtrapolation`` and
  ``InterpolatorDefaultExtrapolation`` modes. The Python port
  hard-codes constant extrapolation in both strike and time
  directions — the SLV calibration is the only caller and uses
  constant extrapolation by default.
- The C++ class is a ``LazyObject`` (recomputes interpolators on
  update); pquantlib's port reads the matrix directly on each call.
  Performance impact is negligible for the L11-W1-D tests.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.termstructures.volatility.equity_fx.local_vol_term_structure import (
    LocalVolTermStructure,
)
from pquantlib.time.date import Date


class FixedLocalVolSurface(LocalVolTermStructure):
    """Local-vol surface backed by a (times, strikes) grid of fixed values."""

    def __init__(
        self,
        *,
        reference_date: Date,
        times: list[float],
        strikes: list[list[float]],
        local_vol_matrix: npt.NDArray[np.float64],
        day_counter: DayCounter,
    ) -> None:
        """Construct from explicit times + per-time strike arrays + lv matrix.

        # C++ parity: fixedlocalvolsurface.hpp:55-62 (per-column strikes form).

        Parameters
        ----------
        reference_date:
            Anchor date for time-from-reference computations.
        times:
            Time grid (year fractions from ``reference_date``).
        strikes:
            ``len(times)`` arrays of strikes; each one is the strike
            grid at the corresponding time slice.
        local_vol_matrix:
            ``(n_strikes, n_times)`` matrix; column ``j`` holds the
            local vols at time ``times[j]`` across ``strikes[j]``.
        day_counter:
            Day counter used by the parent ``TermStructure`` for
            time-from-reference.
        """
        super().__init__(reference_date=reference_date, day_counter=day_counter)
        qassert.require(len(times) > 0, "empty time grid")
        qassert.require(len(strikes) == len(times), "strike/time count mismatch")
        # All strike arrays must have the same length (matches the matrix shape).
        n_strikes = local_vol_matrix.shape[0]
        for j, sj in enumerate(strikes):
            qassert.require(
                len(sj) == n_strikes,
                f"strikes[{j}] has {len(sj)} entries; expected {n_strikes}",
            )
        qassert.require(
            local_vol_matrix.shape[1] == len(times),
            f"matrix has {local_vol_matrix.shape[1]} columns; "
            f"expected {len(times)}",
        )
        self._times: list[float] = list(times)
        self._strikes: list[list[float]] = [list(s) for s in strikes]
        self._matrix: npt.NDArray[np.float64] = np.asarray(
            local_vol_matrix, dtype=np.float64
        )
        # Precompute the min/max strike across all slices.
        self._min_strike: float = min(s[0] for s in self._strikes)
        self._max_strike: float = max(s[-1] for s in self._strikes)

    # --- TermStructure overrides ----------------------------------------

    def max_date(self) -> Date:
        """Maximum date supported by the surface.

        Returns a sentinel max-date — the surface is queried only by
        time inside the SLV pipelines.
        """
        return Date.max_date()

    def min_strike(self) -> float:
        return self._min_strike

    def max_strike(self) -> float:
        return self._max_strike

    # --- local-vol impl -------------------------------------------------

    def _local_vol_at_slice(self, j: int, strike: float) -> float:
        """Linear interpolation in strike across slice ``j``.

        Constant extrapolation at the boundaries.
        """
        s = self._strikes[j]
        col = self._matrix[:, j]
        if strike <= s[0]:
            return float(col[0])
        if strike >= s[-1]:
            return float(col[-1])
        # Find the interval.
        i = 0
        n = len(s)
        for k in range(n - 1):
            if s[k] <= strike <= s[k + 1]:
                i = k
                break
        x0, x1 = s[i], s[i + 1]
        y0, y1 = float(col[i]), float(col[i + 1])
        return y0 + (y1 - y0) * (strike - x0) / (x1 - x0)

    def _local_vol_impl(self, t: float, underlying_level: float) -> float:
        """Bilinear interpolation in (time, strike).

        Constant extrapolation at the boundaries.

        # C++ parity: fixedlocalvolsurface.cpp ``localVolImpl`` —
        # bilinear or linear-in-time of the per-slice linear-in-strike
        # interpolator. The Python port inlines both directions.
        """
        n = len(self._times)
        if t <= self._times[0] or n == 1:
            return self._local_vol_at_slice(0, underlying_level)
        if t >= self._times[-1]:
            return self._local_vol_at_slice(n - 1, underlying_level)
        # Find time interval.
        i = 0
        for k in range(n - 1):
            if self._times[k] <= t <= self._times[k + 1]:
                i = k
                break
        t0, t1 = self._times[i], self._times[i + 1]
        v0 = self._local_vol_at_slice(i, underlying_level)
        v1 = self._local_vol_at_slice(i + 1, underlying_level)
        # Linear in time.
        if math.isclose(t1, t0):
            return v0
        return v0 + (v1 - v0) * (t - t0) / (t1 - t0)

    # --- helpers used by SLV calibration --------------------------------

    def set_column(
        self, j: int, strikes_j: list[float], col: npt.NDArray[np.float64]
    ) -> None:
        """Overwrite column ``j`` with new strikes + local-vol values.

        Used by HestonSlvMcModel's calibration loop which fills one
        column per time step.
        """
        qassert.require(0 <= j < len(self._times), f"slice index {j} out of range")
        qassert.require(
            len(strikes_j) == self._matrix.shape[0],
            f"new strikes have {len(strikes_j)} entries; "
            f"expected {self._matrix.shape[0]}",
        )
        self._strikes[j] = list(strikes_j)
        self._matrix[:, j] = col
        # Refresh min/max strike.
        self._min_strike = min(s[0] for s in self._strikes)
        self._max_strike = max(s[-1] for s in self._strikes)
        self.notify_observers()


__all__ = ["FixedLocalVolSurface"]
