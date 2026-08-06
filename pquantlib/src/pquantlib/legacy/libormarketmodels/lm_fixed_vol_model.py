"""LmFixedVolatilityModel — piecewise-constant, time-homogeneous caplet vols.

# C++ parity: ql/legacy/libormarketmodels/lmfixedvolmodel.{hpp,cpp} (v1.43).

Given a volatility array and a strictly increasing start-time grid, the
volatility of forward ``i`` at time ``t`` is ``volatilities[i - ti]``, where
``ti`` is the index of the bucket containing ``t``. Forwards below ``ti`` have
already fixed and get zero. The model carries NO calibration arguments
(``n_arguments == 0``) and does not implement ``integrated_variance`` — the
base-class failure is what routes ``LfmCovarianceProxy`` into its numerical
integration branch.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.legacy.libormarketmodels.lm_vol_model import LmVolatilityModel
from pquantlib.math.array import Array


class LmFixedVolatilityModel(LmVolatilityModel):
    """Constant-volatility model over a start-time grid.

    # C++ parity: ``class LmFixedVolatilityModel`` (lmfixedvolmodel.hpp:31-43).
    """

    def __init__(self, volatilities: Array, start_times: Sequence[float]) -> None:
        # C++ parity: lmfixedvolmodel.cpp:25-38.
        super().__init__(len(start_times), 0)
        self._volatilities: Array = np.asarray(volatilities, dtype=np.float64).copy()
        self._start_times: list[float] = list(start_times)

        qassert.require(len(self._start_times) > 1, "too few dates")
        qassert.require(
            self._volatilities.size == len(self._start_times),
            "volatility array and fixing time array have to have the same size",
        )
        for i in range(1, len(self._start_times)):
            qassert.require(
                self._start_times[i] > self._start_times[i - 1],
                f"invalid time ({self._start_times[i]}, vs {self._start_times[i - 1]})",
            )

    # --- inspectors -------------------------------------------------------

    def volatilities(self) -> Array:
        """The volatility array, copied.

        Python addition: C++ keeps ``volatilities_`` private with no accessor.
        """
        return self._volatilities.copy()

    def start_times(self) -> list[float]:
        """The start-time grid, copied.

        Python addition: C++ keeps ``startTimes_`` private with no accessor.
        """
        return list(self._start_times)

    # --- volatility -------------------------------------------------------

    def _bucket(self, t: float) -> int:
        """Index of the bucket containing ``t``.

        # C++ parity: the shared prologue of lmfixedvolmodel.cpp:40-55 and
        # :57-67 —
        #     upper_bound(startTimes_.begin(), startTimes_.end() - 1, t)
        #         - startTimes_.begin() - 1
        # Note the search range EXCLUDES the last start time, which is what
        # pins t == startTimes_.back() into the second-to-last bucket instead
        # of running off the end.
        """
        qassert.require(
            self._start_times[0] <= t <= self._start_times[-1],
            "invalid time given for volatility model",
        )
        return bisect.bisect_right(self._start_times, t, hi=len(self._start_times) - 1) - 1

    def volatility(self, t: float, x: Array | None = None) -> Array:
        """# C++ parity: lmfixedvolmodel.cpp:40-55."""
        ti = self._bucket(t)
        tmp = np.zeros(self._size, dtype=np.float64)
        for i in range(ti, self._size):
            tmp[i] = self._volatilities[i - ti]
        return tmp

    def volatility_scalar(self, i: int, t: float, x: Array | None = None) -> float:
        """# C++ parity: lmfixedvolmodel.cpp:57-67.

        Like C++, no check that ``i >= ti``: with ``Size`` being unsigned, the
        C++ expression ``volatilities_[i - ti]`` for ``i < ti`` reads out of
        bounds. Python would instead index from the end, so the guard below
        turns that into an explicit failure rather than a silently wrong
        number.
        """
        ti = self._bucket(t)
        qassert.require(
            i >= ti,
            f"forward {i} has already fixed at time {t} (bucket {ti})",
        )
        return float(self._volatilities[i - ti])

    # --- protected --------------------------------------------------------

    def _generate_arguments(self) -> None:
        """# C++ parity: lmfixedvolmodel.cpp:69 — empty body (no arguments)."""


__all__ = ["LmFixedVolatilityModel"]
