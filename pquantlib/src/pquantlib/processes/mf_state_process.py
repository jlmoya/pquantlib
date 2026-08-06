"""MfStateProcess — state process for the Markov-functional model.

# C++ parity: ql/processes/mfstateprocess.{hpp,cpp} (v1.43) —
# ``class MfStateProcess : public StochasticProcess1D``
# (mfstateprocess.hpp:38-61, mfstateprocess.cpp:24-114).

Describes the driftless process

    dx = sigma(t) e^{a t} dW(t)

parameterised by a mean reversion ``a`` and a piecewise-constant volatility
``sigma(t)`` given as a ``times`` grid plus a ``vols`` vector that is
**one longer** than ``times`` (the last entry covers everything beyond the
last time node).

Bucket lookup is ``upper_bound``, i.e. STRICTLY greater: at ``t`` exactly
equal to ``times[k]`` the bucket ABOVE the node is selected. That boundary
convention is pinned by the cross-validation at every node.

``variance`` has three arms, all reproduced verbatim:

* ``dt < QL_EPSILON`` -> exactly 0.0;
* ``times`` empty -> a single closed-form term;
* otherwise -> a sum over the buckets spanned by ``[t, t+dt]`` plus a
  trailing partial bucket, each term switching on ``reversionZero_``.

Divergences from C++:

* C++ declares ``setTimes`` / ``setVols`` private with ``friend class
  MarkovFunctional``. Python has no access control of that kind, so the two
  mutators are exposed with a leading underscore and a note that only
  ``MarkovFunctional`` is supposed to call them.
* ``checkTimesVols`` compares ``times_.size() == vols_.size() - 1`` in
  UNSIGNED arithmetic, so an empty ``vols`` wraps to ``SIZE_MAX`` and the
  requirement fails. Python's ints do not wrap, but ``len(times) == -1`` is
  false for the same input, so the observable behaviour (a raise) matches.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.math.constants import QL_EPSILON
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D


class MfStateProcess(StochasticProcess1D):
    """Markov-functional state process.

    # C++ parity: ``class MfStateProcess`` (mfstateprocess.hpp:38-61).
    """

    __slots__ = ("_reversion", "_reversion_zero", "_times", "_vols")

    def __init__(
        self,
        reversion: float,
        times: Sequence[float] | npt.NDArray[np.float64],
        vols: Sequence[float] | npt.NDArray[np.float64],
    ) -> None:
        """Construct from the reversion speed and the (times, vols) grid.

        # C++ parity: mfstateprocess.cpp:24-29.
        """
        # No discretization: every method the base would route through one is
        # overridden below. C++ likewise leaves ``discretization_`` null.
        super().__init__()
        self._reversion: float = float(reversion)
        # C++ parity: mfstateprocess.cpp:26-27 —
        # ``reversion_ < QL_EPSILON && -reversion_ < QL_EPSILON``.
        self._reversion_zero: bool = (
            self._reversion < QL_EPSILON and -self._reversion < QL_EPSILON
        )
        self._times: npt.NDArray[np.float64] = np.asarray(times, dtype=np.float64)
        self._vols: npt.NDArray[np.float64] = np.asarray(vols, dtype=np.float64)
        self._check_times_vols()

    # --- MarkovFunctional-only mutators ----------------------------------

    def _set_times(self, times: Sequence[float] | npt.NDArray[np.float64]) -> None:
        """Replace the time grid and notify.

        # C++ parity: mfstateprocess.cpp:31-35. Private in C++ with
        # ``friend class MarkovFunctional``; only that model should call it.
        """
        self._times = np.asarray(times, dtype=np.float64)
        self._check_times_vols()
        self.notify_observers()

    def _set_vols(self, vols: Sequence[float] | npt.NDArray[np.float64]) -> None:
        """Replace the volatility vector and notify.

        # C++ parity: mfstateprocess.cpp:37-41. Private in C++ with
        # ``friend class MarkovFunctional``; only that model should call it.
        """
        self._vols = np.asarray(vols, dtype=np.float64)
        self._check_times_vols()
        self.notify_observers()

    def _check_times_vols(self) -> None:
        """Validate the (times, vols) invariants.

        # C++ parity: mfstateprocess.cpp:43-56.
        """
        n_times = int(self._times.size)
        n_vols = int(self._vols.size)
        qassert.require(
            n_times == n_vols - 1,
            f"number of volatilities ({n_vols}) compared to number of times "
            f"({n_times} must be bigger by one",
        )
        # ``:g`` matches C++'s default ``operator<<(ostream&, Real)`` formatting
        # (6 significant digits, no trailing ".0"), so the messages come out
        # byte-identical to the C++ what() recorded in the reference JSON.
        for i in range(n_times - 1):
            qassert.require(
                self._times[i] < self._times[i + 1],
                f"times must be increasing ({self._times[i]:g}@{i} , "
                f"{self._times[i + 1]:g}@{i + 1})",
            )
        for i in range(n_vols):
            qassert.require(
                self._vols[i] >= 0.0,
                f"volatilities must be non negative ({self._vols[i]:g}@{i})",
            )

    # --- bucket lookup ----------------------------------------------------

    def _bucket(self, t: float) -> int:
        """Index of the volatility bucket containing ``t``.

        # C++ parity: ``std::upper_bound(times_.begin(), times_.end(), t)
        # - times_.begin()`` — STRICTLY greater, so ``t == times_[k]``
        # selects bucket ``k+1``.
        """
        return int(np.searchsorted(self._times, t, side="right"))

    # --- inspectors --------------------------------------------------------

    def reversion(self) -> float:
        """Mean reversion speed.

        # C++ parity: ``reversion_`` is private with no accessor in C++; the
        # Python port exposes a read-only one for testability.
        """
        return self._reversion

    def times(self) -> npt.NDArray[np.float64]:
        """The time grid (a copy — the C++ member is private)."""
        return self._times.copy()

    def vols(self) -> npt.NDArray[np.float64]:
        """The volatility vector (a copy — the C++ member is private)."""
        return self._vols.copy()

    # --- StochasticProcess1D interface ------------------------------------

    def x0(self) -> float:
        # C++ parity: mfstateprocess.cpp:58.
        return 0.0

    def drift_1d(self, t: float, x: float) -> float:
        # C++ parity: mfstateprocess.cpp:60 — identically zero.
        del t, x
        return 0.0

    def diffusion_1d(self, t: float, x: float) -> float:
        # C++ parity: mfstateprocess.cpp:62-66.
        del x
        return float(self._vols[self._bucket(t)])

    def expectation_1d(self, t0: float, x0: float, dt: float) -> float:
        # C++ parity: mfstateprocess.cpp:68-70 — driftless, so the
        # expectation is x0 regardless of t0 and dt.
        del t0, dt
        return x0

    def std_deviation_1d(self, t0: float, x0: float, dt: float) -> float:
        # C++ parity: mfstateprocess.cpp:72-74.
        return math.sqrt(self.variance_1d(t0, x0, dt))

    def variance_1d(self, t0: float, x0: float, dt: float) -> float:
        """Integrated variance over ``[t0, t0+dt]``.

        # C++ parity: mfstateprocess.cpp:76-114.
        """
        del x0
        t = t0
        if dt < QL_EPSILON:
            return 0.0

        rev = self._reversion
        if self._times.size == 0:
            if self._reversion_zero:
                return dt
            return (
                1.0
                / (2.0 * rev)
                * (math.exp(2.0 * rev * (t + dt)) - math.exp(2.0 * rev * t))
            )

        i = self._bucket(t)
        j = self._bucket(t + dt)

        v = 0.0
        for k in range(i, j):
            lower = max(float(self._times[k - 1]) if k > 0 else 0.0, t)
            vol_k = float(self._vols[k])
            if self._reversion_zero:
                v += vol_k * vol_k * (float(self._times[k]) - lower)
            else:
                v += (
                    1.0
                    / (2.0 * rev)
                    * vol_k
                    * vol_k
                    * (math.exp(2.0 * rev * float(self._times[k])) - math.exp(2.0 * rev * lower))
                )

        lower_j = max(float(self._times[j - 1]) if j > 0 else 0.0, t)
        vol_j = float(self._vols[j])
        if self._reversion_zero:
            v += vol_j * vol_j * (t + dt - lower_j)
        else:
            v += (
                1.0
                / (2.0 * rev)
                * vol_j
                * vol_j
                * (math.exp(2.0 * rev * (t + dt)) - math.exp(2.0 * rev * lower_j))
            )

        return v


__all__ = ["MfStateProcess"]
