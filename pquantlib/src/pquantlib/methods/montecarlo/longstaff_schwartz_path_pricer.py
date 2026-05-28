"""LongstaffSchwartzPathPricer — regression-based early-exercise pricer.

# C++ parity: ql/methods/montecarlo/longstaffschwartzpathpricer.hpp (v1.42.1).

Implements the Longstaff-Schwartz (2001) least-squares Monte Carlo
algorithm for American / Bermudan options:

1. **Calibration phase** — every call to ``__call__`` records the
   incoming path into ``_paths`` and returns 0.0 as a placeholder.
   The engine drives this by calling ``addSamples(n_calibration)``
   on a ``MonteCarloModel`` wired to *this* pricer.

2. **Calibration step** (``calibrate()``) — backward induction from
   ``t = len-2`` down to ``t = 1``. At each step:

   * Compute the immediate exercise value for every stored path:
     ``exercise[j] = pricer(path[j], t)``.
   * Filter ITM paths (``exercise[j] > 0``) into a regression matrix
     ``X[j, l] = basis_l(state(path[j], t))`` paired with
     ``y[j] = dF[t] * carry_price[j]`` (the discounted continuation
     value brought back from the previous timestep).
   * Solve ``coef[t-1] = lstsq(X, y)`` (numpy.linalg.lstsq).
   * Update ``carry_price[j]`` per path: if estimated continuation
     value is less than exercise, take the exercise instead.

3. **Pricing phase** — for each fresh path, walk backwards from
   maturity; at every intermediate exercise date evaluate the
   regression to decide if continuation > exercise; track the final
   discounted payoff. ``exerciseProbability`` is accumulated via
   :class:`IncrementalStatistics`.

The Python port uses ``numpy.linalg.lstsq`` (the C++ equivalent is
``GeneralLinearLeastSquares`` which is also a QR-based solver — same
math, different package).

Note on path mutation: ``PathGenerator`` reuses the same ``Path``
instance per call (mutates in place). During calibration we must
deep-copy the ``values`` ndarray to avoid losing history.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.math.statistics.incremental_statistics import IncrementalStatistics
from pquantlib.methods.montecarlo.early_exercise_path_pricer import (
    EarlyExercisePathPricer,
)
from pquantlib.methods.montecarlo.path import Path
from pquantlib.methods.montecarlo.path_pricer import PathPricer

if TYPE_CHECKING:
    from pquantlib.termstructures.yield_term_structure import YieldTermStructure
    from pquantlib.time.time_grid import TimeGrid


class LongstaffSchwartzPathPricer[PathT, StateT](PathPricer[PathT]):
    """Longstaff-Schwartz regression-based early-exercise pricer.

    # C++ parity: ``LongstaffSchwartzPathPricer<PathType>``
    # (longstaffschwartzpathpricer.hpp:52-84).

    Generic over both the path type and the regression state type:

    * ``Path`` -> ``float`` state (1-D underlying value).
    * ``MultiPath`` -> ``ndarray[float64]`` state (multi-asset
      basket).
    """

    __slots__ = (
        "_basis",
        "_calibration_phase",
        "_coeff",
        "_df",
        "_exercise_probability",
        "_len",
        "_path_pricer",
        "_paths",
    )

    def __init__(
        self,
        times: TimeGrid,
        path_pricer: EarlyExercisePathPricer[PathT, StateT],
        term_structure: YieldTermStructure,
    ) -> None:
        """Build a pricer over ``times`` driven by ``path_pricer``.

        # C++ parity: ``LongstaffSchwartzPathPricer`` constructor
        # (longstaffschwartzpathpricer.hpp:86-99).

        Pre-computes the per-period discount factors ``dF[i] =
        TS.discount(times[i+1]) / TS.discount(times[i])`` for
        ``i in 0..len-2``.
        """
        qassert.require(len(times) >= 2, "time grid must have at least 2 points")
        self._calibration_phase: bool = True
        self._path_pricer: EarlyExercisePathPricer[PathT, StateT] = path_pricer
        # C++ allocates Array[len-2] for coefficients (interior exercise dates only),
        # but we leave it as a list for Python ergonomics. Index [i-1] for date i.
        self._coeff: list[npt.NDArray[np.float64]] = [
            np.zeros(0, dtype=np.float64) for _ in range(len(times) - 2)
        ]
        self._df: npt.NDArray[np.float64] = np.empty(len(times) - 1, dtype=np.float64)
        for i in range(len(times) - 1):
            d_next = term_structure.discount(times[i + 1])
            d_cur = term_structure.discount(times[i])
            self._df[i] = d_next / d_cur
        self._basis = path_pricer.basis_system()
        self._len: int = len(times)
        self._paths: list[PathT] = []
        self._exercise_probability: IncrementalStatistics = IncrementalStatistics()

    # --- PathPricer contract -------------------------------------------------

    def __call__(self, path: PathT) -> float:
        """Calibrate (record path) during cal-phase, otherwise price.

        # C++ parity: ``operator()(const PathType&) const`` (longstaffschwartzpathpricer.hpp:101-140).
        """
        if self._calibration_phase:
            # Calibration: store a deep copy of the path so subsequent
            # PathGenerator mutations don't corrupt history.
            # C++ ``paths_.push_back(path)`` — by-value copy.
            self._paths.append(self._clone_path(path))
            return 0.0

        # Pricing: forward-look exercise rule per the trained regression.
        price = self._path_pricer(path, self._len - 1)
        exercised = price > 0.0

        for i in range(self._len - 2, 0, -1):
            price *= float(self._df[i])
            exercise_val = self._path_pricer(path, i)
            if exercise_val > 0.0:
                reg_value = self._path_pricer.state(path, i)
                continuation = self._evaluate_basis(self._coeff[i - 1], reg_value)
                if continuation < exercise_val:
                    price = exercise_val
                    exercised = True

        self._exercise_probability.add(1.0 if exercised else 0.0)
        return price * float(self._df[0])

    # --- calibration ---------------------------------------------------------

    def calibrate(self) -> None:
        """Run the backward-induction regression to fit basis coefficients.

        # C++ parity: ``calibrate()`` (longstaffschwartzpathpricer.hpp:143-206).
        """
        n = len(self._paths)
        qassert.require(n > 0, "no paths stored for calibration")

        # Per-path running prices: start with the terminal exercise value.
        prices = np.array(
            [self._path_pricer(self._paths[j], self._len - 1) for j in range(n)],
            dtype=np.float64,
        )

        for i in range(self._len - 2, 0, -1):
            x_states: list[StateT] = []
            y_values: list[float] = []
            exercise = np.empty(n, dtype=np.float64)

            # Roll back step: filter ITM paths.
            for j in range(n):
                exercise[j] = self._path_pricer(self._paths[j], i)
                if exercise[j] > 0.0:
                    x_states.append(self._path_pricer.state(self._paths[j], i))
                    y_values.append(float(self._df[i]) * float(prices[j]))

            if len(self._basis) <= len(x_states):
                # C++ parity: GeneralLinearLeastSquares solve over basis
                #            ((longstaffschwartzpathpricer.hpp:173)).
                # Build the regression matrix [n_itm, n_basis].
                n_itm = len(x_states)
                n_basis = len(self._basis)
                x_mat = np.empty((n_itm, n_basis), dtype=np.float64)
                for jj in range(n_itm):
                    for ll in range(n_basis):
                        x_mat[jj, ll] = self._basis[ll](x_states[jj])
                y_vec = np.array(y_values, dtype=np.float64)
                coef, *_ = np.linalg.lstsq(x_mat, y_vec, rcond=None)
                self._coeff[i - 1] = coef.astype(np.float64, copy=False)
            else:
                # C++ parity: ``coeff_[i-1] = Array(v_.size(), 0.0)``
                # — fall back to "always exercise if ITM" by zeroing
                # the continuation regression.
                self._coeff[i - 1] = np.zeros(len(self._basis), dtype=np.float64)

            # Per-path continuation update: discount carried price, then
            # check ITM regression vs exercise.
            itm_cursor = 0
            for j in range(n):
                prices[j] *= float(self._df[i])
                if exercise[j] > 0.0:
                    continuation = self._evaluate_basis(
                        self._coeff[i - 1], x_states[itm_cursor]
                    )
                    if continuation < exercise[j]:
                        prices[j] = exercise[j]
                    itm_cursor += 1

        # Release calibration paths: C++ does ``std::vector<PathType>
        # empty; paths_.swap(empty)``.
        self._paths = []
        self._calibration_phase = False

    # --- inspectors ----------------------------------------------------------

    def exercise_probability(self) -> float:
        """Mean ``exercised`` indicator across priced paths.

        # C++ parity: ``exerciseProbability()`` (longstaffschwartzpathpricer.hpp:209-211).
        """
        return self._exercise_probability.mean()

    def is_calibration_phase(self) -> bool:
        """True iff calibrate() has not been called yet."""
        return self._calibration_phase

    def coefficients(self) -> list[npt.NDArray[np.float64]]:
        """Regression coefficients ``coeff_[i-1]`` for each interior date.

        Exposed for testing — C++ keeps these private but our tests
        cross-validate the regression stability under seed.
        """
        return self._coeff

    # --- internals ----------------------------------------------------------

    def _evaluate_basis(
        self, coef: npt.NDArray[np.float64], state: StateT
    ) -> float:
        """Compute ``sum_l coef[l] * basis_l(state)`` — continuation value."""
        out = 0.0
        for ll in range(len(self._basis)):
            out += float(coef[ll]) * self._basis[ll](state)
        return out

    @staticmethod
    def _clone_path(path: PathT) -> PathT:
        """Deep-copy the underlying ndarray so mutating-in-place doesn't lose history.

        For ``Path``: rebuild with ``np.copy(path.values)``. For
        ``MultiPath``: per-asset clone. We dispatch on type.
        """
        if isinstance(path, Path):
            # mypy: ndarray.copy() returns ndarray.
            cloned = Path(path.time_grid, np.copy(path.values))
            # Bypass typing — PathT is constrained to be Path-or-MultiPath
            # but the dispatch is dynamic.
            return cloned  # type: ignore[return-value]
        # MultiPath cloning: assumes a ``paths`` list of Path objects.
        # We only ship the Path variant in L6-A — MultiPath variant deferred.
        raise NotImplementedError(
            "MultiPath cloning not supported in L6-A — "
            "LongstaffSchwartzMultiPathPricer deferred (Phase 6+ carve-out)"
        )


__all__ = ["LongstaffSchwartzPathPricer"]
