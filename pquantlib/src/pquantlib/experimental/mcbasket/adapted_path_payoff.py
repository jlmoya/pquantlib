"""AdaptedPathPayoff — convenience adapter over PathPayoff.

# C++ parity: ql/experimental/mcbasket/adaptedpathpayoff.{hpp,cpp}
#             (v1.42.1).

``AdaptedPathPayoff`` lets a concrete payoff author work against a
small, *adapted* accessor object (:class:`ValuationData`) rather than
the raw ``(path, fts, payments, exercises, states)`` tuple. The
accessor tracks the maximum fixing time read so far and forbids writing
a payment/exercise that depends on a later fixing — enforcing that the
payoff is an adapted (non-anticipating) function.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.experimental.mcbasket.path_payoff import PathPayoff
from pquantlib.termstructures.yield_term_structure import YieldTermStructure


class ValuationData:
    """Adapted accessor over a single path's valuation arrays.

    # C++ parity: ``AdaptedPathPayoff::ValuationData``.
    """

    __slots__ = (
        "_exercises",
        "_forward_term_structures",
        "_maximum_time_read",
        "_path",
        "_payments",
        "_states",
    )

    def __init__(
        self,
        path: npt.NDArray[np.float64],
        forward_term_structures: Sequence[YieldTermStructure],
        payments: npt.NDArray[np.float64],
        exercises: npt.NDArray[np.float64],
        states: list[npt.NDArray[np.float64]],
    ) -> None:
        self._path = path
        self._forward_term_structures = forward_term_structures
        self._payments = payments
        self._exercises = exercises
        self._states = states
        self._maximum_time_read = 0

    def number_of_times(self) -> int:
        # C++ parity: ``path_.columns()``.
        return self._path.shape[1]

    def number_of_assets(self) -> int:
        # C++ parity: ``path_.rows()``.
        return self._path.shape[0]

    def get_asset_value(self, time: int, asset: int) -> float:
        self._maximum_time_read = max(self._maximum_time_read, time)
        return float(self._path[asset, time])

    def get_yield_term_structure(self, time: int) -> YieldTermStructure:
        self._maximum_time_read = max(self._maximum_time_read, time)
        return self._forward_term_structures[time]

    def set_payoff_value(self, time: int, value: float) -> None:
        # Adapted: a payment cannot depend on a later fixing.
        qassert.require(
            time >= self._maximum_time_read, "not adapted payoff: looking into the future"
        )
        self._payments[time] = value

    def set_exercise_data(
        self, time: int, exercise: float, state: npt.NDArray[np.float64]
    ) -> None:
        # Adapted: exercise data cannot depend on a later fixing.
        qassert.require(
            time >= self._maximum_time_read, "not adapted payoff: looking into the future"
        )
        if self._exercises.size != 0:
            self._exercises[time] = exercise
        if len(self._states) != 0:
            self._states[time] = state


class AdaptedPathPayoff(PathPayoff):
    """PathPayoff whose body is written against :class:`ValuationData`.

    # C++ parity: ``AdaptedPathPayoff``.

    Subclasses implement :meth:`_evaluate` (the C++ ``operator()``),
    calling ``data.set_payoff_value`` / ``data.set_exercise_data``.
    """

    def value(
        self,
        path: npt.NDArray[np.float64],
        forward_term_structures: Sequence[YieldTermStructure],
        payments: npt.NDArray[np.float64],
        exercises: npt.NDArray[np.float64],
        states: list[npt.NDArray[np.float64]],
    ) -> None:
        data = ValuationData(path, forward_term_structures, payments, exercises, states)
        self._evaluate(data)

    @abstractmethod
    def _evaluate(self, data: ValuationData) -> None:
        """Compute payments + exercise data from ``data``.

        # C++ parity: ``AdaptedPathPayoff::operator()(ValuationData&)``.
        """


__all__ = ["AdaptedPathPayoff", "ValuationData"]
