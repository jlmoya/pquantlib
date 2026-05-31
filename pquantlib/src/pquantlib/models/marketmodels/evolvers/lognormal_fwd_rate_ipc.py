"""LogNormalFwdRateIpc — iterative predictor-corrector forward-rate evolver.

# C++ parity: ql/models/marketmodels/evolvers/lognormalfwdrateipc.{hpp,cpp}
# (v1.42.1).

Requires the terminal measure. Evolves the log-forwards backward from the last
rate down to the first alive rate, building the predictor drift ``drifts2``
incrementally from the running quantity ``g_[j] = tau_j (f_j + d_j) /
(1 + tau_j f_j)`` and the per-step covariance matrix ``C``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.models.marketmodels.curvestates.lmm_curve_state import LMMCurveState
from pquantlib.models.marketmodels.driftcomputation.lmm_drift_calculator import (
    LMMDriftCalculator,
)
from pquantlib.models.marketmodels.evolution_description import (
    check_compatibility,
    is_in_terminal_measure,
)
from pquantlib.models.marketmodels.evolver import MarketModelEvolver

if TYPE_CHECKING:
    from pquantlib.models.marketmodels.brownian_generator import (
        BrownianGeneratorFactory,
    )
    from pquantlib.models.marketmodels.curve_state import CurveState
    from pquantlib.models.marketmodels.market_model import MarketModel


class LogNormalFwdRateIpc(MarketModelEvolver):
    """Iterative predictor-corrector forward-rate evolver (terminal measure).

    # C++ parity: lognormalfwdrateipc.hpp/.cpp LogNormalFwdRateIpc.
    """

    def __init__(
        self,
        market_model: MarketModel,
        factory: BrownianGeneratorFactory,
        numeraires: list[int],
        initial_step: int = 0,
    ) -> None:
        self._market_model = market_model
        self._numeraires = list(numeraires)
        self._initial_step = initial_step
        self._number_of_rates = market_model.number_of_rates()
        self._number_of_factors = market_model.number_of_factors()
        self._curve_state = LMMCurveState(market_model.evolution().rate_times())
        self._forwards = list(market_model.initial_rates())
        self._displacements = list(market_model.displacements())
        n = self._number_of_rates
        self._log_forwards = [0.0] * n
        self._initial_log_forwards = [0.0] * n
        self._drifts1 = [0.0] * n
        self._initial_drifts = [0.0] * n
        self._g = [0.0] * n
        self._brownians = [0.0] * self._number_of_factors
        self._rate_taus = market_model.evolution().rate_taus()
        self._alive = market_model.evolution().first_alive_rate()

        check_compatibility(market_model.evolution(), numeraires)
        qassert.require(
            is_in_terminal_measure(market_model.evolution(), numeraires),
            "terminal measure required for ipc ",
        )

        steps = market_model.evolution().number_of_steps()
        self._generator = factory.create(self._number_of_factors, steps - initial_step)
        self._current_step = initial_step

        self._calculators: list[LMMDriftCalculator] = []
        self._fixed_drifts: list[list[float]] = []
        for j in range(steps):
            a = market_model.pseudo_root(j)
            self._calculators.append(
                LMMDriftCalculator(
                    a, self._displacements, self._rate_taus, numeraires[j], self._alive[j]
                )
            )
            c = market_model.covariance(j)
            self._fixed_drifts.append([-0.5 * float(c[k, k]) for k in range(n)])

        self._set_forwards(list(market_model.initial_rates()))

    def numeraires(self) -> list[int]:
        return self._numeraires

    def _set_forwards(self, forwards: list[float]) -> None:
        # C++ parity: LogNormalFwdRateIpc::setForwards.
        qassert.require(
            len(forwards) == self._number_of_rates,
            "mismatch between forwards and rateTimes",
        )
        for i in range(self._number_of_rates):
            self._initial_log_forwards[i] = math.log(forwards[i] + self._displacements[i])
        self._calculators[self._initial_step].compute(forwards, self._initial_drifts)

    def set_initial_state(self, curve_state: CurveState) -> None:
        self._set_forwards(curve_state.forward_rates())

    def start_new_path(self) -> float:
        self._current_step = self._initial_step
        self._log_forwards[:] = self._initial_log_forwards
        return self._generator.next_path()

    def advance_step(self) -> float:
        # we're going from T1 to T2.

        # a) compute drifts D1 at T1 (plain — no factor reduction).
        if self._current_step > self._initial_step:
            self._calculators[self._current_step].compute_plain(self._forwards, self._drifts1)
        else:
            self._drifts1[:] = self._initial_drifts

        weight = self._generator.next_step(self._brownians)
        a = self._market_model.pseudo_root(self._current_step)
        c = self._market_model.covariance(self._current_step)
        fixed_drift = self._fixed_drifts[self._current_step]
        brownians = np.asarray(self._brownians, dtype=np.float64)

        alive = self._alive[self._current_step]
        n = self._number_of_rates
        for i in range(n - 1, alive - 1, -1):
            drifts2 = 0.0
            for j in range(i + 1, n):
                drifts2 -= self._g[j] * float(c[i, j])
            self._log_forwards[i] += 0.5 * (self._drifts1[i] + drifts2) + fixed_drift[i]
            self._log_forwards[i] += float(a[i] @ brownians)
            self._forwards[i] = math.exp(self._log_forwards[i]) - self._displacements[i]
            self._g[i] = (
                self._rate_taus[i]
                * (self._forwards[i] + self._displacements[i])
                / (1.0 + self._rate_taus[i] * self._forwards[i])
            )

        self._curve_state.set_on_forward_rates(self._forwards)

        self._current_step += 1
        return weight

    def current_step(self) -> int:
        return self._current_step

    def current_state(self) -> CurveState:
        return self._curve_state
