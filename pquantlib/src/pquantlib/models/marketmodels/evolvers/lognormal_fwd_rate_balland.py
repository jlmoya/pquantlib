"""LogNormalFwdRateBalland — Balland-drift forward-rate evolver.

# C++ parity: ql/models/marketmodels/evolvers/lognormalfwdrateballand.{hpp,cpp}
# (v1.42.1).

A predictor-corrector variant using Balland's drift approximation: after the
predictor step the forwards are replaced by the geometric mean
``sqrt(f_i * f_i^0)`` of the predicted and initial forward before the corrector
drift ``D2`` is computed, and the correction applies the full ``D2 - D1`` (not
the half-step ``(D2 - D1) / 2`` of the plain PC evolver).
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
from pquantlib.models.marketmodels.evolution_description import check_compatibility
from pquantlib.models.marketmodels.evolver import MarketModelEvolver

if TYPE_CHECKING:
    from pquantlib.models.marketmodels.brownian_generator import (
        BrownianGeneratorFactory,
    )
    from pquantlib.models.marketmodels.curve_state import CurveState
    from pquantlib.models.marketmodels.market_model import MarketModel


class LogNormalFwdRateBalland(MarketModelEvolver):
    """Balland-drift forward-rate evolver.

    # C++ parity: lognormalfwdrateballand.hpp/.cpp LogNormalFwdRateBalland.
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
        self._drifts2 = [0.0] * n
        self._initial_drifts = [0.0] * n
        self._brownians = [0.0] * self._number_of_factors
        self._rate_taus = market_model.evolution().rate_taus()
        self._alive = market_model.evolution().first_alive_rate()

        check_compatibility(market_model.evolution(), numeraires)

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
        # C++ parity: setForwards.
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

        # a) compute drifts D1 at T1.
        if self._current_step > self._initial_step:
            self._calculators[self._current_step].compute(self._forwards, self._drifts1)
        else:
            self._drifts1[:] = self._initial_drifts

        weight = self._generator.next_step(self._brownians)
        a = self._market_model.pseudo_root(self._current_step)
        fixed_drift = self._fixed_drifts[self._current_step]
        brownians = np.asarray(self._brownians, dtype=np.float64)
        initial_rates = self._market_model.initial_rates()

        alive = self._alive[self._current_step]
        n = self._number_of_rates
        for i in range(alive, n):
            self._log_forwards[i] += self._drifts1[i] + fixed_drift[i]
            self._log_forwards[i] += float(a[i] @ brownians)
            self._forwards[i] = math.exp(self._log_forwards[i]) - self._displacements[i]

        # Balland: replace predicted forwards by the geometric mean of the
        # predicted and initial forward before the corrector drift.
        for i in range(alive, n):
            self._forwards[i] = math.sqrt(self._forwards[i] * initial_rates[i])

        self._calculators[self._current_step].compute(self._forwards, self._drifts2)

        for i in range(alive, n):
            self._log_forwards[i] += self._drifts2[i] - self._drifts1[i]
            self._forwards[i] = math.exp(self._log_forwards[i]) - self._displacements[i]

        self._curve_state.set_on_forward_rates(self._forwards)

        self._current_step += 1
        return weight

    def current_step(self) -> int:
        return self._current_step

    def current_state(self) -> CurveState:
        return self._curve_state
