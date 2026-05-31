"""LogNormalFwdRateEuler — Euler lognormal forward-rate evolver.

# C++ parity: ql/models/marketmodels/evolvers/lognormalfwdrateeuler.{hpp,cpp}
# (v1.42.1).

Same as the predictor-corrector evolver with the two corrector steps dropped:
a single Euler step using the drift ``D1`` at ``T1`` plus the fixed Ito drift +
the pseudo-root diffusion.
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


class LogNormalFwdRateEuler(MarketModelEvolver):
    """Euler lognormal forward-rate evolver.

    # C++ parity: lognormalfwdrateeuler.hpp/.cpp LogNormalFwdRateEuler.
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
        self._brownians = [0.0] * self._number_of_factors
        self._alive = market_model.evolution().first_alive_rate()

        check_compatibility(market_model.evolution(), numeraires)

        steps = market_model.evolution().number_of_steps()
        self._generator = factory.create(self._number_of_factors, steps - initial_step)
        self._current_step = initial_step

        self._calculators: list[LMMDriftCalculator] = []
        self._fixed_drifts: list[list[float]] = []
        rate_taus = market_model.evolution().rate_taus()
        for j in range(steps):
            a = market_model.pseudo_root(j)
            self._calculators.append(
                LMMDriftCalculator(
                    a, self._displacements, rate_taus, numeraires[j], self._alive[j]
                )
            )
            variances = np.einsum("ij,ij->i", a, a)
            self._fixed_drifts.append([-0.5 * float(v) for v in variances])

        self._set_forwards(list(market_model.initial_rates()))

    def numeraires(self) -> list[int]:
        return self._numeraires

    def browniansThisStep(self) -> list[float]:  # noqa: N802
        """The Gaussian increments used in the most recent step (pathwise vegas).

        # C++ parity: LogNormalFwdRateEuler::browniansThisStep.
        """
        return self._brownians

    def _set_forwards(self, forwards: list[float]) -> None:
        # C++ parity: LogNormalFwdRateEuler::setForwards.
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

        # b) evolve forwards up to T2 using D1 (single Euler step).
        weight = self._generator.next_step(self._brownians)
        a = self._market_model.pseudo_root(self._current_step)
        fixed_drift = self._fixed_drifts[self._current_step]
        brownians = np.asarray(self._brownians, dtype=np.float64)

        alive = self._alive[self._current_step]
        for i in range(alive, self._number_of_rates):
            self._log_forwards[i] += self._drifts1[i] + fixed_drift[i]
            self._log_forwards[i] += float(a[i] @ brownians)
            self._forwards[i] = math.exp(self._log_forwards[i]) - self._displacements[i]

        # c) update curve state.
        self._curve_state.set_on_forward_rates(self._forwards)

        self._current_step += 1
        return weight

    def current_step(self) -> int:
        return self._current_step

    def current_state(self) -> CurveState:
        return self._curve_state
