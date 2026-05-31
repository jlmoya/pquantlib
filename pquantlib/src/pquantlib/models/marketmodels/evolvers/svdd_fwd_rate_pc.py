"""SVDDFwdRatePc — SVD displaced-diffusion PC evolver with a vol process.

# C++ parity: ql/models/marketmodels/evolvers/svddfwdratepc.{hpp,cpp}
# (v1.42.1). The C++ class is named ``SVDDFwdRatePc`` (the file is
# ``svddfwdratepc``); pquantlib keeps that name.

Displaced-diffusion LMM with an external, *uncorrelated* stochastic volatility
process ("Shifted BGM" with a stochastic vol process, after Brace, *Engineering
BGM*). Each step the Brownian generator produces
``numberOfFactors + variatesPerStep`` Gaussians; the vol-process variates are
interleaved among the rate variates by a fixed ``is_vol_variate`` schedule. The
vol process returns a step standard-deviation ``s``; the step's drift is scaled
by ``s^2`` and its diffusion by ``s`` (predictor-corrector as in the plain PC
evolver). Only ``initial_step == 0`` is currently supported (matching C++).
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
    from pquantlib.models.marketmodels.evolvers.market_model_vol_process import (
        MarketModelVolProcess,
    )
    from pquantlib.models.marketmodels.market_model import MarketModel


class SVDDFwdRatePc(MarketModelEvolver):
    """SVD displaced-diffusion PC evolver with an external vol process.

    # C++ parity: svddfwdratepc.hpp/.cpp SVDDFwdRatePc.
    """

    def __init__(
        self,
        market_model: MarketModel,
        factory: BrownianGeneratorFactory,
        vol_process: MarketModelVolProcess,
        first_volatility_factor: int,
        volatility_factor_step: int,
        numeraires: list[int],
        initial_step: int = 0,
    ) -> None:
        self._market_model = market_model
        self._vol_process = vol_process
        self._first_volatility_factor = first_volatility_factor
        self._vol_factors_per_step = vol_process.variates_per_step()
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
        variates_per_step = self._number_of_factors + self._vol_factors_per_step
        self._all_brownians = [0.0] * variates_per_step
        self._brownians = [0.0] * self._number_of_factors
        self._vol_brownians = [0.0] * self._vol_factors_per_step
        self._is_vol_variate = [False] * variates_per_step
        self._alive = market_model.evolution().first_alive_rate()

        qassert.require(
            initial_step == 0, "initial step zero only supported currently. "
        )
        check_compatibility(market_model.evolution(), numeraires)

        steps = market_model.evolution().number_of_steps()
        self._generator = factory.create(
            self._number_of_factors + self._vol_factors_per_step, steps - initial_step
        )
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

        # build the volatility-variate schedule.
        self._first_volatility_factor = min(
            self._first_volatility_factor, variates_per_step - self._vol_factors_per_step
        )
        vol_increment = (
            variates_per_step - self._first_volatility_factor
        ) // self._vol_factors_per_step
        for i in range(self._vol_factors_per_step):
            self._is_vol_variate[self._first_volatility_factor + i * vol_increment] = True

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
        self._vol_process.next_path()
        return self._generator.next_path()

    def advance_step(self) -> float:
        # we're going from T1 to T2.

        # a) compute drifts D1 at T1.
        if self._current_step > self._initial_step:
            self._calculators[self._current_step].compute(self._forwards, self._drifts1)
        else:
            self._drifts1[:] = self._initial_drifts

        # b) draw all brownians and split between vol process + forwards.
        weight = self._generator.next_step(self._all_brownians)
        j = 0
        k = 0
        for i in range(len(self._all_brownians)):
            if self._is_vol_variate[i]:
                self._vol_brownians[j] = self._all_brownians[i]
                j += 1
            else:
                self._brownians[k] = self._all_brownians[i]
                k += 1

        # vol process step -> standard-deviation multiplier for this step.
        weight2 = self._vol_process.next_step(self._vol_brownians)
        sd_multiplier = self._vol_process.step_sd()
        variance_multiplier = sd_multiplier * sd_multiplier

        a = self._market_model.pseudo_root(self._current_step)
        fixed_drift = self._fixed_drifts[self._current_step]
        brownians = np.asarray(self._brownians, dtype=np.float64)

        alive = self._alive[self._current_step]
        n = self._number_of_rates
        for i in range(alive, n):
            self._log_forwards[i] += variance_multiplier * (
                self._drifts1[i] + fixed_drift[i]
            )
            self._log_forwards[i] += sd_multiplier * float(a[i] @ brownians)
            self._forwards[i] = math.exp(self._log_forwards[i]) - self._displacements[i]

        # c) recompute drifts D2 using the predicted forwards.
        self._calculators[self._current_step].compute(self._forwards, self._drifts2)

        # d) correct forwards using both drifts.
        for i in range(alive, n):
            self._log_forwards[i] += (
                variance_multiplier * (self._drifts2[i] - self._drifts1[i]) / 2.0
            )
            self._forwards[i] = math.exp(self._log_forwards[i]) - self._displacements[i]

        # e) update curve state.
        self._curve_state.set_on_forward_rates(self._forwards)

        self._current_step += 1
        return weight * weight2

    def current_step(self) -> int:
        return self._current_step

    def current_state(self) -> CurveState:
        return self._curve_state
