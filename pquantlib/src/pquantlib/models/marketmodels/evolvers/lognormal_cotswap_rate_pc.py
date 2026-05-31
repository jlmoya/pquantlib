"""LogNormalCotSwapRatePc — coterminal-swap-rate predictor-corrector evolver.

# C++ parity: ql/models/marketmodels/evolvers/lognormalcotswapratepc.{hpp,cpp}
# (v1.42.1).

Predictor-corrector evolver for the coterminal-swap market model: the state
variables are the coterminal swap rates (all swaps share the final payment
date), evolved in log coordinates under the ``SMMDriftCalculator`` drift. The
intermediate ``CoterminalSwapCurveState`` is refreshed after the predictor
step so the corrector drift sees the predicted swap rates.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.models.marketmodels.curvestates.coterminal_swap_curve_state import (
    CoterminalSwapCurveState,
)
from pquantlib.models.marketmodels.driftcomputation.smm_drift_calculator import (
    SMMDriftCalculator,
)
from pquantlib.models.marketmodels.evolution_description import check_compatibility
from pquantlib.models.marketmodels.evolver import MarketModelEvolver

if TYPE_CHECKING:
    from pquantlib.models.marketmodels.brownian_generator import (
        BrownianGeneratorFactory,
    )
    from pquantlib.models.marketmodels.curve_state import CurveState
    from pquantlib.models.marketmodels.market_model import MarketModel


class LogNormalCotSwapRatePc(MarketModelEvolver):
    """Coterminal-swap-rate predictor-corrector evolver.

    # C++ parity: lognormalcotswapratepc.hpp/.cpp LogNormalCotSwapRatePc.
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
        self._curve_state = CoterminalSwapCurveState(
            market_model.evolution().rate_times()
        )
        self._swap_rates = list(market_model.initial_rates())
        self._displacements = list(market_model.displacements())
        n = self._number_of_rates
        self._log_swap_rates = [0.0] * n
        self._initial_log_swap_rates = [0.0] * n
        self._drifts1 = [0.0] * n
        self._drifts2 = [0.0] * n
        self._initial_drifts = [0.0] * n
        self._brownians = [0.0] * self._number_of_factors
        self._alive = market_model.evolution().first_alive_rate()

        check_compatibility(market_model.evolution(), numeraires)

        steps = market_model.evolution().number_of_steps()
        self._generator = factory.create(self._number_of_factors, steps - initial_step)
        self._current_step = initial_step

        self._calculators: list[SMMDriftCalculator] = []
        self._fixed_drifts: list[list[float]] = []
        rate_taus = market_model.evolution().rate_taus()
        for j in range(steps):
            a = market_model.pseudo_root(j)
            self._calculators.append(
                SMMDriftCalculator(
                    a, self._displacements, rate_taus, numeraires[j], self._alive[j]
                )
            )
            variances = np.einsum("ij,ij->i", a, a)
            self._fixed_drifts.append([-0.5 * float(v) for v in variances])

        self._set_coterminal_swap_rates(list(market_model.initial_rates()))

    def numeraires(self) -> list[int]:
        return self._numeraires

    def _set_coterminal_swap_rates(self, swap_rates: list[float]) -> None:
        # C++ parity: setCoterminalSwapRates.
        qassert.require(
            len(swap_rates) == self._number_of_rates,
            "mismatch between swapRates and rateTimes",
        )
        for i in range(self._number_of_rates):
            self._initial_log_swap_rates[i] = math.log(
                swap_rates[i] + self._displacements[i]
            )
        self._curve_state.set_on_coterminal_swap_rates(swap_rates)
        self._calculators[self._initial_step].compute(self._curve_state, self._initial_drifts)

    def set_initial_state(self, curve_state: CurveState) -> None:
        # C++ parity: setInitialState casts to a CoterminalSwapCurveState.
        qassert.require(
            isinstance(curve_state, CoterminalSwapCurveState),
            "CoterminalSwapCurveState required",
        )
        assert isinstance(curve_state, CoterminalSwapCurveState)
        self._set_coterminal_swap_rates(curve_state.coterminal_swap_rates())

    def start_new_path(self) -> float:
        self._current_step = self._initial_step
        self._log_swap_rates[:] = self._initial_log_swap_rates
        return self._generator.next_path()

    def advance_step(self) -> float:
        # we're going from T1 to T2.

        # a) compute drifts D1 at T1.
        if self._current_step > self._initial_step:
            self._calculators[self._current_step].compute(self._curve_state, self._drifts1)
        else:
            self._drifts1[:] = self._initial_drifts

        # b) evolve swap rates up to T2 using D1.
        weight = self._generator.next_step(self._brownians)
        a = self._market_model.pseudo_root(self._current_step)
        fixed_drift = self._fixed_drifts[self._current_step]
        brownians = np.asarray(self._brownians, dtype=np.float64)

        alive = self._alive[self._current_step]
        n = self._number_of_rates
        for i in range(alive, n):
            self._log_swap_rates[i] += self._drifts1[i] + fixed_drift[i]
            self._log_swap_rates[i] += float(a[i] @ brownians)
            self._swap_rates[i] = math.exp(self._log_swap_rates[i]) - self._displacements[i]

        # intermediate curve state update (predictor).
        self._curve_state.set_on_coterminal_swap_rates(self._swap_rates)

        # c) recompute drifts D2 using the predicted swap rates.
        self._calculators[self._current_step].compute(self._curve_state, self._drifts2)

        # d) correct swap rates using both drifts.
        for i in range(alive, n):
            self._log_swap_rates[i] += (self._drifts2[i] - self._drifts1[i]) / 2.0
            self._swap_rates[i] = math.exp(self._log_swap_rates[i]) - self._displacements[i]

        # e) update curve state.
        self._curve_state.set_on_coterminal_swap_rates(self._swap_rates)

        self._current_step += 1
        return weight

    def current_step(self) -> int:
        return self._current_step

    def current_state(self) -> CurveState:
        return self._curve_state
