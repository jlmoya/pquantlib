"""NormalFwdRatePc — predictor-corrector normal forward-rate evolver.

# C++ parity: ql/models/marketmodels/evolvers/normalfwdratepc.{hpp,cpp}
# (v1.42.1).

Predictor-corrector evolver for the *normal* (rather than log-normal) forward
rate dynamics: the forwards evolve additively (``f += drift + A . Z``) using the
``LMMNormalDriftCalculator``, with no log transform, no displacement and no
``-0.5 sigma^2`` Ito drift. Negative forwards are admissible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pquantlib.models.marketmodels.curvestates.lmm_curve_state import LMMCurveState
from pquantlib.models.marketmodels.driftcomputation.lmm_normal_drift_calculator import (
    LMMNormalDriftCalculator,
)
from pquantlib.models.marketmodels.evolution_description import check_compatibility
from pquantlib.models.marketmodels.evolver import MarketModelEvolver

if TYPE_CHECKING:
    from pquantlib.models.marketmodels.brownian_generator import (
        BrownianGeneratorFactory,
    )
    from pquantlib.models.marketmodels.curve_state import CurveState
    from pquantlib.models.marketmodels.market_model import MarketModel


class NormalFwdRatePc(MarketModelEvolver):
    """Predictor-corrector normal forward-rate evolver.

    # C++ parity: normalfwdratepc.hpp/.cpp NormalFwdRatePc.
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
        self._initial_forwards = list(market_model.initial_rates())
        n = self._number_of_rates
        self._drifts1 = [0.0] * n
        self._drifts2 = [0.0] * n
        self._initial_drifts = [0.0] * n
        self._brownians = [0.0] * self._number_of_factors
        self._alive = market_model.evolution().first_alive_rate()

        check_compatibility(market_model.evolution(), numeraires)

        steps = market_model.evolution().number_of_steps()
        self._generator = factory.create(self._number_of_factors, steps - initial_step)
        self._current_step = initial_step

        self._calculators: list[LMMNormalDriftCalculator] = []
        rate_taus = market_model.evolution().rate_taus()
        for j in range(steps):
            a = market_model.pseudo_root(j)
            self._calculators.append(
                LMMNormalDriftCalculator(a, rate_taus, numeraires[j], self._alive[j])
            )

        self._set_forwards(list(market_model.initial_rates()))

    def numeraires(self) -> list[int]:
        return self._numeraires

    def _set_forwards(self, forwards: list[float]) -> None:
        # C++ parity: NormalFwdRatePc::setForwards (the dangling empty loop in
        # the C++ source is a no-op; only the initial-drift compute matters).
        self._calculators[self._initial_step].compute(forwards, self._initial_drifts)

    def set_initial_state(self, curve_state: CurveState) -> None:
        self._set_forwards(curve_state.forward_rates())

    def start_new_path(self) -> float:
        self._current_step = self._initial_step
        self._forwards[:] = self._initial_forwards
        return self._generator.next_path()

    def advance_step(self) -> float:
        # we're going from T1 to T2.

        # a) compute drifts D1 at T1.
        if self._current_step > self._initial_step:
            self._calculators[self._current_step].compute(self._forwards, self._drifts1)
        else:
            self._drifts1[:] = self._initial_drifts

        # b) evolve forwards up to T2 using D1.
        weight = self._generator.next_step(self._brownians)
        a = self._market_model.pseudo_root(self._current_step)
        brownians = np.asarray(self._brownians, dtype=np.float64)

        alive = self._alive[self._current_step]
        n = self._number_of_rates
        for i in range(alive, n):
            self._forwards[i] += self._drifts1[i]
            self._forwards[i] += float(a[i] @ brownians)

        # c) recompute drifts D2 using the predicted forwards.
        self._calculators[self._current_step].compute(self._forwards, self._drifts2)

        # d) correct forwards using both drifts.
        for i in range(alive, n):
            self._forwards[i] += (self._drifts2[i] - self._drifts1[i]) / 2.0

        # e) update curve state.
        self._curve_state.set_on_forward_rates(self._forwards)

        self._current_step += 1
        return weight

    def current_step(self) -> int:
        return self._current_step

    def current_state(self) -> CurveState:
        return self._curve_state
