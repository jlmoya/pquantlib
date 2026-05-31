"""LogNormalFwdRateEulerConstrained — constrained Euler forward-rate evolver.

# C++ parity:
# ql/models/marketmodels/evolvers/lognormalfwdrateeulerconstrained.{hpp,cpp}
# (v1.42.1).

A ``ConstrainedEvolver`` (Fries-Joshi proxy simulation): the constrained Euler
evolver pins a single forward rate per step to a target value via importance
sampling, returning the likelihood-ratio weight so that downstream estimators
remain unbiased. Currently only single-forward-rate constraints
(``end == start + 1``) are supported (matching C++).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.models.marketmodels.constrained_evolver import ConstrainedEvolver
from pquantlib.models.marketmodels.curvestates.lmm_curve_state import LMMCurveState
from pquantlib.models.marketmodels.driftcomputation.lmm_drift_calculator import (
    LMMDriftCalculator,
)
from pquantlib.models.marketmodels.evolution_description import check_compatibility

if TYPE_CHECKING:
    from pquantlib.models.marketmodels.brownian_generator import (
        BrownianGeneratorFactory,
    )
    from pquantlib.models.marketmodels.curve_state import CurveState
    from pquantlib.models.marketmodels.market_model import MarketModel


class LogNormalFwdRateEulerConstrained(ConstrainedEvolver):
    """Constrained Euler forward-rate evolver (importance sampling).

    # C++ parity: lognormalfwdrateeulerconstrained.hpp/.cpp
    # LogNormalFwdRateEulerConstrained.
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

        # constraint inputs (set via set_constraint_type / set_this_constraint).
        self._start_index_of_swap_rate: list[int] = []
        self._end_index_of_swap_rate: list[int] = []
        self._rate_constraints: list[float] = []
        self._is_constraint_active: list[bool] = []
        self._covariances: list[list[float]] = []

        check_compatibility(market_model.evolution(), numeraires)

        steps = market_model.evolution().number_of_steps()
        self._generator = factory.create(self._number_of_factors, steps - initial_step)
        self._current_step = initial_step

        self._calculators: list[LMMDriftCalculator] = []
        self._fixed_drifts: list[list[float]] = []
        self._variances: list[list[float]] = []
        rate_taus = market_model.evolution().rate_taus()
        for j in range(steps):
            a = market_model.pseudo_root(j)
            self._calculators.append(
                LMMDriftCalculator(
                    a, self._displacements, rate_taus, numeraires[j], self._alive[j]
                )
            )
            variances = np.einsum("ij,ij->i", a, a)
            self._variances.append([float(v) for v in variances])
            self._fixed_drifts.append([-0.5 * float(v) for v in variances])

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

    def set_constraint_type(
        self,
        start_index_of_swap_rate: list[int],
        end_index_of_swap_rate: list[int],
    ) -> None:
        # C++ parity: setConstraintType.
        qassert.require(
            len(start_index_of_swap_rate) == len(self._numeraires),
            "Size mismatch in constraint specification.",
        )
        qassert.require(
            len(end_index_of_swap_rate) == len(self._numeraires),
            "Size mismatch in constraint specification.",
        )
        self._start_index_of_swap_rate = list(start_index_of_swap_rate)
        self._end_index_of_swap_rate = list(end_index_of_swap_rate)

        self._covariances = []
        n = self._number_of_rates
        for i in range(len(self._start_index_of_swap_rate)):
            qassert.require(
                self._start_index_of_swap_rate[i] + 1 == self._end_index_of_swap_rate[i],
                "constrained euler currently only implemented for forward rates",
            )
            a = self._market_model.pseudo_root(self._current_step)
            covariances = [0.0] * n
            si = self._start_index_of_swap_rate[i]
            for j in range(n):
                cov = 0.0
                for k in range(self._number_of_factors):
                    cov += float(a[si, k]) * float(a[j, k])
                covariances[j] = cov
            self._covariances.append(covariances)

    def set_this_constraint(
        self,
        rate_constraints: list[float],
        is_constraint_active: list[bool],
    ) -> None:
        # C++ parity: setThisConstraint.
        qassert.require(
            len(rate_constraints) == len(self._numeraires),
            "wrong number of constraints specified",
        )
        qassert.require(
            len(is_constraint_active) == len(self._numeraires),
            "wrong number of isConstraintActive specified",
        )
        self._rate_constraints = [
            math.log(rate_constraints[i] + self._displacements[i])
            for i in range(len(rate_constraints))
        ]
        self._is_constraint_active = list(is_constraint_active)

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

        # b) evolve forwards up to T2 using D1.
        weight = self._generator.next_step(self._brownians)
        a = self._market_model.pseudo_root(self._current_step)
        fixed_drift = self._fixed_drifts[self._current_step]
        brownians = np.asarray(self._brownians, dtype=np.float64)

        alive = self._alive[self._current_step]
        n = self._number_of_rates
        for i in range(alive, n):
            self._log_forwards[i] += self._drifts1[i] + fixed_drift[i]
            self._log_forwards[i] += float(a[i] @ brownians)

        # check constraint active for this step.
        if self._is_constraint_active[self._current_step]:
            index = self._start_index_of_swap_rate[self._current_step]
            # compute error.
            required_shift = self._rate_constraints[self._current_step] - self._log_forwards[index]
            multiplier = required_shift / self._variances[self._current_step][index]
            # shift each rate by multiplier * weighting of index rate.
            for i in range(alive, n):
                self._log_forwards[i] += multiplier * self._covariances[self._current_step][i]
            # likelihood-ratio weight: divide original density by shifted density.
            weights_effect = 1.0
            phi = CumulativeNormalDistribution()
            for k in range(self._number_of_factors):
                shift = multiplier * float(a[index, k])
                original_density = phi.derivative(self._brownians[k] + shift)
                new_density = phi.derivative(self._brownians[k])
                weights_effect *= original_density / new_density
            weight *= weights_effect

        for i in range(alive, n):
            self._forwards[i] = math.exp(self._log_forwards[i]) - self._displacements[i]

        # c) update curve state.
        self._curve_state.set_on_forward_rates(self._forwards)

        self._current_step += 1
        return weight

    def current_step(self) -> int:
        return self._current_step

    def current_state(self) -> CurveState:
        return self._curve_state
