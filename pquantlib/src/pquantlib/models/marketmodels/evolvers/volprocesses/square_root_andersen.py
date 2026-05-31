"""SquareRootAndersen — Andersen QE square-root variance process.

# C++ parity: ql/models/marketmodels/evolvers/volprocesses/
# squarerootandersen.{hpp,cpp} (v1.42.1).

Andersen's Quadratic-Exponential (QE) discretization of a square-root (CIR-type)
variance process ``dv = k (theta - v) dt + epsilon sqrt(v) dW``, used as the
external vol process of ``SVDDFwdRatePc``. Each evolution step is split into
``number_sub_steps`` QE sub-steps; the step standard deviation is the
``(w1, w2)``-weighted average of the sub-path variance.

# C++ parity quirk: the first ``number_sub_steps`` entries of ``eMinuskDt_`` are
# left zero-initialized (only ``dt_`` is filled for the first evolution time),
# so the very first evolution time's sub-steps use ``exp(-k dt) = 0`` in
# ``DoOneSubStep``. This is faithfully reproduced.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.models.marketmodels.evolvers.market_model_vol_process import (
    MarketModelVolProcess,
)


class SquareRootAndersen(MarketModelVolProcess):
    """Andersen QE discretization of a square-root variance process.

    # C++ parity: squarerootandersen.hpp/.cpp SquareRootAndersen.
    """

    def __init__(
        self,
        mean_level: float,
        reversion_speed: float,
        vol_var: float,
        v0: float,
        evolution_times: list[float],
        number_sub_steps: int,
        w1: float,
        w2: float,
        cut_point: float = 1.5,
    ) -> None:
        self._theta = mean_level
        self._k = reversion_speed
        self._epsilon = vol_var
        self._v0 = v0
        self._number_sub_steps = number_sub_steps
        size = len(evolution_times) * number_sub_steps
        self._dt = [0.0] * size
        self._e_minus_k_dt = [0.0] * size
        self._w1 = w1
        self._w2 = w2
        self._psi_c = cut_point
        self._v_path = [0.0] * (size + 1)

        j = 0
        for _ in range(number_sub_steps):
            self._dt[j] = evolution_times[0] / number_sub_steps
            j += 1
        for i in range(1, len(evolution_times)):
            dt = (evolution_times[i] - evolution_times[i - 1]) / number_sub_steps
            ekdt = math.exp(-self._k * dt)
            qassert.require(dt > 0.0, "Steps must be of positive size.")
            for _ in range(number_sub_steps):
                self._dt[j] = dt
                self._e_minus_k_dt[j] = ekdt
                j += 1
        self._v_path[0] = self._v0

        # evolving values.
        self._v = v0
        self._current_step = 0
        self._sub_step = 0
        self._state = [0.0]
        self._phi = CumulativeNormalDistribution()

    def variates_per_step(self) -> int:
        return self._number_sub_steps

    def number_steps(self) -> int:
        # C++ parity: returns dt_.size() * numberSubSteps_ (verbatim).
        return len(self._dt) * self._number_sub_steps

    def next_path(self) -> None:
        self._v = self._v0
        self._current_step = 0
        self._sub_step = 0

    def _do_one_sub_step(self, vt: float, z: float, j: int) -> float:
        # C++ parity: DoOneSubStep (mutates v by reference; here returns it).
        eminusk_t = self._e_minus_k_dt[j]
        m = self._theta + (vt - self._theta) * eminusk_t
        s2 = (
            vt * self._epsilon * self._epsilon * eminusk_t * (1 - eminusk_t) / self._k
            + self._theta
            * self._epsilon
            * self._epsilon
            * (1 - eminusk_t)
            * (1 - eminusk_t)
            / (2 * self._k)
        )
        s = math.sqrt(s2)
        psi = s * s / (m * m)
        if psi <= self._psi_c:
            psiinv = 1.0 / psi
            b2 = 2.0 * psiinv - 1 + math.sqrt(2 * psiinv * (2 * psiinv - 1.0))
            b = math.sqrt(b2)
            a = m / (1 + b2)
            return a * (b + z) * (b + z)
        p = (psi - 1.0) / (psi + 1.0)
        beta = (1.0 - p) / m
        u = self._phi(z)
        if u < p:
            return 0.0
        return math.log((1.0 - p) / (1.0 - u)) / beta

    def next_step(self, variates: list[float]) -> float:
        # C++ parity: nextstep.
        for j in range(self._number_sub_steps):
            self._v = self._do_one_sub_step(self._v, variates[j], self._sub_step)
            self._sub_step += 1
            self._v_path[self._sub_step] = self._v
        self._current_step += 1
        return 1.0  # no importance sampling here

    def step_sd(self) -> float:
        # C++ parity: stepSd.
        qassert.require(self._current_step > 0, "nextStep must be called before stepSd")
        step_variance = 0.0
        last_step_start = (self._current_step - 1) * self._number_sub_steps
        for k in range(self._number_sub_steps):
            step_variance += (
                self._w1 * self._v_path[k + last_step_start]
                + self._w2 * self._v_path[k + last_step_start + 1]
            )
        step_variance /= self._number_sub_steps
        return math.sqrt(step_variance)

    def state_variables(self) -> list[float]:
        # C++ parity: stateVariables.
        self._state[0] = self._v
        return self._state

    def number_state_variables(self) -> int:
        return 1
