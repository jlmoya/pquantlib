"""GemanRoncoroniProcess — energy spot-price process (mean-revert + spikes).

# C++ parity: ql/experimental/processes/gemanroncoroniprocess.{hpp,cpp}
# (v1.42.1).

Describes the Geman-Roncoroni electricity-price process::

    dE(t) = [d/dt mu(t) + theta1 (mu(t) - E(t^-))] dt
            + sigma dW(t) + h(E(t^-)) dJ(t)
    mu(t) = alpha + beta t + gamma cos(eps + 2 pi t)
            + delta cos(zeta + 4 pi t)

The deterministic-seasonality mean ``mu`` plus a fast mean reversion
(``theta1``) and an asymmetric jump term (intensity ``theta2``, jump
size driven by ``theta3`` / ``psi``) capture the spike structure of
energy spot prices.

Python notes:

* ``evolve_1d(t0, x0, dt, dw)`` mirrors the C++ 4-arg ``evolve``: it
  lazily builds a Mersenne-Twister uniform RNG (seed
  ``int(1234 * dw + 56789)``) and draws the two jump-driver uniforms,
  then forwards to the deterministic 5-arg core. This is *not* a
  reproducible-across-dw path (the seed depends on ``dw``) — it matches
  the C++ behaviour exactly.
* ``evolve_du(t0, x0, dt, dw, du)`` is the deterministic 5-arg core
  (C++ ``evolve(Time, Real, Time, Real, const Array&)``): the caller
  supplies the jump-driver Array ``du`` (``du[0]`` = inter-arrival
  uniform, ``du[1]`` = jump-size uniform).
"""

from __future__ import annotations

import math
from typing import final

import numpy as np
import numpy.typing as npt

from pquantlib.math.randomnumbers.mersenne_twister import MersenneTwisterUniformRng
from pquantlib.processes.euler_discretization import EulerDiscretization
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D


@final
class GemanRoncoroniProcess(StochasticProcess1D):
    """Geman-Roncoroni energy spot-price process.

    # C++ parity: ``class GemanRoncoroniProcess : public
    # StochasticProcess1D`` in gemanroncoroniprocess.hpp:46.
    """

    __slots__ = (
        "_a",
        "_alpha",
        "_b",
        "_beta",
        "_d",
        "_delta",
        "_eps",
        "_gamma",
        "_k",
        "_psi",
        "_sig2",
        "_tau",
        "_theta1",
        "_theta2",
        "_theta3",
        "_urng",
        "_x0",
        "_zeta",
    )

    def __init__(
        self,
        x0: float,
        alpha: float,
        beta: float,
        gamma: float,
        delta: float,
        eps: float,
        zeta: float,
        d: float,
        k: float,
        tau: float,
        sig2: float,
        a: float,
        b: float,
        theta1: float,
        theta2: float,
        theta3: float,
        psi: float,
    ) -> None:
        # C++ parity: ctor forwards an EulerDiscretization to the base.
        super().__init__(EulerDiscretization())
        self._x0: float = float(x0)
        self._alpha: float = float(alpha)
        self._beta: float = float(beta)
        self._gamma: float = float(gamma)
        self._delta: float = float(delta)
        self._eps: float = float(eps)
        self._zeta: float = float(zeta)
        self._d: float = float(d)
        self._k: float = float(k)
        self._tau: float = float(tau)
        self._sig2: float = float(sig2)
        self._a: float = float(a)
        self._b: float = float(b)
        self._theta1: float = float(theta1)
        self._theta2: float = float(theta2)
        self._theta3: float = float(theta3)
        self._psi: float = float(psi)
        self._urng: MersenneTwisterUniformRng | None = None

    def x0(self) -> float:
        # C++ parity: gemanroncoroniprocess.cpp x0().
        return self._x0

    def drift_1d(self, t: float, x: float) -> float:
        """Drift = mu'(t) + theta1 (mu(t) - x).

        # C++ parity: ``GemanRoncoroniProcess::drift``.
        """
        mu = (
            self._alpha
            + self._beta * t
            + self._gamma * math.cos(self._eps + 2.0 * math.pi * t)
            + self._delta * math.cos(self._zeta + 4.0 * math.pi * t)
        )
        mu_prime = (
            self._beta
            - self._gamma * 2.0 * math.pi * math.sin(self._eps + 2.0 * math.pi * t)
            - self._delta * 4.0 * math.pi * math.sin(self._zeta + 4.0 * math.pi * t)
        )
        return mu_prime + self._theta1 * (mu - x)

    def diffusion_1d(self, t: float, x: float) -> float:
        """Diffusion = sqrt(sig2 + a cos^2(pi t + b)).

        # C++ parity: ``GemanRoncoroniProcess::diffusion`` (ignores x).
        """
        return math.sqrt(self._sig2 + self._a * (math.cos(math.pi * t + self._b) ** 2))

    def std_deviation_1d(self, t0: float, x0: float, dt: float) -> float:
        """Closed-form OU-style standard deviation over ``[t0, t0+dt]``.

        # C++ parity: ``GemanRoncoroniProcess::stdDeviation`` —
        # ``sqrt(sig2t / (2 theta1) (1 - exp(-2 theta1 dt)))`` where
        # ``sig2t = sig2 + a cos^2(pi t0 + b)``.
        """
        sig2t = self._sig2 + self._a * (math.cos(math.pi * t0 + self._b) ** 2)
        return math.sqrt(
            sig2t / (2.0 * self._theta1) * (1.0 - math.exp(-2.0 * self._theta1 * dt))
        )

    def evolve_1d(self, t0: float, x0: float, dt: float, dw: float) -> float:
        """Evolve with an internal RNG for the jump drivers.

        # C++ parity: ``GemanRoncoroniProcess::evolve(Time, Real, Time,
        # Real)`` — lazily seeds an MT uniform RNG with
        # ``(unsigned long)(1234 * dw + 56789)`` and draws du[0], du[1].
        """
        if self._urng is None:
            self._urng = MersenneTwisterUniformRng(int(1234 * dw + 56789))
        du = np.zeros(3, dtype=np.float64)
        du[0] = self._urng.next().value
        du[1] = self._urng.next().value
        return self.evolve_du(t0, x0, dt, dw, du)

    def evolve_du(
        self,
        t0: float,
        x0: float,
        dt: float,
        dw: float,
        du: npt.NDArray[np.float64],
    ) -> float:
        """Deterministic evolve given the jump-driver Array ``du``.

        # C++ parity: ``GemanRoncoroniProcess::evolve(Time, Real, Time,
        # Real, const Array&)``.

        ``du[0]`` is the inter-arrival uniform; ``du[1]`` the jump-size
        uniform. Below the spike threshold ``mu + d`` an OU step is taken
        (plus a jump if the inter-arrival time falls inside ``dt``);
        above it the price reverts down by the jump size ``j``.
        """
        t = t0 + 0.5 * dt
        mu = (
            self._alpha
            + self._beta * t
            + self._gamma * math.cos(self._eps + 2.0 * math.pi * t)
            + self._delta * math.cos(self._zeta + 4.0 * math.pi * t)
        )
        j = (
            -1.0
            / self._theta3
            * math.log(1.0 + du[1] * (math.exp(-self._theta3 * self._psi) - 1.0))
        )

        if x0 <= mu + self._d:
            # OU step via the base StochasticProcess1D.evolve at time t.
            ret_val = StochasticProcess1D.evolve_1d(self, t, x0, dt, dw)
            jump_intensity = self._theta2 * (
                2.0 / (1.0 + abs(math.sin(math.pi * (t - self._tau) / self._k))) - 1.0
            )
            interarrival = -1.0 / jump_intensity * math.log(du[0])
            if interarrival < dt:
                ret_val += j
            return ret_val
        return x0 - j


__all__ = ["GemanRoncoroniProcess"]
