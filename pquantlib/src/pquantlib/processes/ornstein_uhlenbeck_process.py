"""Ornstein-Uhlenbeck mean-reverting 1-D process.

# C++ parity: ql/processes/ornsteinuhlenbeckprocess.{hpp,cpp} (v1.42.1).

Describes the SDE

    dx_t = a (b - x_t) dt + sigma dW_t

with constants ``a`` (speed of mean reversion), ``sigma`` (volatility),
``x0`` (initial state), and ``b`` (long-run mean). Closed-form
expectation and variance over a step ``dt``:

    E[x_{t0+dt} | x_{t0} = x0]   = b + (x0 - b) * exp(-a * dt)
    Var[x_{t0+dt} | x_{t0} = x0] = sigma^2 / (2a) * (1 - exp(-2 a dt))

with the algebraic ``a -> 0`` limit Var = sigma^2 * dt (matching the
C++ ``std::sqrt(QL_EPSILON)`` branch).
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.math.constants import QL_EPSILON
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D


class OrnsteinUhlenbeckProcess(StochasticProcess1D):
    """Mean-reverting Ornstein-Uhlenbeck process.

    # C++ parity: ``class OrnsteinUhlenbeckProcess : public StochasticProcess1D``
    # in ornsteinuhlenbeckprocess.hpp:41-63 (v1.42.1).
    """

    __slots__ = ("_level", "_speed", "_volatility", "_x0")

    def __init__(
        self,
        speed: float,
        vol: float,
        x0: float = 0.0,
        level: float = 0.0,
    ) -> None:
        """Construct an OU process.

        # C++ parity: ornsteinuhlenbeckprocess.cpp:27-34.

        Raises if ``vol`` is negative.
        """
        # No 1-D discretization needed: every closed-form override is
        # provided directly. We pass None to the base.
        super().__init__(discretization=None)
        qassert.require(vol >= 0.0, f"negative volatility given: {vol}")
        self._x0: float = float(x0)
        self._speed: float = float(speed)
        self._level: float = float(level)
        self._volatility: float = float(vol)

    # --- inspectors -----------------------------------------------------

    def x0(self) -> float:
        # C++ parity: ornsteinuhlenbeckprocess.hpp:67-69 (inline).
        return self._x0

    def speed(self) -> float:
        # C++ parity: ornsteinuhlenbeckprocess.hpp:71-73 (inline).
        return self._speed

    def volatility(self) -> float:
        # C++ parity: ornsteinuhlenbeckprocess.hpp:75-77 (inline).
        return self._volatility

    def level(self) -> float:
        # C++ parity: ornsteinuhlenbeckprocess.hpp:79-81 (inline).
        return self._level

    # --- StochasticProcess1D scalar interface ---------------------------

    def drift_1d(self, t: float, x: float) -> float:
        # C++ parity: ornsteinuhlenbeckprocess.hpp:83-85 (inline).
        return self._speed * (self._level - x)

    def diffusion_1d(self, t: float, x: float) -> float:
        # C++ parity: ornsteinuhlenbeckprocess.hpp:87-89 (inline).
        return self._volatility

    def expectation_1d(self, t0: float, x0: float, dt: float) -> float:
        # C++ parity: ornsteinuhlenbeckprocess.hpp:91-94 (inline).
        return self._level + (x0 - self._level) * math.exp(-self._speed * dt)

    def variance_1d(self, t0: float, x0: float, dt: float) -> float:
        # C++ parity: ornsteinuhlenbeckprocess.cpp:36-44.
        if abs(self._speed) < math.sqrt(QL_EPSILON):
            # Algebraic limit for small mean-reversion speed.
            return self._volatility * self._volatility * dt
        sigma2 = self._volatility * self._volatility
        return 0.5 * sigma2 / self._speed * (1.0 - math.exp(-2.0 * self._speed * dt))

    def std_deviation_1d(self, t0: float, x0: float, dt: float) -> float:
        # C++ parity: ornsteinuhlenbeckprocess.hpp:96-99 (inline).
        return math.sqrt(self.variance_1d(t0, x0, dt))
