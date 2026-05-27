"""OrnsteinUhlenbeckProcess — mean-reverting 1-D process.

# C++ parity: ql/processes/ornsteinuhlenbeckprocess.{hpp,cpp} (v1.42.1).

The process is governed by
``dx_t = a (r - x_t) dt + sigma dW_t``
where ``a`` is the mean-reversion speed, ``r`` the long-term level
and ``sigma`` the volatility.

All ``drift`` / ``diffusion`` / ``expectation`` / ``variance`` /
``std_deviation`` overloads are closed-form (see C++ inline
definitions in ``ornsteinuhlenbeckprocess.hpp``). No discretization
is required.

The class is reused by G2Process / G2ForwardProcess / HullWhiteProcess /
HullWhiteForwardProcess as the underlying 1-D building block.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.math.constants import QL_EPSILON
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D


class OrnsteinUhlenbeckProcess(StochasticProcess1D):
    """Ornstein-Uhlenbeck mean-reverting 1-D process.

    # C++ parity: ``class OrnsteinUhlenbeckProcess`` in
    # ql/processes/ornsteinuhlenbeckprocess.hpp:41-63 (v1.42.1).
    """

    __slots__ = ("_level", "_speed", "_volatility", "_x0")

    def __init__(
        self,
        speed: float,
        vol: float,
        x0: float = 0.0,
        level: float = 0.0,
    ) -> None:
        # C++ parity: ornsteinuhlenbeckprocess.cpp:26-32 — negative-vol
        # check is the only constraint (speed can be zero/negative for
        # the algebraic-limit branch in variance()).
        super().__init__(discretization=None)
        qassert.require(vol >= 0.0, "negative volatility given")
        self._x0: float = float(x0)
        self._speed: float = float(speed)
        self._level: float = float(level)
        self._volatility: float = float(vol)

    # --- inspectors ----------------------------------------------------

    def x0(self) -> float:
        # C++ parity: ``OrnsteinUhlenbeckProcess::x0`` (inline).
        return self._x0

    def speed(self) -> float:
        """Mean-reversion speed ``a``.

        # C++ parity: ``OrnsteinUhlenbeckProcess::speed`` (inline).
        """
        return self._speed

    def volatility(self) -> float:
        """Process volatility ``sigma``.

        # C++ parity: ``OrnsteinUhlenbeckProcess::volatility`` (inline).
        """
        return self._volatility

    def level(self) -> float:
        """Long-run level ``r``.

        # C++ parity: ``OrnsteinUhlenbeckProcess::level`` (inline).
        """
        return self._level

    # --- scalar overrides ----------------------------------------------

    def drift_1d(self, t: float, x: float) -> float:
        # C++ parity: inline drift — ``speed_ * (level_ - x)``.
        return self._speed * (self._level - x)

    def diffusion_1d(self, t: float, x: float) -> float:
        # C++ parity: inline diffusion — ``volatility_``.
        return self._volatility

    def expectation_1d(self, t0: float, x0: float, dt: float) -> float:
        # C++ parity: inline expectation —
        # ``level_ + (x0 - level_) * std::exp(-speed_*dt)``.
        return self._level + (x0 - self._level) * math.exp(-self._speed * dt)

    def variance_1d(self, t0: float, x0: float, dt: float) -> float:
        # C++ parity: ornsteinuhlenbeckprocess.cpp:34-42 —
        # algebraic limit for tiny |speed|.
        if abs(self._speed) < math.sqrt(QL_EPSILON):
            return self._volatility * self._volatility * dt
        return (
            0.5
            * self._volatility
            * self._volatility
            / self._speed
            * (1.0 - math.exp(-2.0 * self._speed * dt))
        )

    def std_deviation_1d(self, t0: float, x0: float, dt: float) -> float:
        # C++ parity: inline stdDeviation — ``sqrt(variance)``.
        return math.sqrt(self.variance_1d(t0, x0, dt))


__all__ = ["OrnsteinUhlenbeckProcess"]
