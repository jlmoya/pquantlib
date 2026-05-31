"""GeneralizedOrnsteinUhlenbeckProcess — OU with time-dependent coefficients.

# C++ parity: ql/experimental/shortrate/generalizedornsteinuhlenbeckprocess.{hpp,cpp}
# (v1.42.1).

Describes the Ornstein-Uhlenbeck SDE

    dx = a(t) (level - x_t) dt + sigma(t) dW_t

where both the mean-reversion speed ``a(t)`` and the volatility
``sigma(t)`` are user-supplied callables of time (piecewise-linear in
the canonical use, but any callable is accepted). Unlike the plain
``OrnsteinUhlenbeckProcess`` (constant ``a`` / ``sigma``) and the
``ExtendedOrnsteinUhlenbeckProcess`` (time-dependent mean but constant
``a`` / ``sigma``), this one varies both rate coefficients.

The closed-form expectation / variance use the *current-time* values
``a(t0)`` / ``sigma(t0)`` exactly as the C++ class does (the
coefficients are treated as locally constant over the step ``dt``):

    E[x_{t0+dt} | x0]   = level + (x0 - level) * exp(-a(t0) * dt)
    Var[x_{t0+dt} | x0] = sigma(t0)^2 / (2 a(t0)) * (1 - exp(-2 a(t0) dt))

with the algebraic ``a(t0) -> 0`` limit ``Var = sigma(t0)^2 * dt``
(matching the C++ ``std::sqrt(QL_EPSILON)`` branch). This is the kernel
process underlying ``GeneralizedHullWhite``'s numerical (trinomial-tree)
dynamics.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import final

from pquantlib import qassert
from pquantlib.math.constants import QL_EPSILON
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D


@final
class GeneralizedOrnsteinUhlenbeckProcess(StochasticProcess1D):
    """Ornstein-Uhlenbeck process with time-dependent speed and volatility.

    # C++ parity: ``class GeneralizedOrnsteinUhlenbeckProcess : public
    # StochasticProcess1D`` in generalizedornsteinuhlenbeckprocess.hpp:43-69.

    Parameters
    ----------
    speed
        Mean-reversion speed callable ``a(t) -> float``.
    vol
        Volatility callable ``sigma(t) -> float``.
    x0
        Initial state (must be non-negative).
    level
        Long-run mean (must be non-negative).
    """

    __slots__ = ("_level", "_speed", "_volatility", "_x0")

    def __init__(
        self,
        speed: Callable[[float], float],
        vol: Callable[[float], float],
        x0: float = 0.0,
        level: float = 0.0,
    ) -> None:
        # C++ parity: generalizedornsteinuhlenbeckprocess.cpp:26-32.
        # No 1-D discretization: closed-form drift/diffusion/expectation/
        # variance are all provided directly.
        super().__init__(discretization=None)
        qassert.require(x0 >= 0.0, f"negative initial data given: {x0}")
        qassert.require(level >= 0.0, f"negative level given: {level}")
        self._x0: float = float(x0)
        self._level: float = float(level)
        self._speed: Callable[[float], float] = speed
        self._volatility: Callable[[float], float] = vol

    # --- StochasticProcess1D interface ----------------------------------

    def x0(self) -> float:
        # C++ parity: generalizedornsteinuhlenbeckprocess.cpp:34-36.
        return self._x0

    def drift_1d(self, t: float, x: float) -> float:
        # C++ parity: generalizedornsteinuhlenbeckprocess.cpp:38-40 —
        # speed_(t) * (level_ - x).
        return self._speed(t) * (self._level - x)

    def diffusion_1d(self, t: float, x: float) -> float:
        # C++ parity: generalizedornsteinuhlenbeckprocess.cpp:42-44 —
        # volatility_(t) (state-independent).
        return self._volatility(t)

    def expectation_1d(self, t0: float, x0: float, dt: float) -> float:
        # C++ parity: generalizedornsteinuhlenbeckprocess.cpp:46-49 —
        # level_ + (x0 - level_) * exp(-speed_(t0) * dt).
        return self._level + (x0 - self._level) * math.exp(-self._speed(t0) * dt)

    def variance_1d(self, t0: float, x0: float, dt: float) -> float:
        # C++ parity: generalizedornsteinuhlenbeckprocess.cpp:56-67.
        speed = self._speed(t0)
        vol = self._volatility(t0)
        if speed < math.sqrt(QL_EPSILON):
            # Algebraic limit for small mean-reversion speed.
            return vol * vol * dt
        return 0.5 * vol * vol / speed * (1.0 - math.exp(-2.0 * speed * dt))

    def std_deviation_1d(self, t0: float, x0: float, dt: float) -> float:
        # C++ parity: generalizedornsteinuhlenbeckprocess.cpp:51-54.
        return math.sqrt(self.variance_1d(t0, x0, dt))

    # --- inspectors -----------------------------------------------------

    def speed(self, t: float) -> float:
        # C++ parity: generalizedornsteinuhlenbeckprocess.cpp:70-72.
        return self._speed(t)

    def volatility(self, t: float) -> float:
        # C++ parity: generalizedornsteinuhlenbeckprocess.cpp:74-76.
        return self._volatility(t)

    def level(self) -> float:
        # C++ parity: generalizedornsteinuhlenbeckprocess.cpp:78-80.
        return self._level


__all__ = ["GeneralizedOrnsteinUhlenbeckProcess"]
