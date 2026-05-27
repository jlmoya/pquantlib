"""Cox-Ingersoll-Ross square-root mean-reverting 1-D process.

# C++ parity: ql/processes/coxingersollrossprocess.{hpp,cpp} (v1.42.1).

Describes the SDE

    dx_t = k (theta - x_t) dt + sigma sqrt(x_t) dW_t

with mean-reversion speed ``k``, long-run mean ``theta``, volatility
``sigma``, and initial state ``x0``. The conditional expectation is

    E[x_{t0+dt} | x_{t0} = x0] = theta + (x0 - theta) * exp(-k*dt)

and the conditional variance is

    Var[x_{t0+dt}] = x0 * (sigma^2 / k) * (e^{-k dt} - e^{-2k dt})
                   + theta * (sigma^2 / (2k)) * (1 - e^{-k dt})^2

(matches C++ ``CoxIngersollRossProcess::variance`` literally, using the
fields ``x0_`` and ``level_`` from the ctor, not the current state).

The evolution step ``evolve_1d`` uses Andersen's Quadratic Exponential
scheme. The ``dw`` argument is interpreted as a standard normal increment
(consistent with the C++ ``evolve(t0, x0, dt, dw)`` convention used by
trees and Monte-Carlo wrappers).
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D


class CoxIngersollRossProcess(StochasticProcess1D):
    """Square-root mean-reverting CIR process.

    # C++ parity: ``class CoxIngersollRossProcess : public StochasticProcess1D``
    # in coxingersollrossprocess.hpp (v1.42.1).

    The ctor signature mirrors C++ exactly:
    ``CoxIngersollRossProcess(speed, vol, x0=0.0, level=0.0)``.
    """

    __slots__ = ("_level", "_speed", "_volatility", "_x0")

    def __init__(
        self,
        speed: float,
        vol: float,
        x0: float = 0.0,
        level: float = 0.0,
    ) -> None:
        """Construct a CIR process.

        # C++ parity: coxingersollrossprocess.cpp ctor (v1.42.1) —
        # ``x0_(x0), speed_(speed), level_(level), volatility_(vol)``.

        Raises if ``vol`` is negative.
        """
        super().__init__(discretization=None)
        qassert.require(vol >= 0.0, f"negative volatility given: {vol}")
        self._x0: float = float(x0)
        self._speed: float = float(speed)
        self._level: float = float(level)
        self._volatility: float = float(vol)

    # --- inspectors -----------------------------------------------------

    def x0(self) -> float:
        # C++ parity: coxingersollrossprocess.hpp x0() inline.
        return self._x0

    def speed(self) -> float:
        # C++ parity: coxingersollrossprocess.hpp speed() inline.
        return self._speed

    def volatility(self) -> float:
        # C++ parity: coxingersollrossprocess.hpp volatility() inline.
        return self._volatility

    def level(self) -> float:
        # C++ parity: coxingersollrossprocess.hpp level() inline.
        return self._level

    # --- StochasticProcess1D scalar interface ---------------------------

    def drift_1d(self, t: float, x: float) -> float:
        # C++ parity: coxingersollrossprocess.hpp drift(Time, Real) inline.
        return self._speed * (self._level - x)

    def diffusion_1d(self, t: float, x: float) -> float:
        # C++ parity: coxingersollrossprocess.hpp diffusion(Time, Real) inline.
        #
        # NOTE: This mirrors the C++ implementation exactly. The C++
        # version returns ``volatility_`` (a constant!) rather than
        # ``volatility_ * sqrt(x)`` — see coxingersollrossprocess.hpp
        # line 95-97. The "sqrt(x)" factor lives instead in the
        # ``evolve`` discretization (the Quadratic Exponential scheme).
        # This is a known C++ idiosyncrasy preserved for parity.
        return self._volatility

    def expectation_1d(self, t0: float, x0: float, dt: float) -> float:
        # C++ parity: coxingersollrossprocess.hpp expectation() inline.
        return self._level + (x0 - self._level) * math.exp(-self._speed * dt)

    def variance_1d(self, t0: float, x0: float, dt: float) -> float:
        # C++ parity: coxingersollrossprocess.cpp variance() —
        # uses *constructor-time* x0_ and level_, not the current x0
        # argument. This idiosyncrasy is preserved for parity.
        exponent1 = math.exp(-self._speed * dt)
        exponent2 = math.exp(-2.0 * self._speed * dt)
        fraction = (self._volatility * self._volatility) / self._speed
        return (
            self._x0 * fraction * (exponent1 - exponent2)
            + self._level * fraction * (1.0 - exponent1) * (1.0 - exponent1)
        )

    def std_deviation_1d(self, t0: float, x0: float, dt: float) -> float:
        # C++ parity: coxingersollrossprocess.hpp stdDeviation() inline.
        return math.sqrt(self.variance_1d(t0, x0, dt))

    def evolve_1d(self, t0: float, x0: float, dt: float, dw: float) -> float:
        """Andersen Quadratic-Exponential evolution step.

        # C++ parity: coxingersollrossprocess.hpp evolve() inline.

        Reference: Leif Andersen, "Efficient Simulation of the Heston
        Stochastic Volatility Model".
        """
        ex = math.exp(-self._speed * dt)
        m = self._level + (x0 - self._level) * ex
        s2 = (
            x0 * self._volatility * self._volatility * ex / self._speed * (1.0 - ex)
            + self._level * self._volatility * self._volatility / (2.0 * self._speed) * (1.0 - ex) * (1.0 - ex)
        )
        psi = s2 / (m * m)

        if psi <= 1.5:
            two_over_psi = 2.0 / psi
            b2 = two_over_psi - 1.0 + math.sqrt(two_over_psi * (two_over_psi - 1.0))
            b = math.sqrt(b2)
            a = m / (1.0 + b2)
            return a * (b + dw) * (b + dw)
        # psi > 1.5 branch
        p = (psi - 1.0) / (psi + 1.0)
        beta = (1.0 - p) / m
        u = CumulativeNormalDistribution()(dw)
        if u <= p:
            return 0.0
        return math.log((1.0 - p) / (1.0 - u)) / beta
