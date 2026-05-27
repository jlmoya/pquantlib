"""CoxIngersollRossProcess — square-root mean-reverting 1-D process.

# C++ parity: ql/processes/coxingersollrossprocess.{hpp,cpp} (v1.42.1).

The process is governed by
``dx_t = k (theta - x_t) dt + sigma sqrt(x_t) dW_t``
where ``k`` is the mean-reversion speed (``speed``), ``theta`` the
long-term mean (``level``) and ``sigma`` the volatility coefficient.

Caveat — variance formula divergence (v1.42.1, coxingersollrossprocess.cpp:32-38):
the C++ code computes

    var = x0_ * (sigma^2 / k) * (exp(-k*dt) - exp(-2*k*dt))
        + level_ * (sigma^2 / k) * (1 - exp(-k*dt))^2

which uses the *initial* ``x0_`` rather than the ``x0`` argument to
``variance(t0, x0, dt)``. This is a documented quirk of the C++
implementation: the function signature accepts ``x0`` but ignores it.
We preserve that behaviour bit-exactly to cross-validate against the
C++ probe; downstream callers that want a "true" conditional variance
at non-initial ``x0`` should use the QE-scheme ``evolve`` path or
roll their own.

# C++ parity: discretization uses Andersen's Quadratic Exponential
# (QE) scheme — implemented in ``evolve_1d``. The inline ``drift``
# matches OU exactly; ``diffusion`` is the *constant* coefficient
# (sigma — not sigma*sqrt(x); the sqrt(x) is absorbed into the QE
# variance formula). This is one of the documented quirks of the C++
# CIR process — the ``diffusion`` accessor reports ``sigma``, not
# ``sigma*sqrt(x)`` as the SDE would suggest. We preserve parity.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D

_PHI = CumulativeNormalDistribution()


class CoxIngersollRossProcess(StochasticProcess1D):
    """CIR mean-reverting square-root 1-D process.

    # C++ parity: ``class CoxIngersollRossProcess`` in
    # ql/processes/coxingersollrossprocess.hpp:45-70 (v1.42.1).
    """

    __slots__ = ("_level", "_speed", "_volatility", "_x0")

    def __init__(
        self,
        speed: float,
        vol: float,
        x0: float = 0.0,
        level: float = 0.0,
    ) -> None:
        # C++ parity: coxingersollrossprocess.cpp:24-30 — negative-vol
        # check is the only ctor constraint.
        super().__init__(discretization=None)
        qassert.require(vol >= 0.0, "negative volatility given")
        self._x0: float = float(x0)
        self._speed: float = float(speed)
        self._level: float = float(level)
        self._volatility: float = float(vol)

    # --- inspectors ----------------------------------------------------

    def x0(self) -> float:
        # C++ parity: inline x0.
        return self._x0

    def speed(self) -> float:
        """Mean-reversion speed ``k``.

        # C++ parity: inline speed.
        """
        return self._speed

    def volatility(self) -> float:
        """Process volatility ``sigma``.

        # C++ parity: inline volatility.
        """
        return self._volatility

    def level(self) -> float:
        """Long-run mean ``theta``.

        # C++ parity: inline level.
        """
        return self._level

    # --- scalar overrides ----------------------------------------------

    def drift_1d(self, t: float, x: float) -> float:
        # C++ parity: inline drift — ``speed_ * (level_ - x)``.
        return self._speed * (self._level - x)

    def diffusion_1d(self, t: float, x: float) -> float:
        # C++ parity: inline diffusion — ``volatility_`` (NOT
        # ``volatility_*sqrt(x)``, despite the SDE form). See module
        # docstring for the parity note.
        return self._volatility

    def expectation_1d(self, t0: float, x0: float, dt: float) -> float:
        # C++ parity: inline expectation —
        # ``level_ + (x0 - level_) * std::exp(-speed_*dt)``.
        return self._level + (x0 - self._level) * math.exp(-self._speed * dt)

    def variance_1d(self, t0: float, x0: float, dt: float) -> float:
        # C++ parity: coxingersollrossprocess.cpp:32-38. Note that the
        # C++ implementation uses ``x0_`` (the stored initial value) —
        # not the ``x0`` parameter — for the leading term. See module
        # docstring.
        exponent1 = math.exp(-self._speed * dt)
        exponent2 = math.exp(-2.0 * self._speed * dt)
        fraction = (self._volatility * self._volatility) / self._speed
        leading = self._x0 * fraction * (exponent1 - exponent2)
        return leading + self._level * fraction * (1.0 - exponent1) * (1.0 - exponent1)

    def std_deviation_1d(self, t0: float, x0: float, dt: float) -> float:
        # C++ parity: inline stdDeviation — ``sqrt(variance)``.
        return math.sqrt(self.variance_1d(t0, x0, dt))

    def evolve_1d(self, t0: float, x0: float, dt: float, dw: float) -> float:
        # C++ parity: coxingersollrossprocess.hpp:108-138 — Andersen QE
        # scheme. The Brownian increment ``dw`` is interpreted as a
        # standard-normal draw (zero-mean unit-variance), matching the
        # C++ caller's convention.
        ex = math.exp(-self._speed * dt)
        m = self._level + (x0 - self._level) * ex
        sigma2 = self._volatility * self._volatility
        s2 = (
            x0 * sigma2 * ex / self._speed * (1.0 - ex)
            + self._level * sigma2 / (2.0 * self._speed) * (1.0 - ex) * (1.0 - ex)
        )
        if m == 0.0:
            # Defensive: m can be exactly zero when level_=0 and x0=0.
            # The C++ code would divide-by-zero below; mirror that with
            # an explicit return.
            return 0.0
        psi = s2 / (m * m)
        if psi <= 1.5:
            b2 = 2.0 / psi - 1.0 + math.sqrt(2.0 / psi * (2.0 / psi - 1.0))
            b = math.sqrt(b2)
            a = m / (1.0 + b2)
            return a * (b + dw) * (b + dw)
        # large-psi branch — log-CDF inversion
        p = (psi - 1.0) / (psi + 1.0)
        beta = (1.0 - p) / m
        u = _PHI(dw)
        return 0.0 if u <= p else math.log((1.0 - p) / (1.0 - u)) / beta


__all__ = ["CoxIngersollRossProcess"]
