"""HullWhiteForwardProcess — Hull-White short-rate under the T-forward measure.

# C++ parity: ql/processes/hullwhiteprocess.{hpp,cpp} (v1.42.1) —
# the ``HullWhiteForwardProcess`` declared alongside ``HullWhiteProcess``.

The base 1-D process is an Ornstein-Uhlenbeck process whose long-run
level is the initial instantaneous forward rate ``f(0,0)`` from the
provided yield curve. The drift adds a deterministic time-dependent
term ``alpha_drift(t)`` plus the forward-measure correction
``-B(t, T) * sigma^2``.

The single-factor Hull-White short-rate model itself (which holds this
process internally) lives in L4-B. This module is a bonus for L4-D
since it gives the cluster a complete forward-measure process suite.

Divergences from C++:

* The C++ ``HullWhiteForwardProcess`` ctor does NOT check that ``a``
  or ``sigma`` are non-negative (unlike the non-forward HW process).
  We preserve that asymmetry exactly.
* The forward-rate finite-difference shift inside ``drift`` matches
  C++ at ``0.0001`` (1 bp).
"""

from __future__ import annotations

import math

from pquantlib.math.constants import QL_EPSILON
from pquantlib.processes.forward_measure_process import ForwardMeasureProcess1D
from pquantlib.processes.ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency


class HullWhiteForwardProcess(ForwardMeasureProcess1D):
    """Hull-White process under the T-forward measure.

    # C++ parity: ``class HullWhiteForwardProcess`` in
    # ql/processes/hullwhiteprocess.hpp:61-86 (v1.42.1).
    """

    __slots__ = ("_a", "_h", "_process", "_sigma")

    def __init__(self, h: YieldTermStructure, a: float, sigma: float) -> None:
        # C++ parity: hullwhiteprocess.cpp:82-88 — no negativity checks
        # on a or sigma (asymmetry with HullWhiteProcess; we preserve it).
        super().__init__()
        self._h: YieldTermStructure = h
        self._a: float = float(a)
        self._sigma: float = float(sigma)
        # C++ initialises the underlying OU process with the initial
        # instantaneous forward rate as its long-run level.
        f0 = h.forward_rate(0.0, 0.0, Compounding.Continuous, Frequency.NoFrequency).rate()
        self._process: OrnsteinUhlenbeckProcess = OrnsteinUhlenbeckProcess(a, sigma, f0)

    # --- inspectors ----------------------------------------------------

    def a(self) -> float:
        return self._a

    def sigma(self) -> float:
        return self._sigma

    def x0(self) -> float:
        # C++ parity: hullwhiteprocess.cpp:90-92.
        return self._process.x0()

    # --- closed-form helpers -------------------------------------------

    def alpha(self, t: float) -> float:
        """Deterministic shift function ``alpha(t)``.

        # C++ parity: hullwhiteprocess.cpp:124-132. The ``a -> 0``
        # algebraic limit substitutes ``sigma_*t`` for ``(sigma/a)*(1-exp(-at))``.
        """
        if self._a > QL_EPSILON:
            alfa = (self._sigma / self._a) * (1.0 - math.exp(-self._a * t))
        else:
            alfa = self._sigma * t
        alfa *= 0.5 * alfa
        alfa += self._h.forward_rate(
            t, t, Compounding.Continuous, Frequency.NoFrequency
        ).rate()
        return alfa

    def M_T(self, s: float, t: float, T: float) -> float:  # noqa: N802, N803 — math symbol
        """Forward-measure drift integral.

        # C++ parity: hullwhiteprocess.cpp:134-146. The ``a -> 0``
        # algebraic limit gives ``(sigma^2/2)*(t-s)*(2*T-t-s)``.
        """
        if self._a > QL_EPSILON:
            coeff = (self._sigma * self._sigma) / (self._a * self._a)
            exp1 = math.exp(-self._a * (t - s))
            exp2 = math.exp(-self._a * (T - t))
            exp3 = math.exp(-self._a * (T + t - 2.0 * s))
            return coeff * (1.0 - exp1) - 0.5 * coeff * (exp2 - exp3)
        # a -> 0 algebraic limit
        coeff = (self._sigma * self._sigma) / 2.0
        return coeff * (t - s) * (2.0 * T - t - s)

    def B(self, t: float, T: float) -> float:  # noqa: N802, N803 — math symbol
        """HW bond-price coefficient ``B(t,T) = (1-exp(-a*(T-t)))/a``.

        # C++ parity: hullwhiteprocess.cpp:148-152. The ``a -> 0``
        # algebraic limit returns ``T-t``.
        """
        if self._a > QL_EPSILON:
            return (1.0 / self._a) * (1.0 - math.exp(-self._a * (T - t)))
        return T - t

    # --- StochasticProcess1D overrides ---------------------------------

    def drift_1d(self, t: float, x: float) -> float:
        # C++ parity: hullwhiteprocess.cpp:94-102. The forward-rate
        # derivative ``f_prime`` is computed by finite-difference at
        # a 1 bp shift (matches C++).
        alpha_drift = self._sigma * self._sigma / (2.0 * self._a) * (
            1.0 - math.exp(-2.0 * self._a * t)
        )
        shift = 0.0001
        f = self._h.forward_rate(t, t, Compounding.Continuous, Frequency.NoFrequency).rate()
        fup = self._h.forward_rate(
            t + shift, t + shift, Compounding.Continuous, Frequency.NoFrequency
        ).rate()
        f_prime = (fup - f) / shift
        alpha_drift += self._a * f + f_prime
        return (
            self._process.drift_1d(t, x)
            + alpha_drift
            - self.B(t, self._T) * self._sigma * self._sigma
        )

    def diffusion_1d(self, t: float, x: float) -> float:
        # C++ parity: hullwhiteprocess.cpp:104-106.
        return self._process.diffusion_1d(t, x)

    def expectation_1d(self, t0: float, x0: float, dt: float) -> float:
        # C++ parity: hullwhiteprocess.cpp:108-113.
        return (
            self._process.expectation_1d(t0, x0, dt)
            + self.alpha(t0 + dt)
            - self.alpha(t0) * math.exp(-self._a * dt)
            - self.M_T(t0, t0 + dt, self._T)
        )

    def std_deviation_1d(self, t0: float, x0: float, dt: float) -> float:
        # C++ parity: hullwhiteprocess.cpp:115-118.
        return self._process.std_deviation_1d(t0, x0, dt)

    def variance_1d(self, t0: float, x0: float, dt: float) -> float:
        # C++ parity: hullwhiteprocess.cpp:120-122.
        return self._process.variance_1d(t0, x0, dt)


__all__ = ["HullWhiteForwardProcess"]
