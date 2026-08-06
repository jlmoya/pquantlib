"""HullWhiteProcess — Hull-White short rate under the risk-neutral measure.

# C++ parity: ql/processes/hullwhiteprocess.{hpp,cpp} (v1.43) —
# ``class HullWhiteProcess : public StochasticProcess1D``
# (hullwhiteprocess.hpp:35-57, hullwhiteprocess.cpp:24-80).

The process is an Ornstein-Uhlenbeck process whose long-run level is the
initial instantaneous forward rate ``f(0,0)`` taken from the supplied yield
curve, with a deterministic, time-dependent drift shift added on top:

    alpha_drift(t) = sigma^2 / (2a) * (1 - exp(-2 a t)) + a f(t) + f'(t)

``f'(t)`` is a forward finite difference at a 1 bp shift, exactly as in C++.

The T-forward-measure sibling ``HullWhiteForwardProcess`` lives in
``hull_white_forward_process.py``; this class is its risk-neutral counterpart
and differs in exactly two places: ``drift`` omits the ``-B(t,T) sigma^2``
correction and ``expectation`` omits the ``-M_T(t0, t0+dt, T)`` one.

Divergences from C++:

* # C++ parity divergence: C++ takes ``Handle<YieldTermStructure>``; this
  port does not implement ``Handle<T>`` and threads the curve directly.
* C++ ``HullWhiteProcess`` DOES check ``a >= 0`` and ``sigma >= 0``, while
  ``HullWhiteForwardProcess`` does not. The asymmetry is real and is
  preserved on both sides.
* ``alpha`` guards the ``a -> 0`` limit with ``a > QL_EPSILON`` but
  ``drift`` does not, so ``drift`` divides by ``a`` unguarded. That is the
  C++ behaviour (hullwhiteprocess.cpp:38-46) and is not "fixed" here.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.math.constants import QL_EPSILON
from pquantlib.processes.ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency


class HullWhiteProcess(StochasticProcess1D):
    """Hull-White short-rate process under the risk-neutral measure.

    # C++ parity: ``class HullWhiteProcess`` (hullwhiteprocess.hpp:35-57).
    """

    __slots__ = ("_a", "_h", "_process", "_sigma")

    def __init__(self, h: YieldTermStructure, a: float, sigma: float) -> None:
        # C++ parity: hullwhiteprocess.cpp:24-32 — the underlying OU process
        # is seeded with the initial instantaneous forward rate as its level,
        # then both parameters are range-checked.
        super().__init__()
        self._h: YieldTermStructure = h
        self._a: float = float(a)
        self._sigma: float = float(sigma)
        f0 = h.forward_rate(0.0, 0.0, Compounding.Continuous, Frequency.NoFrequency).rate()
        self._process: OrnsteinUhlenbeckProcess = OrnsteinUhlenbeckProcess(a, sigma, f0)
        qassert.require(self._a >= 0.0, "negative a given")
        qassert.require(self._sigma >= 0.0, "negative sigma given")

    # --- inspectors ------------------------------------------------------

    def a(self) -> float:
        # C++ parity: hullwhiteprocess.cpp:74-76.
        return self._a

    def sigma(self) -> float:
        # C++ parity: hullwhiteprocess.cpp:78-80.
        return self._sigma

    def x0(self) -> float:
        # C++ parity: hullwhiteprocess.cpp:34-36.
        return self._process.x0()

    # --- closed-form helper ----------------------------------------------

    def alpha(self, t: float) -> float:
        """Deterministic shift ``alpha(t)``.

        # C++ parity: hullwhiteprocess.cpp:65-72. The ``a -> 0`` algebraic
        # limit substitutes ``sigma*t`` for ``(sigma/a)*(1-exp(-a t))``.
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

    # --- StochasticProcess1D overrides -----------------------------------

    def drift_1d(self, t: float, x: float) -> float:
        # C++ parity: hullwhiteprocess.cpp:38-46. Note there is no ``a -> 0``
        # guard here (unlike ``alpha``): C++ divides by ``a`` unconditionally.
        alpha_drift = (
            self._sigma * self._sigma / (2.0 * self._a) * (1.0 - math.exp(-2.0 * self._a * t))
        )
        shift = 0.0001
        f = self._h.forward_rate(t, t, Compounding.Continuous, Frequency.NoFrequency).rate()
        fup = self._h.forward_rate(
            t + shift, t + shift, Compounding.Continuous, Frequency.NoFrequency
        ).rate()
        f_prime = (fup - f) / shift
        alpha_drift += self._a * f + f_prime
        return self._process.drift_1d(t, x) + alpha_drift

    def diffusion_1d(self, t: float, x: float) -> float:
        # C++ parity: hullwhiteprocess.cpp:48-50.
        return self._process.diffusion_1d(t, x)

    def expectation_1d(self, t0: float, x0: float, dt: float) -> float:
        # C++ parity: hullwhiteprocess.cpp:52-55.
        return (
            self._process.expectation_1d(t0, x0, dt)
            + self.alpha(t0 + dt)
            - self.alpha(t0) * math.exp(-self._a * dt)
        )

    def std_deviation_1d(self, t0: float, x0: float, dt: float) -> float:
        # C++ parity: hullwhiteprocess.cpp:57-59.
        return self._process.std_deviation_1d(t0, x0, dt)

    def variance_1d(self, t0: float, x0: float, dt: float) -> float:
        # C++ parity: hullwhiteprocess.cpp:61-63.
        return self._process.variance_1d(t0, x0, dt)


__all__ = ["HullWhiteProcess"]
