"""BatesProcess — Heston stochastic volatility + Merton log-normal jumps.

# C++ parity: ql/processes/batesprocess.{hpp,cpp} (v1.42.1).

The Bates process extends the Heston square-root vol process with a
compound-Poisson log-normal jump-diffusion in the spot::

    dS(t)   = (r - q - lambda * m) * S * dt + sqrt(V) * S * dW_1
              + (exp(J) - 1) * S * dN
    dV(t)   = kappa * (theta - V) * dt + sigma * sqrt(V) * dW_2
    dW_1 dW_2 = rho dt
    omega(J) = (1 / sqrt(2 * pi * delta^2)) * exp(-(J - nu)^2 / (2 * delta^2))

with ``m = exp(nu + 0.5 * delta^2) - 1`` the expected jump multiplier
under the lognormal measure. The drift on S is corrected by
``-lambda * m`` so the discounted spot remains a martingale.

L4-C scope (this module): the parameter accessors + the drift
correction. The ``evolve`` MC-time-step (which needs an
``InverseCumulativePoisson`` sampler) is deferred — none of the L4-C
analytic-engine calibration paths exercise it.

Divergences from C++:

* ``Handle<Quote>`` / ``Handle<YieldTermStructure>`` collapse to direct
  references.
* The C++ ``HestonProcess::Discretization`` enum is not threaded
  through (pquantlib's HestonProcess drops it; see hestonprocess.py).
* ``evolve`` is not overridden — calls fall through to the base
  StochasticProcess.evolve which uses the Euler discretization. The
  L5 MC engine port can refine if needed.
* The C++ ``CumulativeNormalDistribution`` member ``cumNormalDist_``
  used by ``evolve`` is omitted (no evolve override).
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from pquantlib.processes.heston_process import HestonProcess
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.yield_term_structure import YieldTermStructure


class BatesProcess(HestonProcess):
    """Heston process + Merton compound-Poisson log-normal jumps.

    # C++ parity: ``class BatesProcess : public HestonProcess`` in
    # ql/processes/batesprocess.hpp:49-70 (v1.42.1).
    """

    __slots__ = ("_delta", "_lambda", "_m", "_nu")

    def __init__(
        self,
        *,
        risk_free_rate: YieldTermStructure,
        dividend_yield: YieldTermStructure,
        s0: Quote,
        v0: float,
        kappa: float,
        theta: float,
        sigma: float,
        rho: float,
        lambda_: float,
        nu: float,
        delta: float,
    ) -> None:
        # Python convention: ``lambda_`` keyword because ``lambda`` is
        # reserved. The C++ field is ``lambda_``; same.
        super().__init__(
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            s0=s0,
            v0=v0,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            rho=rho,
        )
        self._lambda: float = lambda_
        self._nu: float = nu
        self._delta: float = delta
        # C++ parity: batesprocess.cpp:37 — m_ = exp(nu + 0.5*delta^2) - 1.
        self._m: float = math.exp(nu + 0.5 * delta * delta) - 1.0

    # --- jump-parameter accessors ---------------------------------------

    @property
    def lambda_(self) -> float:
        """Poisson jump intensity (mean number of jumps per year).

        # C++ parity: ``BatesProcess::lambda`` in batesprocess.cpp:69-71.

        Python convention: trailing underscore because ``lambda`` is
        a reserved keyword. Callers may also use ``getattr(p, 'lambda_')``.
        """
        return self._lambda

    @property
    def nu(self) -> float:
        """Mean log-jump size.

        # C++ parity: ``BatesProcess::nu`` in batesprocess.cpp:73-75.
        """
        return self._nu

    @property
    def delta(self) -> float:
        """Jump-size standard deviation (lognormal scale).

        # C++ parity: ``BatesProcess::delta`` in batesprocess.cpp:77-79.
        """
        return self._delta

    @property
    def m(self) -> float:
        """Expected jump multiplier: ``exp(nu + 0.5*delta^2) - 1``.

        # C++ parity: ``BatesProcess::m_`` in batesprocess.hpp:68.

        Pre-computed at construction; used by ``drift`` for the
        martingale correction ``-lambda * m`` on the spot drift.
        """
        return self._m

    # --- HestonProcess / StochasticProcess overrides --------------------

    def factors(self) -> int:
        """Number of independent Brownian factors = HestonFactors + 2 = 4.

        # C++ parity: ``BatesProcess::factors`` in batesprocess.cpp:65-67.

        Two extra factors drive the Poisson arrival + the log-normal
        jump size (used by the MC evolve, which is deferred).
        """
        return super().factors() + 2

    def drift(self, t: float, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Heston drift with the martingale jump correction ``-lambda * m``.

        # C++ parity: ``BatesProcess::drift`` in batesprocess.cpp:40-44.
        """
        drift_heston = super().drift(t, x)
        drift_heston[0] -= self._lambda * self._m
        return drift_heston


__all__ = ["BatesProcess"]
