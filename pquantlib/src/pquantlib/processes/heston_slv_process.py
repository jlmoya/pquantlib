"""HestonSLVProcess — Heston stochastic-LOCAL-volatility process.

# C++ parity: ql/processes/hestonslvprocess.{hpp,cpp} (v1.43) —
# ``class HestonSLVProcess : public StochasticProcess``
# (hestonslvprocess.hpp:33-82, hestonslvprocess.cpp:32-126).

A Heston process whose spot volatility is multiplied by a *leverage
function* ``L(t, S)`` — a ``LocalVolTermStructure`` — and whose vol-of-vol is
scaled by a *mixing factor*::

    dS = (r - q - 0.5 (sqrt(v) L)^2) S dt + sqrt(v) L S dW_1
    dv = kappa (theta - v) dt + (mixingFactor * sigma) sqrt(v) dW_2

The parameters are pulled off the wrapped ``HestonProcess`` in
``setParameters()`` and refreshed on every ``update()``, so relinking the
Heston process's inputs propagates.

Three details that are easy to lose in translation and are pinned by the
cross-validation:

* ``drift`` and ``diffusion`` floor the leveraged vol at ``1e-8``
  (``std::max(1e-8, sqrt(v) * L)``) but the vol-of-vol term ``sigma2`` is NOT
  floored — it is ``mixedSigma * sqrt(v)`` raw.
* ``drift``'s second component is ``kappa*(theta - x[1])`` — the RAW variance,
  not the truncated one that ``HestonProcess::drift`` uses.
* ``evolve`` is a Quadratic-Exponential scheme with a ``psi < 1.5`` switch;
  in the ``psi >= 1.5`` arm the variance is set to EXACTLY 0.0 whenever the
  uniform draw falls below ``p``.

Divergences from C++:

* # C++ parity divergence: C++ takes ``ext::shared_ptr<HestonProcess>`` and
  ``ext::shared_ptr<LocalVolTermStructure>``; this port threads the objects
  directly. The accessors that C++ returns as ``const Handle<...>&``
  (``s0``, ``dividendYield``, ``riskFreeRate``) return the objects.
* ``forwardRate(t, t, Continuous)`` in C++ leaves ``freq`` at its ``Annual``
  default; the Python call passes ``Frequency.Annual`` explicitly. With
  ``Continuous`` compounding the frequency is unused, so this is naming only.
* ``drift`` / ``diffusion`` take ``sqrt(x[1])`` without guarding the sign. For
  a NEGATIVE variance C++ returns NaN and carries on; ``math.sqrt`` raises
  ``ValueError`` instead. Neither language produces a usable number, and no
  caller in this port feeds a negative variance in, so the raise is left as
  the louder of the two failures rather than papered over with a NaN.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.processes.stochastic_process import StochasticProcess
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.volatility.equity_fx.local_vol_term_structure import (
    LocalVolTermStructure,
)
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency


class HestonSLVProcess(StochasticProcess):
    """Heston stochastic-local-volatility process.

    # C++ parity: ``class HestonSLVProcess : public StochasticProcess``.
    """

    def __init__(
        self,
        heston_process: HestonProcess,
        leverage_fct: LocalVolTermStructure,
        mixing_factor: float = 1.0,
    ) -> None:
        # C++ parity: hestonslvprocess.cpp:32-39.
        super().__init__()
        self._mixing_factor: float = float(mixing_factor)
        self._heston_process: HestonProcess = heston_process
        self._leverage_fct: LocalVolTermStructure = leverage_fct
        # Declared here so the attributes exist before set_parameters runs.
        self._kappa: float = 0.0
        self._theta: float = 0.0
        self._sigma: float = 0.0
        self._rho: float = 0.0
        self._v0: float = 0.0
        self._mixed_sigma: float = 0.0
        heston_process.register_with(self)
        self._set_parameters()

    def _set_parameters(self) -> None:
        """Copy the Heston parameters across and rebuild ``mixedSigma``.

        # C++ parity: ``HestonSLVProcess::setParameters``
        # (hestonslvprocess.cpp:119-126).
        """
        self._v0 = self._heston_process.v0
        self._kappa = self._heston_process.kappa
        self._theta = self._heston_process.theta
        self._sigma = self._heston_process.sigma
        self._rho = self._heston_process.rho
        self._mixed_sigma = self._mixing_factor * self._sigma

    def update(self) -> None:
        """Refresh the cached parameters, then propagate.

        # C++ parity: hestonslvprocess.cpp:41-44.
        """
        self._set_parameters()
        super().update()

    # --- inspectors --------------------------------------------------------

    def v0(self) -> float:
        # C++ parity: hestonslvprocess.hpp:55 (inline).
        return self._v0

    def rho(self) -> float:
        # C++ parity: hestonslvprocess.hpp:56 (inline).
        return self._rho

    def kappa(self) -> float:
        # C++ parity: hestonslvprocess.hpp:57 (inline).
        return self._kappa

    def theta(self) -> float:
        # C++ parity: hestonslvprocess.hpp:58 (inline).
        return self._theta

    def sigma(self) -> float:
        """The UNMIXED vol-of-vol.

        # C++ parity: hestonslvprocess.hpp:59 (inline) — returns ``sigma_``,
        # not ``mixedSigma_``.
        """
        return self._sigma

    def mixing_factor(self) -> float:
        # C++ parity: hestonslvprocess.hpp:60 (inline).
        return self._mixing_factor

    def leverage_fct(self) -> LocalVolTermStructure:
        # C++ parity: hestonslvprocess.hpp:61-63 (inline).
        return self._leverage_fct

    def s0(self) -> Quote:
        # C++ parity: hestonslvprocess.hpp:65 (inline).
        return self._heston_process.s0()

    def dividend_yield(self) -> YieldTermStructure:
        # C++ parity: hestonslvprocess.hpp:66-68 (inline).
        return self._heston_process.dividend_yield()

    def risk_free_rate(self) -> YieldTermStructure:
        # C++ parity: hestonslvprocess.hpp:69-71 (inline).
        return self._heston_process.risk_free_rate()

    # --- StochasticProcess interface --------------------------------------

    def size(self) -> int:
        # C++ parity: hestonslvprocess.hpp:39 (inline).
        return 2

    def factors(self) -> int:
        # C++ parity: hestonslvprocess.hpp:40 (inline).
        return 2

    def initial_values(self) -> npt.NDArray[np.float64]:
        # C++ parity: hestonslvprocess.hpp:44-46 (inline) — delegates.
        return self._heston_process.initial_values()

    def apply(
        self, x0: npt.NDArray[np.float64], dx: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        # C++ parity: hestonslvprocess.hpp:47-49 (inline) — delegates.
        return self._heston_process.apply(x0, dx)

    def drift(self, t: float, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Drift of (ln S, v).

        # C++ parity: hestonslvprocess.cpp:46-59. The leveraged vol is floored
        # at 1e-8; the variance drift uses the RAW ``x[1]``.
        """
        vol = max(
            1e-8,
            math.sqrt(float(x[1]))
            * self._leverage_fct.local_vol_at_time(t, float(x[0]), True),
        )
        r = self.risk_free_rate().forward_rate(
            t, t, Compounding.Continuous, Frequency.Annual
        ).rate()
        q = self.dividend_yield().forward_rate(
            t, t, Compounding.Continuous, Frequency.Annual
        ).rate()
        return np.array(
            [r - q - 0.5 * vol * vol, self._kappa * (self._theta - float(x[1]))],
            dtype=np.float64,
        )

    def diffusion(self, t: float, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Lower-triangular 2x2 diffusion.

        # C++ parity: hestonslvprocess.cpp:61-74. ``sigma2`` uses the MIXED
        # vol-of-vol and is deliberately NOT floored.
        """
        vol = max(
            1e-8,
            math.sqrt(float(x[1]))
            * self._leverage_fct.local_vol_at_time(t, float(x[0]), True),
        )
        sigma2 = self._mixed_sigma * math.sqrt(float(x[1]))
        sqrhov = math.sqrt(1.0 - self._rho * self._rho)
        return np.array(
            [[vol, 0.0], [self._rho * sigma2, sqrhov * sigma2]],
            dtype=np.float64,
        )

    def evolve(
        self,
        t0: float,
        x0: npt.NDArray[np.float64],
        dt: float,
        dw: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Quadratic-Exponential step for the variance, log-normal for the spot.

        # C++ parity: hestonslvprocess.cpp:76-117, transcribed term by term.
        """
        ret = np.empty(2, dtype=np.float64)

        ex = math.exp(-self._kappa * dt)

        v_start = float(x0[1])
        m = self._theta + (v_start - self._theta) * ex
        s2 = v_start * self._mixed_sigma * self._mixed_sigma * ex / self._kappa * (
            1.0 - ex
        ) + self._theta * self._mixed_sigma * self._mixed_sigma / (2.0 * self._kappa) * (
            1.0 - ex
        ) * (1.0 - ex)
        psi = s2 / (m * m)

        if psi < 1.5:
            b2 = 2.0 / psi - 1.0 + math.sqrt(2.0 / psi * (2.0 / psi - 1.0))
            b = math.sqrt(b2)
            a = m / (1.0 + b2)
            ret[1] = a * (b + float(dw[1])) * (b + float(dw[1]))
        else:
            p = (psi - 1.0) / (psi + 1.0)
            beta = (1.0 - p) / m
            u = CumulativeNormalDistribution()(float(dw[1]))
            ret[1] = 0.0 if u <= p else math.log((1.0 - p) / (1.0 - u)) / beta

        mu = (
            self.risk_free_rate()
            .forward_rate(t0, t0 + dt, Compounding.Continuous, Frequency.Annual)
            .rate()
            - self.dividend_yield()
            .forward_rate(t0, t0 + dt, Compounding.Continuous, Frequency.Annual)
            .rate()
        )

        rho1 = math.sqrt(1.0 - self._rho * self._rho)

        l_0 = self._leverage_fct.local_vol_at_time(t0, float(x0[0]), True)
        v_0 = 0.5 * (v_start + float(ret[1])) * l_0 * l_0

        ret[0] = float(x0[0]) * math.exp(
            mu * dt
            - 0.5 * v_0 * dt
            + self._rho
            / self._mixed_sigma
            * l_0
            * (
                float(ret[1])
                - self._kappa * self._theta * dt
                + 0.5 * (v_start + float(ret[1])) * self._kappa * dt
                - v_start
            )
            + rho1 * math.sqrt(v_0 * dt) * float(dw[0])
        )

        return ret

    def time(self, date: Date) -> float:
        # C++ parity: hestonslvprocess.hpp:73 (inline) — delegates.
        return self._heston_process.time(date)


__all__ = ["HestonSLVProcess"]
