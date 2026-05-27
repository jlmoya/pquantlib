"""HestonProcess — square-root stochastic-volatility (Heston) process.

# C++ parity: ql/processes/hestonprocess.{hpp,cpp} (v1.42.1).

The Heston process describes a spot S and its variance V jointly::

    dS(t) = (r - q) * S * dt + sqrt(V) * S * dW_1
    dV(t) = kappa * (theta - V) * dt + sigma * sqrt(V) * dW_2
    dW_1 dW_2 = rho * dt

Internally calculations on S are in log-space (`apply` does
``[x0[0]*exp(dx[0]), x0[1]+dx[1]]``).

L4-C scope (this module): the analytic-engine entry path. The
``Discretization`` enum and the exact-sampling / Bessel-function /
PDF machinery in the C++ source are deferred — none of the L4-C
calibration tests exercise them. The Python port keeps a single
``FullTruncation`` semantic for ``drift`` / ``diffusion``:

* If V > 0, vol = sqrt(V); else vol = 0.0 (full truncation).

That matches both ``FullTruncation`` and (in practice) ``Reflection`` /
``PartialTruncation`` on the analytic side — the discretization
choice only affects MC simulation paths, which are out of scope.

Divergences from C++:

* ``Handle<Quote>`` / ``Handle<YieldTermStructure>`` collapse to a
  direct reference (pquantlib convention from L2 / L3).
* The ``Discretization`` enum is dropped — only ``FullTruncation``
  semantics are implemented. If MC-based engines ever land, the enum
  can be re-introduced as a parameter on the L5 MC engine, not on the
  process.
* The internal exact-sampling characteristic function ``Phi`` and the
  ``pdf`` method are not ported — they require modified Bessel
  functions + Gauss-Laguerre quadrature + non-central chi-square
  inversion. None of the calibration paths in L4-C need them.

The diffusion matrix is the (lower-triangular) Cholesky factor of the
correlation matrix, so that ``diffusion * dW`` produces correlated
increments::

    diffusion = [ sqrt(V),                 0                          ]
                [ rho * sigma * sqrt(V),   sqrt(1-rho^2) * sigma * sqrt(V) ]

When V == 0 the C++ source plants a tiny ``1e-8`` in the diffusion to
preserve some correlation information; we mirror that.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from pquantlib.processes.euler_discretization import EulerDiscretization
from pquantlib.processes.stochastic_process import StochasticProcess
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency


class HestonProcess(StochasticProcess):
    """Square-root stochastic-volatility process.

    # C++ parity: ``class HestonProcess : public StochasticProcess``.
    """

    __slots__ = (
        "_dividend_yield",
        "_kappa",
        "_rho",
        "_risk_free_rate",
        "_s0",
        "_sigma",
        "_theta",
        "_v0",
    )

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
    ) -> None:
        super().__init__(EulerDiscretization())
        self._risk_free_rate: YieldTermStructure = risk_free_rate
        self._dividend_yield: YieldTermStructure = dividend_yield
        self._s0: Quote = s0
        self._v0: float = v0
        self._kappa: float = kappa
        self._theta: float = theta
        self._sigma: float = sigma
        self._rho: float = rho
        # C++ parity: hestonprocess.cpp:51-53 — register with all
        # observables so process notifies its own observers on change.
        risk_free_rate.register_with(self)
        dividend_yield.register_with(self)
        s0.register_with(self)

    # --- inspectors -----------------------------------------------------

    @property
    def v0(self) -> float:
        """Initial variance.

        # C++ parity: ``HestonProcess::v0`` in hestonprocess.hpp:77.
        """
        return self._v0

    @property
    def kappa(self) -> float:
        """Mean-reversion speed of variance.

        # C++ parity: ``HestonProcess::kappa`` in hestonprocess.hpp:79.
        """
        return self._kappa

    @property
    def theta(self) -> float:
        """Long-term variance level.

        # C++ parity: ``HestonProcess::theta`` in hestonprocess.hpp:80.
        """
        return self._theta

    @property
    def sigma(self) -> float:
        """Volatility of variance.

        # C++ parity: ``HestonProcess::sigma`` in hestonprocess.hpp:81.
        """
        return self._sigma

    @property
    def rho(self) -> float:
        """Correlation between spot and variance Brownians.

        # C++ parity: ``HestonProcess::rho`` in hestonprocess.hpp:78.
        """
        return self._rho

    def s0(self) -> Quote:
        """Spot quote.

        # C++ parity: ``HestonProcess::s0`` in hestonprocess.hpp:83.
        """
        return self._s0

    def risk_free_rate(self) -> YieldTermStructure:
        """Risk-free yield curve.

        # C++ parity: ``HestonProcess::riskFreeRate`` in hestonprocess.hpp:85.
        """
        return self._risk_free_rate

    def dividend_yield(self) -> YieldTermStructure:
        """Dividend-yield curve.

        # C++ parity: ``HestonProcess::dividendYield`` in hestonprocess.hpp:84.
        """
        return self._dividend_yield

    # --- StochasticProcess overrides ------------------------------------

    def size(self) -> int:
        """State dimension = 2 (spot + variance).

        # C++ parity: ``HestonProcess::size`` in hestonprocess.cpp:56-58.
        """
        return 2

    def factors(self) -> int:
        """Independent Brownian factors = 2 for the analytic-engine path.

        # C++ parity: ``HestonProcess::factors`` in hestonprocess.cpp:60-64.
        # Returns 3 only for ``BroadieKaya*`` exact-sampling discretizations,
        # which are out of scope for L4-C. The two-Brownian default matches
        # ``FullTruncation`` / ``Reflection`` / ``PartialTruncation``.
        """
        return 2

    def initial_values(self) -> npt.NDArray[np.float64]:
        """Initial state = (S(0), V(0)).

        # C++ parity: ``HestonProcess::initialValues`` in hestonprocess.cpp:66-68.
        """
        return np.array([self._s0.value(), self._v0], dtype=np.float64)

    def drift(self, t: float, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Heston drift vector.

        # C++ parity: ``HestonProcess::drift`` in hestonprocess.cpp:70-81.

        Uses full-truncation semantics: ``vol = sqrt(max(V, 0))``.
        """
        v = float(x[1])
        vol = math.sqrt(v) if v > 0.0 else 0.0
        # C++ uses ``forwardRate(t, t, Continuous)`` — the instantaneous
        # forward rate at ``t``. Python's ``forward_rate(t, t, ...)`` has
        # a year-fraction bug at the t1==t2 branch (passes 0.0 to
        # implied_rate, which rejects non-positive times). Sidestep by
        # passing an explicit small finite window — matches the L3-D
        # ``GeneralizedBlackScholesProcess`` workaround.
        dt = 0.0001
        r = self._risk_free_rate.forward_rate(
            t, t + dt, Compounding.Continuous, Frequency.NoFrequency, True
        ).rate()
        q = self._dividend_yield.forward_rate(
            t, t + dt, Compounding.Continuous, Frequency.NoFrequency, True
        ).rate()
        return np.array(
            [
                r - q - 0.5 * vol * vol,
                self._kappa * (self._theta - vol * vol),
            ],
            dtype=np.float64,
        )

    def diffusion(self, t: float, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Heston diffusion matrix (lower-triangular Cholesky of corr matrix).

        # C++ parity: ``HestonProcess::diffusion`` in hestonprocess.cpp:83-102.

        For the analytic-engine path we use the FullTruncation
        semantics (``vol = sqrt(max(V, 0))``); when V <= 0 we still
        plant a tiny ``1e-8`` vol so the diffusion matrix retains
        correlation structure (matches the C++ branch that "expose
        some correlation information" even at near-zero variance).
        """
        del t  # diffusion is time-homogeneous; arg present for API symmetry
        v = float(x[1])
        vol = math.sqrt(v) if v > 0.0 else 1e-8
        sigma_vol = self._sigma * vol
        sq_rho = math.sqrt(1.0 - self._rho * self._rho)
        return np.array(
            [
                [vol, 0.0],
                [self._rho * sigma_vol, sq_rho * sigma_vol],
            ],
            dtype=np.float64,
        )

    def apply(
        self,
        x0: npt.NDArray[np.float64],
        dx: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Apply an increment: ``S = S0 * exp(dx_S)``, ``V = V0 + dx_V``.

        # C++ parity: ``HestonProcess::apply`` in hestonprocess.cpp:104-110.
        """
        return np.array(
            [
                float(x0[0]) * math.exp(float(dx[0])),
                float(x0[1]) + float(dx[1]),
            ],
            dtype=np.float64,
        )

    def time(self, date: Date) -> float:
        """Year fraction via the risk-free curve's day counter.

        # C++ parity: ``HestonProcess::time`` — uses the risk-free curve's
        # day counter for date → time conversion.
        """
        return self._risk_free_rate.day_counter().year_fraction(
            self._risk_free_rate.reference_date(), date
        )


__all__ = ["HestonProcess"]
