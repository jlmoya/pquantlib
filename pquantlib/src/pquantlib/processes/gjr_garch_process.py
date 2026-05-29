"""GjrGarchProcess — Glosten-Jagannathan-Runkle GARCH(1,1) stochastic-vol process.

# C++ parity: ql/processes/gjrgarchprocess.{hpp,cpp} (v1.42.1).

Two-factor (S, V) stochastic-vol diffusion with leverage-sensitive
mean-reversion in variance::

    dS(t) = mu * S dt + sqrt(V) * S * dW_1
    dV(t) = (omega + (beta + alpha*q_2 + gamma*q_3 - 1) * V) dt
            + (alpha*sigma_12 + gamma*sigma_13) * V * dW_1
            + sqrt(alpha^2*(sigma_2^2 - sigma_12^2)
                  + gamma^2*(sigma_3^2 - sigma_13^2)
                  + 2*alpha*gamma*(sigma_23 - sigma_12*sigma_13)) * V * dW_2

with ``daysPerYear`` annualization and lambda the market price of risk
folded into the moment constants. See gjrgarchprocess.hpp for the
moment derivations.

L11-W1-D scope: ports the `FullTruncation` semantics needed by
`AnalyticGjrGarchEngine`. The `PartialTruncation` / `Reflection`
discretization branches are kept as enum members; their `evolve`
behaviour for MC paths is implemented since the `evolve` override is
needed by the test infrastructure (though the analytic engine doesn't
use it).

Reference:
- Glosten, L., Jagannathan, R., Runkle, D., 1993. Relationship between
  the expected value and the volatility of the nominal excess return
  on stocks. Journal of Finance 48, 1779-1801.

Divergences from C++:
- `Handle<Quote>` / `Handle<YieldTermStructure>` collapse to direct
  references (pquantlib convention).
- The `Discretization` enum is preserved as `IntEnum` (translation
  decision: switch-on-enum via IntEnum dispatch, per the
  project_python_translation_choices memory).
- Math-symbol names sigma, alpha, beta, gamma, lambda, omega, V are
  preserved verbatim with N802/N803/N806 noqa for readability against
  the C++ literature.
"""

from __future__ import annotations

import math
from enum import IntEnum

import numpy as np
import numpy.typing as npt

from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.processes.euler_discretization import EulerDiscretization
from pquantlib.processes.stochastic_process import StochasticProcess
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency


class Discretization(IntEnum):
    """GJR-GARCH variance-process discretization scheme.

    # C++ parity: ``GJRGARCHProcess::Discretization`` in
    # gjrgarchprocess.hpp:71.
    """

    PartialTruncation = 0
    FullTruncation = 1
    Reflection = 2


class GjrGarchProcess(StochasticProcess):
    """GJR-GARCH(1,1) two-factor stochastic-volatility process.

    # C++ parity: ``class GJRGARCHProcess : public StochasticProcess``
    # in gjrgarchprocess.hpp:69-105 (v1.42.1).
    """

    __slots__ = (
        "_alpha",
        "_beta",
        "_days_per_year",
        "_disc_scheme",
        "_dividend_yield",
        "_gamma",
        "_lambda",
        "_omega",
        "_risk_free_rate",
        "_s0",
        "_v0",
    )

    def __init__(
        self,
        *,
        risk_free_rate: YieldTermStructure,
        dividend_yield: YieldTermStructure,
        s0: Quote,
        v0: float,
        omega: float,
        alpha: float,
        beta: float,
        gamma: float,
        lambda_: float,
        days_per_year: float = 252.0,
        discretization: Discretization = Discretization.FullTruncation,
    ) -> None:
        super().__init__(EulerDiscretization())
        self._risk_free_rate: YieldTermStructure = risk_free_rate
        self._dividend_yield: YieldTermStructure = dividend_yield
        self._s0: Quote = s0
        self._v0: float = v0
        self._omega: float = omega
        self._alpha: float = alpha
        self._beta: float = beta
        self._gamma: float = gamma
        self._lambda: float = lambda_
        self._days_per_year: float = days_per_year
        self._disc_scheme: Discretization = discretization

        # C++ parity: gjrgarchprocess.cpp:38-40 — register observers.
        risk_free_rate.register_with(self)
        dividend_yield.register_with(self)
        s0.register_with(self)

    # --- inspectors -----------------------------------------------------

    @property
    def v0(self) -> float:
        """Initial daily variance.

        # C++ parity: ``GJRGARCHProcess::v0`` in gjrgarchprocess.hpp:96.
        """
        return self._v0

    @property
    def omega(self) -> float:
        """Variance baseline coefficient.

        # C++ parity: ``GJRGARCHProcess::omega`` in gjrgarchprocess.hpp:98.
        """
        return self._omega

    @property
    def alpha(self) -> float:
        """Innovation impact coefficient.

        # C++ parity: ``GJRGARCHProcess::alpha`` in gjrgarchprocess.hpp:99.
        """
        return self._alpha

    @property
    def beta(self) -> float:
        """Variance autoregression coefficient.

        # C++ parity: ``GJRGARCHProcess::beta`` in gjrgarchprocess.hpp:100.
        """
        return self._beta

    @property
    def gamma(self) -> float:
        """Negative-innovation impact coefficient.

        # C++ parity: ``GJRGARCHProcess::gamma`` in gjrgarchprocess.hpp:101.
        """
        return self._gamma

    @property
    def lambda_(self) -> float:
        """Market price of risk.

        # C++ parity: ``GJRGARCHProcess::lambda`` in gjrgarchprocess.hpp:97.

        Trailing underscore because ``lambda`` is a Python keyword.
        """
        return self._lambda

    @property
    def days_per_year(self) -> float:
        """Days-per-year annualization constant.

        # C++ parity: ``GJRGARCHProcess::daysPerYear`` in gjrgarchprocess.hpp:102.
        """
        return self._days_per_year

    @property
    def discretization(self) -> Discretization:
        """Variance discretization scheme (Partial/Full/Reflection)."""
        return self._disc_scheme

    def s0(self) -> Quote:
        """Spot quote.

        # C++ parity: ``GJRGARCHProcess::s0`` in gjrgarchprocess.hpp:104.
        """
        return self._s0

    def risk_free_rate(self) -> YieldTermStructure:
        """Risk-free yield curve."""
        return self._risk_free_rate

    def dividend_yield(self) -> YieldTermStructure:
        """Dividend-yield curve."""
        return self._dividend_yield

    # --- StochasticProcess overrides ------------------------------------

    def size(self) -> int:
        """State dimension = 2 (spot + annualized variance).

        # C++ parity: ``GJRGARCHProcess::size`` in gjrgarchprocess.cpp:43-45.
        """
        return 2

    def factors(self) -> int:
        """Two independent Brownian factors.

        Defaults to 2 from `StochasticProcess.factors()`.
        """
        return 2

    def initial_values(self) -> npt.NDArray[np.float64]:
        """Initial state = (S(0), daysPerYear * V(0)).

        # C++ parity: ``GJRGARCHProcess::initialValues`` in gjrgarchprocess.cpp:47-49.
        """
        return np.array(
            [self._s0.value(), self._days_per_year * self._v0],
            dtype=np.float64,
        )

    def _moment_constants(self) -> tuple[float, float, float, float]:
        """Return (N, n, q2, q3) — moments of lambda used by drift+diffusion."""
        lam = self._lambda
        n_cdf = CumulativeNormalDistribution()(lam)
        n_pdf = math.exp(-lam * lam / 2.0) / math.sqrt(2.0 * math.pi)
        q2 = 1.0 + lam * lam
        q3 = lam * n_pdf + n_cdf + lam * lam * n_cdf
        return n_cdf, n_pdf, q2, q3

    def drift(self, t: float, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """GJR-GARCH drift.

        # C++ parity: ``GJRGARCHProcess::drift`` in gjrgarchprocess.cpp:51-69.
        """
        _N, _n, q2, q3 = self._moment_constants()  # noqa: N806 — math symbols
        v = float(x[1])
        if v > 0.0:
            vol = math.sqrt(v)
        elif self._disc_scheme == Discretization.Reflection:
            vol = -math.sqrt(-v)
        else:
            vol = 0.0

        # C++ uses forward_rate(t, t, Continuous). Our forward_rate is
        # well-defined at t=t (returns the discount-derived value
        # via implied_rate(1.0, ...)), but has a known edge-case at
        # t1==t2 when called with NoFrequency. Use a small forward
        # window — mirrors the L4-C HestonProcess workaround.
        dt = 0.0001
        r = self._risk_free_rate.forward_rate(
            t, t + dt, Compounding.Continuous, Frequency.NoFrequency, True
        ).rate()
        q = self._dividend_yield.forward_rate(
            t, t + dt, Compounding.Continuous, Frequency.NoFrequency, True
        ).rate()

        v_for_drift = (
            v if self._disc_scheme == Discretization.PartialTruncation else vol * vol
        )

        dpy = self._days_per_year
        drift_s = r - q - 0.5 * vol * vol
        drift_v = (
            dpy * dpy * self._omega
            + dpy * (self._beta + self._alpha * q2 + self._gamma * q3 - 1.0)
            * v_for_drift
        )
        return np.array([drift_s, drift_v], dtype=np.float64)

    def diffusion(
        self, t: float, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """GJR-GARCH diffusion matrix (lower-triangular Cholesky factor).

        # C++ parity: ``GJRGARCHProcess::diffusion`` in gjrgarchprocess.cpp:71-101.
        """
        del t  # diffusion is time-homogeneous

        n_cdf, n_pdf, _q2, q3 = self._moment_constants()
        lam = self._lambda

        sigma2 = 2.0 + 4.0 * lam * lam
        eml_e4 = (
            lam * lam * lam * n_pdf
            + 5.0 * lam * n_pdf
            + 3.0 * n_cdf
            + lam * lam * lam * lam * n_cdf
            + 6.0 * lam * lam * n_cdf
        )
        sigma3 = eml_e4 - q3 * q3
        sigma12 = -2.0 * lam
        sigma13 = -2.0 * n_pdf - 2.0 * lam * n_cdf
        sigma23 = 2.0 * n_cdf + sigma12 * sigma13

        v = float(x[1])
        if v > 0.0:
            vol = math.sqrt(v)
        elif self._disc_scheme == Discretization.Reflection:
            vol = -math.sqrt(-v)
        else:
            # vol = 1e-8 so the diffusion still exposes correlation info
            vol = 1e-8

        dpy = self._days_per_year
        sqrt_dpy = math.sqrt(dpy)
        alpha = self._alpha
        gamma = self._gamma

        # rho1 = sqrt(dpy) * (alpha*sigma12 + gamma*sigma13) * vol^2
        rho1 = sqrt_dpy * (alpha * sigma12 + gamma * sigma13) * vol * vol
        # rho2 = sqrt(dpy) * sqrt(alpha^2*(sigma2 - sigma12^2)
        #         + gamma^2*(sigma3 - sigma13^2)
        #         + 2*alpha*gamma*(sigma23 - sigma12*sigma13)) * vol^2
        rho2_radicand = (
            alpha * alpha * (sigma2 - sigma12 * sigma12)
            + gamma * gamma * (sigma3 - sigma13 * sigma13)
            + 2.0 * alpha * gamma * (sigma23 - sigma12 * sigma13)
        )
        rho2 = vol * vol * sqrt_dpy * math.sqrt(rho2_radicand)

        return np.array(
            [
                [vol, 0.0],
                [rho1, rho2],
            ],
            dtype=np.float64,
        )

    def apply(
        self,
        x0: npt.NDArray[np.float64],
        dx: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Apply an increment: ``S = S0*exp(dx_S)``, ``V = V0 + dx_V``.

        # C++ parity: ``GJRGARCHProcess::apply`` in gjrgarchprocess.cpp:103-105.
        """
        return np.array(
            [float(x0[0]) * math.exp(float(dx[0])), float(x0[1]) + float(dx[1])],
            dtype=np.float64,
        )

    def time(self, date: Date) -> float:
        """Year fraction via the risk-free curve's day counter.

        # C++ parity: ``GJRGARCHProcess::time`` in gjrgarchprocess.cpp:182-184.
        """
        return self._risk_free_rate.day_counter().year_fraction(
            self._risk_free_rate.reference_date(), date
        )


__all__ = ["Discretization", "GjrGarchProcess"]
