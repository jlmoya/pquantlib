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
* ``drift`` / ``diffusion`` keep a single full-truncation semantic,
  because the analytic-engine path never varies it.

* ``evolve`` DOES depend on the discretization, and is overridden here
  rather than inherited from ``StochasticProcess`` — the generic
  ``apply(expectation(...), stdDeviation(...)*dw)`` is not what C++ does
  for a Heston path and does not reproduce ``MCEuropeanHestonEngine``.
  Five of the nine C++ schemes are ported: ``PartialTruncation``,
  ``FullTruncation``, ``Reflection``, ``QuadraticExponential`` and
  ``QuadraticExponentialMartingale`` (the C++ default). The three
  ``BroadieKayaExactScheme*`` members and ``NonCentralChiSquareVariance``
  are deliberately absent from the enum rather than present-and-broken:
  they need the exact-sampling characteristic function / non-central
  chi-square inversion that this module already documents as unported,
  and ``factors()`` would have to return 3 for the Broadie-Kaya family.
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

import cmath
import math
from enum import IntEnum

import numpy as np
import numpy.typing as npt
from scipy.special import iv  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
from scipy.stats import ncx2  # pyright: ignore[reportMissingTypeStubs]

from pquantlib import qassert
from pquantlib.math.constants import QL_EPSILON
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.distributions.inverse_cumulative_normal import (
    InverseCumulativeNormal,
)
from pquantlib.math.integrals.gaussian_quadrature import GaussLaguerreIntegration
from pquantlib.math.integrals.segment import SegmentIntegral
from pquantlib.processes.euler_discretization import EulerDiscretization
from pquantlib.processes.stochastic_process import StochasticProcess
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency


class Discretization(IntEnum):
    """Path-discretization scheme for :meth:`HestonProcess.evolve`.

    # C++ parity: ``HestonProcess::Discretization`` (hestonprocess.hpp:56-66).

    The C++ enumerator order is ``PartialTruncation, FullTruncation,
    Reflection, NonCentralChiSquareVariance, QuadraticExponential,
    QuadraticExponentialMartingale, BroadieKayaExactSchemeLobatto,
    BroadieKayaExactSchemeLaguerre, BroadieKayaExactSchemeTrapezoidal``; the
    integer values below preserve it so a round-trip through an int is stable
    even though the unported members are absent.

    ``QuadraticExponentialMartingale`` is the C++ constructor default.
    """

    PartialTruncation = 0
    FullTruncation = 1
    Reflection = 2
    QuadraticExponential = 4
    QuadraticExponentialMartingale = 5


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
        discretization: Discretization = Discretization.QuadraticExponentialMartingale,
    ) -> None:
        super().__init__(EulerDiscretization())
        # NOTE the name: ``StochasticProcess`` already owns ``_discretization``
        # (the Euler/other *object* used by ``expectation`` / ``std_deviation``).
        # Storing the scheme enum under that name would silently replace it.
        self._disc_scheme: Discretization = discretization
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
        # ALIGN(processes): C++ HestonProcess::drift (hestonprocess.cpp:70-81)
        # uses the INSTANTANEOUS forward, forwardRate(t, t, Continuous). The
        # 1e-4 window this used to pass belongs to
        # GeneralizedBlackScholesProcess::drift, a different formula, and the
        # premise justifying it ("forward_rate(t, t, ...) has a year-fraction
        # bug") is stale — YieldTermStructure.forward_rate centres a _DT window
        # on t exactly as yieldtermstructure.cpp:169-171 does. On a flat curve
        # the two agree, which is why it survived; on the non-flat curve pinned
        # by references/v143/processes/tail.json they differ by ~3e-7.
        r = self._risk_free_rate.forward_rate(
            t, t, Compounding.Continuous, Frequency.Annual
        ).rate()
        q = self._dividend_yield.forward_rate(
            t, t, Compounding.Continuous, Frequency.Annual
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

    def discretization(self) -> Discretization:
        """The path-discretization scheme in force.

        # C++ parity: the ``discretization_`` member (hestonprocess.hpp:107).
        """
        return self._disc_scheme

    def evolve(  # noqa: PLR0915 — verbatim transcription of a C++ switch
        self,
        t0: float,
        x0: npt.NDArray[np.float64],
        dt: float,
        dw: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """One discretized step of (S, V).

        # C++ parity: ``HestonProcess::evolve`` (hestonprocess.cpp:396-...).

        This override matters: the inherited
        ``apply(expectation(t0, x0, dt), stdDeviation(t0, x0, dt) * dw)`` is a
        plain Euler step and is *not* what C++ simulates, so an MC Heston
        engine built on the inherited version cannot reproduce C++ at any
        sample count.
        """
        v0 = float(x0[1])
        s0 = float(x0[0])
        dw0 = float(dw[0])
        dw1 = float(dw[1])
        sdt = math.sqrt(dt)
        sqrhov = math.sqrt(1.0 - self._rho * self._rho)
        scheme = self._disc_scheme

        if scheme in (
            Discretization.PartialTruncation,
            Discretization.FullTruncation,
            Discretization.Reflection,
        ):
            if scheme == Discretization.Reflection:
                vol = math.sqrt(abs(v0))
            else:
                vol = math.sqrt(v0) if v0 > 0.0 else 0.0
            vol2 = self._sigma * vol
            mu = self._forward_rate_spread(t0, dt) - 0.5 * vol * vol
            if scheme == Discretization.PartialTruncation:
                nu = self._kappa * (self._theta - v0)
            else:
                nu = self._kappa * (self._theta - vol * vol)
            s1 = s0 * math.exp(mu * dt + vol * dw0 * sdt)
            base = vol * vol if scheme == Discretization.Reflection else v0
            v1 = base + nu * dt + vol2 * sdt * (self._rho * dw0 + sqrhov * dw1)
            return np.array([s1, v1], dtype=np.float64)

        # QuadraticExponential[Martingale]: Leif Andersen, "Efficient Simulation
        # of the Heston Stochastic Volatility Model".
        sigma = self._sigma
        kappa = self._kappa
        theta = self._theta
        rho = self._rho
        ex = math.exp(-kappa * dt)
        m = theta + (v0 - theta) * ex
        s2 = v0 * sigma * sigma * ex / kappa * (1 - ex) + theta * sigma * sigma / (
            2 * kappa
        ) * (1 - ex) * (1 - ex)
        psi = s2 / (m * m)

        g1 = 0.5
        g2 = 0.5
        k0 = -rho * kappa * theta * dt / sigma
        k1 = g1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
        k2 = g2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
        k3 = g1 * dt * (1 - rho * rho)
        k4 = g2 * dt * (1 - rho * rho)
        a_coef = k2 + 0.5 * k4
        martingale = scheme == Discretization.QuadraticExponentialMartingale

        if psi < 1.5:
            b2 = 2 / psi - 1 + math.sqrt(2 / psi * (2 / psi - 1))
            b = math.sqrt(b2)
            a = m / (1 + b2)
            if martingale:
                qassert.require(a_coef < 1 / (2 * a), "illegal value")
                k0 = (
                    -a_coef * b2 * a / (1 - 2 * a_coef * a)
                    + 0.5 * math.log(1 - 2 * a_coef * a)
                    - (k1 + 0.5 * k3) * v0
                )
            v1 = a * (b + dw1) * (b + dw1)
        else:
            p = (psi - 1) / (psi + 1)
            beta = (1 - p) / m
            u = CumulativeNormalDistribution()(dw1)
            if martingale:
                qassert.require(a_coef < beta, "illegal value")
                k0 = -math.log(p + beta * (1 - p) / (beta - a_coef)) - (
                    k1 + 0.5 * k3
                ) * v0
            v1 = 0.0 if u <= p else math.log((1 - p) / (1 - u)) / beta

        mu = self._forward_rate_spread(t0, dt)
        s1 = s0 * math.exp(
            mu * dt + k0 + k1 * v0 + k2 * v1 + math.sqrt(k3 * v0 + k4 * v1) * dw0
        )
        return np.array([s1, v1], dtype=np.float64)

    def _forward_rate_spread(self, t0: float, dt: float) -> float:
        """``r(t0, t0+dt) - q(t0, t0+dt)``, continuously compounded.

        # C++ parity: the ``riskFreeRate_->forwardRate(t0, t0+dt, Continuous)
        # - dividendYield_->forwardRate(t0, t0+dt, Continuous)`` pair that
        # every branch of ``HestonProcess::evolve`` computes. Unlike
        # :meth:`drift`, this uses the *actual* step, so no zero-window
        # workaround is needed and the result is bit-comparable with C++.
        """
        r = self._risk_free_rate.forward_rate(
            t0, t0 + dt, Compounding.Continuous, Frequency.NoFrequency, True
        ).rate()
        q = self._dividend_yield.forward_rate(
            t0, t0 + dt, Compounding.Continuous, Frequency.NoFrequency, True
        ).rate()
        return r - q

    def time(self, date: Date) -> float:
        """Year fraction via the risk-free curve's day counter.

        # C++ parity: ``HestonProcess::time`` — uses the risk-free curve's
        # day counter for date → time conversion.
        """
        return self._risk_free_rate.day_counter().year_fraction(
            self._risk_free_rate.reference_date(), date
        )

    # --- Fokker-Planck density, from the other side of this merge ---
    # C++ HestonProcess::pdf and its characteristic-function helpers.
    # FdmHestonGreensFct consumes pdf(); the Discretization-bearing side
    # did not carry them. Neither side had both, so both are grafted here.

    def _phi(self, a: complex, nu_0: float, nu_t: float, dt: float) -> complex:
        """# C++ parity: the anonymous-namespace ``Phi``."""
        theta = self._theta
        kappa = self._kappa
        sigma = self._sigma
        sigma2 = sigma * sigma

        ga = cmath.sqrt(kappa * kappa - 2.0 * sigma2 * a * 1j)
        d = 4.0 * theta * kappa / sigma2
        nu = 0.5 * d - 1.0

        z = ga * cmath.exp(-0.5 * ga * dt) / (1.0 - cmath.exp(-ga * dt))
        log_z = -0.5 * ga * dt + cmath.log(ga / (1.0 - cmath.exp(-ga * dt)))

        alpha = 4.0 * ga * cmath.exp(-0.5 * ga * dt) / (sigma2 * (1.0 - cmath.exp(-ga * dt)))
        beta = (
            4.0
            * kappa
            * math.exp(-0.5 * kappa * dt)
            / (sigma2 * (1.0 - math.exp(-kappa * dt)))
        )

        if nu_t > 1e-8:
            root = math.sqrt(nu_0 * nu_t)
            bessel_ratio = complex(iv(nu, root * alpha)) / complex(iv(nu, root * beta))
        else:
            bessel_ratio = (alpha / beta) ** nu

        return (
            ga
            * cmath.exp(-0.5 * (ga - kappa) * dt)
            * (1.0 - math.exp(-kappa * dt))
            / (kappa * (1.0 - cmath.exp(-ga * dt)))
            * cmath.exp(
                (nu_0 + nu_t)
                / sigma2
                * (
                    kappa * (1.0 + math.exp(-kappa * dt)) / (1.0 - math.exp(-kappa * dt))
                    - ga * (1.0 + cmath.exp(-ga * dt)) / (1.0 - cmath.exp(-ga * dt))
                )
            )
            * cmath.exp(nu * log_z)
            / z**nu
            * bessel_ratio
        )

    def _ph(self, x: float, u: float, nu_0: float, nu_t: float, dt: float) -> float:
        """# C++ parity: the anonymous-namespace ``ph``."""
        return 2.0 / math.pi * math.cos(u * x) * self._phi(complex(u, 0.0), nu_0, nu_t, dt).real

    def _int_ph(
        self, a: float, x: float, y: float, nu_0: float, nu_t: float, t: float
    ) -> float:
        """# C++ parity: the anonymous-namespace ``int_ph``.

        ``y`` can be negative: the Cornish-Fisher bound that ``pdf`` uses as
        the upper integration limit goes negative at short horizons, and C++
        then integrates backwards. ``std::sqrt`` of a negative double is a
        quiet NaN in C++ but a ``ValueError`` in Python, so the square root is
        spelled out here to propagate NaN instead — that is what makes the
        port reproduce C++'s NaN rather than raising where C++ returns.
        """
        rho = self._rho
        kappa = self._kappa
        sigma = self._sigma
        x0 = math.log(self._s0.value())

        radicand = 2.0 * math.pi * (1.0 - rho * rho) * y
        if radicand < 0.0:
            return math.nan

        integral = _GAUSS_LAGUERRE_128(lambda u: self._ph(y, u, nu_0, nu_t, t))
        return (
            integral
            / math.sqrt(radicand)
            * math.exp(
                -0.5
                * (x - x0 - a + y * (0.5 - rho * kappa / sigma)) ** 2
                / (y * (1.0 - rho * rho))
            )
        )

    def _cornish_fisher_eps(self, nu_0: float, nu_t: float, dt: float, eps: float) -> float:
        """# C++ parity: the anonymous-namespace ``cornishFisherEps``.

        Four central differences of the moment-generating function at step
        ``d = 1e-2`` give the first four moments; a Cornish-Fisher expansion
        then estimates the ``1-eps`` quantile.
        """
        d = 1e-2
        p2 = self._phi(complex(0.0, -2.0 * d), nu_0, nu_t, dt).real
        p1 = self._phi(complex(0.0, -d), nu_0, nu_t, dt).real
        p0 = self._phi(complex(0.0, 0.0), nu_0, nu_t, dt).real
        pm1 = self._phi(complex(0.0, d), nu_0, nu_t, dt).real
        pm2 = self._phi(complex(0.0, 2.0 * d), nu_0, nu_t, dt).real

        avg = (pm2 - 8.0 * pm1 + 8.0 * p1 - p2) / (12.0 * d)
        m2 = (-pm2 + 16.0 * pm1 - 30.0 * p0 + 16.0 * p1 - p2) / (12.0 * d * d)
        var = m2 - avg * avg
        std_dev = math.sqrt(var)

        m3 = (-0.5 * pm2 + pm1 - p1 + 0.5 * p2) / (d * d * d)
        skew = (m3 - 3.0 * var * avg - avg * avg * avg) / (var * std_dev)

        m4 = (pm2 - 4.0 * pm1 + 6.0 * p0 - 4.0 * p1 + p2) / (d * d * d * d)
        kurt = (m4 - 4.0 * m3 * avg + 6.0 * m2 * avg * avg - 3.0 * avg**4) / (var * var)

        q = _INV_CUM_NORMAL(1.0 - eps)
        w = (
            q
            + (q * q - 1.0) / 6.0 * skew
            + (q * q * q - 3.0 * q) / 24.0 * (kurt - 3.0)
            - (2.0 * q * q * q - 5.0 * q) / 36.0 * skew * skew
        )
        return avg + w * std_dev

    def pdf(self, x: float, v: float, t: float, eps: float = 1e-3) -> float:
        """Joint terminal density of ``(ln S, v)`` at horizon ``t``.

        # C++ parity: ``HestonProcess::pdf``.

        The first ``while`` loop in C++ widens ``upper`` until the integrand
        decays, but its result is then immediately overwritten by the
        Cornish-Fisher bound. It is reproduced anyway — it is not dead code in
        the sense of being removable: it can throw or loop, and the number of
        evaluations is observable.
        """
        sigma = self._sigma
        kappa = self._kappa
        rho = self._rho
        theta = self._theta
        v0 = self._v0

        k = sigma * sigma * (1.0 - math.exp(-kappa * t)) / (4.0 * kappa)
        a = math.log(
            self._dividend_yield.discount(t) / self._risk_free_rate.discount(t)
        ) + rho / sigma * (v - v0 - kappa * theta * t)

        x0 = math.log(self._s0.value())
        upper = max(0.1, -(x - x0 - a) / (0.5 - rho * kappa / sigma))
        f = 0.0
        df = 1.0

        while df > 0.0 or f > 0.1 * eps:
            f1 = x - x0 - a + upper * (0.5 - rho * kappa / sigma)
            f2 = -0.5 * f1 * f1 / (upper * (1.0 - rho * rho))

            df = (
                1.0
                / math.sqrt(2.0 * math.pi * (1.0 - rho * rho))
                * (
                    -0.5 / (upper * math.sqrt(upper)) * math.exp(f2)
                    + 1.0
                    / math.sqrt(upper)
                    * math.exp(f2)
                    * (-0.5 / (1.0 - rho * rho))
                    * (
                        -1.0 / (upper * upper) * f1 * f1
                        + 2.0 / upper * f1 * (0.5 - rho * kappa / sigma)
                    )
                )
            )
            f = math.exp(f2) / math.sqrt(2.0 * math.pi * (1.0 - rho * rho) * upper)
            upper *= 1.5

        upper = 2.0 * self._cornish_fisher_eps(v0, v, t, 1e-3)

        df_chi = 4.0 * theta * kappa / (sigma * sigma)
        ncp = (
            4.0
            * kappa
            * math.exp(-kappa * t)
            / ((sigma * sigma) * (1.0 - math.exp(-kappa * t)))
            * v0
        )

        return (
            SegmentIntegral(100)(
                lambda xi: self._int_ph(a, x, xi, v0, v, t), QL_EPSILON, upper
            )
            * float(ncx2.pdf(v / k, df_chi, ncp))  # pyright: ignore[reportUnknownMemberType]
            / k
        )


_GAUSS_LAGUERRE_128 = GaussLaguerreIntegration(128)
_INV_CUM_NORMAL = InverseCumulativeNormal()


__all__ = ["HestonProcess"]


__all__ = ["Discretization", "HestonProcess"]

