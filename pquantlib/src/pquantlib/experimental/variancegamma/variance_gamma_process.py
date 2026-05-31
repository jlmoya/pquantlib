"""VarianceGammaProcess — Brownian motion time-changed by a Gamma process.

# C++ parity: ql/experimental/variancegamma/variancegammaprocess.{hpp,cpp}
# (v1.42.1).

A Variance-Gamma process evaluates a drifted Brownian motion
``db = theta dt + sigma dW`` at random times driven by a Gamma process
with mean 1 and variance rate ``nu``::

    X(t) = B(T)   where T ~ Gamma(t/nu, nu)

The three structural parameters are:

* ``sigma`` — the Brownian volatility,
* ``nu`` — the Gamma variance rate (controls kurtosis),
* ``theta`` — the Brownian drift (controls skew).

Like the C++ class, ``drift`` / ``diffusion`` are NOT implemented (the
process is consumed only via its characteristic function in the analytic
/ FFT engines); calling either raises ``LibraryException``. The
discretization slot is an ``EulerDiscretization`` purely for ctor parity.
"""

from __future__ import annotations

from typing import final

from pquantlib.exceptions import LibraryException
from pquantlib.processes.euler_discretization import EulerDiscretization
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.yield_term_structure import YieldTermStructure


@final
class VarianceGammaProcess(StochasticProcess1D):
    """Variance-Gamma stochastic process.

    # C++ parity: ``class VarianceGammaProcess : public
    # StochasticProcess1D`` in variancegammaprocess.hpp:50.

    Parameters
    ----------
    s0
        Spot quote.
    dividend_yield
        Dividend-yield curve.
    risk_free_rate
        Risk-free curve.
    sigma, nu, theta
        Variance-Gamma structural parameters.
    """

    __slots__ = (
        "_dividend_yield",
        "_nu",
        "_risk_free_rate",
        "_s0",
        "_sigma",
        "_theta",
    )

    def __init__(
        self,
        s0: Quote,
        dividend_yield: YieldTermStructure,
        risk_free_rate: YieldTermStructure,
        sigma: float,
        nu: float,
        theta: float,
    ) -> None:
        # C++ parity: ctor forwards an EulerDiscretization + registers.
        super().__init__(EulerDiscretization())
        self._s0: Quote = s0
        self._dividend_yield: YieldTermStructure = dividend_yield
        self._risk_free_rate: YieldTermStructure = risk_free_rate
        self._sigma: float = float(sigma)
        self._nu: float = float(nu)
        self._theta: float = float(theta)
        risk_free_rate.register_with(self)
        dividend_yield.register_with(self)
        s0.register_with(self)

    def x0(self) -> float:
        # C++ parity: variancegammaprocess.cpp x0() = s0_->value().
        return self._s0.value()

    def drift_1d(self, t: float, x: float) -> float:
        # C++ parity: QL_FAIL("not implemented yet").
        raise LibraryException("not implemented yet")

    def diffusion_1d(self, t: float, x: float) -> float:
        # C++ parity: QL_FAIL("not implemented yet").
        raise LibraryException("not implemented yet")

    # --- accessors ------------------------------------------------------

    def sigma(self) -> float:
        # C++ parity: sigma() accessor.
        return self._sigma

    def nu(self) -> float:
        # C++ parity: nu() accessor.
        return self._nu

    def theta(self) -> float:
        # C++ parity: theta() accessor.
        return self._theta

    def s0(self) -> Quote:
        # C++ parity: s0() accessor.
        return self._s0

    def dividend_yield(self) -> YieldTermStructure:
        # C++ parity: dividendYield() accessor.
        return self._dividend_yield

    def risk_free_rate(self) -> YieldTermStructure:
        # C++ parity: riskFreeRate() accessor.
        return self._risk_free_rate


__all__ = ["VarianceGammaProcess"]
