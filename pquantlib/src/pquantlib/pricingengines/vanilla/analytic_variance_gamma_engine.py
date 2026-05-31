"""VarianceGammaEngine — analytic European pricing under a VG process.

# C++ parity: ql/experimental/variancegamma/analyticvariancegammaengine.{hpp,cpp}
# (v1.42.1).

Prices a European vanilla option by conditioning on the Gamma subordinator
``T`` and integrating the conditional Black-Scholes price against the Gamma
density (Madan-Carr-Chang 1998). For each subordinator value ``x`` the
conditional spot/vol are::

    s0_adj  = s0 * exp(theta*x + omega*t + sigma^2*x/2)
    vol_adj = sigma * sqrt(x)                 (= sigma*sqrt(x/t)*sqrt(t))

where ``omega = log(1 - theta*nu - sigma^2*nu/2) / nu`` is the
martingale-correction drift; the conditional BS price is then weighted by
the Gamma(t/nu, nu) density and integrated over ``x in (0, inf)``.

Divergence from C++ (delegation):

* C++ splits the integral at 0.1 and uses ``GaussKronrodNonAdaptive``
  on [0, 0.1] plus ``GaussLobattoIntegral`` on [0.1, infinity], with an
  adaptive search for the effective upper limit (grow ``infinity`` by
  1.5x until the integrand tail falls below ``absErr * 1e-4``). The
  pquantlib port delegates the quadrature to
  ``scipy.integrate.quad`` over the same split with the same adaptive
  upper-limit search — the documented LOOSE tolerance (special-function
  quadrature) covers the residual difference between the Gauss-Kronrod /
  Gauss-Lobatto rules and scipy's QUADPACK. The integrand is otherwise
  a letter-for-letter port of the C++ ``Integrand``.
"""

from __future__ import annotations

import math
from typing import cast, final

from scipy.integrate import quad  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.experimental.variancegamma.variance_gamma_process import (
    VarianceGammaProcess,
)
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.distributions.gamma_function import GammaFunction
from pquantlib.option import OptionArguments
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.black_scholes_calculator import BlackScholesCalculator
from pquantlib.pricingengines.generic_engine import GenericEngine


class _Integrand:
    """Conditional-BS-price * Gamma-density integrand.

    # C++ parity: anonymous-namespace ``Integrand`` in
    # analyticvariancegammaengine.cpp:33.
    """

    __slots__ = (
        "_dividend_discount",
        "_gamma_denom",
        "_nu",
        "_omega",
        "_payoff",
        "_risk_free_discount",
        "_s0",
        "_sigma",
        "_t",
        "_theta",
    )

    def __init__(
        self,
        payoff: StrikedTypePayoff,
        s0: float,
        t: float,
        risk_free_discount: float,
        dividend_discount: float,
        sigma: float,
        nu: float,
        theta: float,
    ) -> None:
        self._payoff: StrikedTypePayoff = payoff
        self._s0: float = s0
        self._t: float = t
        self._risk_free_discount: float = risk_free_discount
        self._dividend_discount: float = dividend_discount
        self._sigma: float = sigma
        self._nu: float = nu
        self._theta: float = theta
        self._omega: float = (
            math.log(1.0 - theta * nu - (sigma * sigma * nu) / 2.0) / nu
        )
        # Precompute the Gamma-pdf denominator (shape = t/nu, scale = nu).
        gf = GammaFunction()
        self._gamma_denom: float = math.exp(gf.log_value(t / nu)) * (nu ** (t / nu))

    def __call__(self, x: float) -> float:
        # Conditional Black-Scholes price at subordinator value x.
        s0_adj = self._s0 * math.exp(
            self._theta * x
            + self._omega * self._t
            + (self._sigma * self._sigma * x) / 2.0
        )
        vol_adj = self._sigma * math.sqrt(x / self._t)
        vol_adj *= math.sqrt(self._t)

        bs = BlackScholesCalculator(
            self._payoff,
            s0_adj,
            self._dividend_discount,
            vol_adj,
            self._risk_free_discount,
        )
        bs_price = bs.value()

        # Weight by the Gamma density.
        gamp = (
            (x ** (self._t / self._nu - 1.0)) * math.exp(-x / self._nu)
        ) / self._gamma_denom
        return bs_price * gamp


@final
class VarianceGammaEngine(GenericEngine[OptionArguments, OneAssetOptionResults]):
    """Analytic Variance-Gamma engine for European vanilla options.

    # C++ parity: ``class VarianceGammaEngine : public
    # VanillaOption::engine`` in analyticvariancegammaengine.hpp:38.

    Parameters
    ----------
    process
        The :class:`VarianceGammaProcess`.
    absolute_error
        Target absolute quadrature error (default 1e-5; matches C++).
    """

    def __init__(
        self,
        process: VarianceGammaProcess,
        absolute_error: float = 1e-5,
    ) -> None:
        super().__init__(OptionArguments(), OneAssetOptionResults())
        qassert.require(absolute_error > 0, "absolute error must be positive")
        self._process: VarianceGammaProcess = process
        self._abs_err: float = float(absolute_error)
        process.register_with(self)

    def calculate(self) -> None:
        """Integrate the conditional BS price against the Gamma density.

        # C++ parity: ``VarianceGammaEngine::calculate``.
        """
        args = self._arguments
        results = self._results

        qassert.require(args.exercise is not None, "no exercise given")
        qassert.require(args.payoff is not None, "no payoff given")
        assert args.exercise is not None
        assert args.payoff is not None

        qassert.require(
            args.exercise.type() == Exercise.Type.European,
            "not an European Option",
        )
        qassert.require(
            isinstance(args.payoff, StrikedTypePayoff),
            "non-striked payoff given",
        )
        assert isinstance(args.payoff, StrikedTypePayoff)
        payoff: StrikedTypePayoff = args.payoff

        process = self._process
        last_date = args.exercise.last_date()
        dividend_discount = process.dividend_yield().discount(last_date)
        risk_free_discount = process.risk_free_rate().discount(last_date)

        rfdc = process.risk_free_rate().day_counter()
        t = rfdc.year_fraction(process.risk_free_rate().reference_date(), last_date)

        f = _Integrand(
            payoff,
            process.x0(),
            t,
            risk_free_discount,
            dividend_discount,
            process.sigma(),
            process.nu(),
            process.theta(),
        )

        # Adaptive upper-limit search (C++: grow by 1.5x until the tail
        # integrand falls below absErr * 1e-4).
        infinity = 15.0 * math.sqrt(process.nu() * t)
        target = self._abs_err * 1e-4
        val = f(infinity)
        while abs(val) > target:
            infinity *= 1.5
            val = f(infinity)

        # Split the integral at 0.1 (C++ uses Kronrod on [0,0.1] +
        # Lobatto on [0.1, infinity]; we delegate both to scipy.quad).
        split = 0.1
        pv_a, _ = cast(
            "tuple[float, float]",
            quad(f, 0.0, split, epsabs=self._abs_err, epsrel=self._abs_err, limit=1000),
        )
        pv_b, _ = cast(
            "tuple[float, float]",
            quad(f, split, infinity, epsabs=self._abs_err, epsrel=self._abs_err, limit=2000),
        )
        results.value = pv_a + pv_b


__all__ = ["VarianceGammaEngine"]
