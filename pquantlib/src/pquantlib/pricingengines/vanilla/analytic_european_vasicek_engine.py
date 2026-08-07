"""AnalyticBlackVasicekEngine — European vanilla under Black-Scholes + Vasicek rates.

# C++ parity: ql/pricingengines/vanilla/analyticeuropeanvasicekengine.{hpp,cpp}
# (v1.43) — ``class AnalyticBlackVasicekEngine : public VanillaOption::engine``.
#
# Note the C++ HEADER name (analyticeuropeanvasicekengine.hpp) does not match the
# CLASS name (AnalyticBlackVasicekEngine); the module name here mirrors the header,
# per the port's naming convention, and the class name mirrors the class.

Prices a European vanilla when the short rate follows Vasicek

    dr_t = a (b - r_t) dt + sigma_r dW_r

correlated with the equity Brownian at a constant ``correlation``. The result is
a Black-76-shaped formula on the Vasicek zero-coupon bond,

    upsilon = Integral_0^T [ sigma_s^2 + 2 rho sigma_s sigma_r g(T-u)
                             + sigma_r^2 g(T-u)^2 ] du,     g(t) = (1-e^{-kappa t})/kappa
    d+-     = (ln((S/K)/P(0,T)) +- upsilon/2) / sqrt(upsilon)
    value   = eps * (S * N(eps d+) - P(0,T) * K * N(eps d-))

with ``eps = +1`` for a call and ``-1`` for a put. The integral is evaluated with
``SimpsonIntegral(1e-5, 1000)``, constructed once in the C++ constructor.

Reference: http://hsrm-mathematik.de/WS201516/master/option-pricing/Black-Scholes-Vasicek-Model.pdf

Behaviour a port must copy rather than tidy
-------------------------------------------
* **The equity vol is read at t = 0, not at maturity.**
  ``Real sigma_s = blackProcess_->blackVolatility()->blackVol(t, K);`` with
  ``t = 0`` (analyticeuropeanvasicekengine.cpp:79). With a flat surface this is
  invisible; with a term structure it is a different number entirely. The
  reference cases ``vasicek_vol_read_at_time_zero_{1y,5y}`` pin it.
* **``T`` uses the risk-free curve's own day counter and reference date**, not
  the process's ``time()`` helper.
* **Only ``results.value`` is filled.** No Greeks, no additional results.
* The zero-coupon bond is the *Vasicek model's* ``P(0, T, r0)``, not the
  process's discount curve; the two agree only when the Vasicek parameters
  happen to reproduce the curve.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.integrals.simpson import SimpsonIntegral
from pquantlib.models.shortrate.onefactor.vasicek import Vasicek
from pquantlib.option import OptionArguments
from pquantlib.payoffs import OptionType, StrikedTypePayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)


def _g_k(t: float, kappa: float) -> float:
    """``g(t) = (1 - exp(-kappa t)) / kappa``.

    # C++ parity: free function ``g_k`` in the anonymous namespace at
    # analyticeuropeanvasicekengine.cpp:28-30.
    """
    return (1.0 - math.exp(-kappa * t)) / kappa


class _IntegrandVasicek:
    """Instantaneous total-variance rate of ``ln S`` under the joint dynamics.

    # C++ parity: ``class integrand_vasicek`` in the anonymous namespace at
    # analyticeuropeanvasicekengine.cpp:32-46. Anonymous-namespace type with no
    # public C++ name, so it is private here too.
    """

    __slots__ = ("_correlation", "_kappa", "_sigma_r", "_sigma_s", "_t_end")

    def __init__(
        self,
        sigma_s: float,
        sigma_r: float,
        correlation: float,
        kappa: float,
        t_end: float,
    ) -> None:
        self._sigma_s: float = sigma_s
        self._sigma_r: float = sigma_r
        self._correlation: float = correlation
        self._kappa: float = kappa
        self._t_end: float = t_end

    def __call__(self, u: float, /) -> float:
        g = _g_k(self._t_end - u, self._kappa)
        return (
            self._sigma_s * self._sigma_s
            + 2.0 * self._correlation * self._sigma_s * self._sigma_r * g
            + self._sigma_r * self._sigma_r * g * g
        )


class AnalyticBlackVasicekEngine(GenericEngine[OptionArguments, OneAssetOptionResults]):
    """European vanilla under Black-Scholes with a Vasicek short rate.

    # C++ parity: ``class AnalyticBlackVasicekEngine`` in
    # analyticeuropeanvasicekengine.hpp:41-53.
    """

    def __init__(
        self,
        black_process: GeneralizedBlackScholesProcess,
        vasicek_process: Vasicek,
        correlation: float,
    ) -> None:
        """# C++ parity: analyticeuropeanvasicekengine.cpp:50-59."""
        super().__init__(OptionArguments(), OneAssetOptionResults())
        self._black_process: GeneralizedBlackScholesProcess = black_process
        self._vasicek_process: Vasicek = vasicek_process
        # C++ parity: simpsonIntegral_(new SimpsonIntegral(1e-5, 1000)) — built
        # once in the ctor, with those exact knobs.
        self._simpson_integral: SimpsonIntegral = SimpsonIntegral(1e-5, 1000)
        self._correlation: float = float(correlation)
        black_process.register_with(self)
        vasicek_process.register_with(self)

    def correlation(self) -> float:
        """Equity / short-rate correlation supplied at construction."""
        return self._correlation

    def calculate(self) -> None:
        """# C++ parity: ``AnalyticBlackVasicekEngine::calculate`` at
        # analyticeuropeanvasicekengine.cpp:61-89.
        """
        args = self._arguments
        results = self._results

        qassert.require(args.exercise is not None, "no exercise given")
        qassert.require(args.payoff is not None, "no payoff given")
        assert args.exercise is not None
        assert args.payoff is not None

        qassert.require(
            args.exercise.type() == Exercise.Type.European,
            "not an European option",
        )
        qassert.require(
            isinstance(args.payoff, StrikedTypePayoff), "non-striked payoff given"
        )
        assert isinstance(args.payoff, StrikedTypePayoff)
        payoff: StrikedTypePayoff = args.payoff

        f = CumulativeNormalDistribution()

        process = self._black_process
        risk_free = process.risk_free_rate()

        t = 0.0
        # C++ parity: cpp:74 — the risk-free curve's OWN day counter and
        # reference date, not process.time().
        t_end = risk_free.day_counter().year_fraction(
            risk_free.reference_date(), args.exercise.last_date()
        )
        kappa = self._vasicek_process.a()
        s_t = process.x0()
        k = payoff.strike()
        # C++ parity: cpp:79 — the vol is sampled at t == 0, NOT at t_end.
        sigma_s = process.black_volatility().black_vol_at_time(t, k)
        sigma_r = self._vasicek_process.sigma()
        r_t = self._vasicek_process.r0()

        zcb = self._vasicek_process.discount_bond_scalar(t, t_end, r_t)
        epsilon = 1.0 if payoff.option_type() == OptionType.Call else -1.0
        upsilon = self._simpson_integral(
            _IntegrandVasicek(sigma_s, sigma_r, self._correlation, kappa, t_end),
            t,
            t_end,
        )
        sqrt_upsilon = math.sqrt(upsilon)
        log_moneyness = math.log((s_t / k) / zcb)
        d_positive = (log_moneyness + upsilon / 2.0) / sqrt_upsilon
        d_negative = (log_moneyness - upsilon / 2.0) / sqrt_upsilon
        n_d1 = f(epsilon * d_positive)
        n_d2 = f(epsilon * d_negative)

        results.reset()
        results.value = epsilon * ((s_t * n_d1) - (zcb * k * n_d2))


__all__ = ["AnalyticBlackVasicekEngine"]
