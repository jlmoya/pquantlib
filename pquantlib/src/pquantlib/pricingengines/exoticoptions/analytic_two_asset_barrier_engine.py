"""AnalyticTwoAssetBarrierEngine — Heynen-Kat closed form (1994).

# C++ parity: ql/pricingengines/barrier/analytictwoassetbarrierengine.{hpp,cpp}
# (v1.42.1).

Closed-form pricing for a barrier option where the payoff is on the
first asset (S1) and the barrier is monitored on the second asset (S2).
Both assets are driven by correlated BSM processes with constant
parameters.

Reference: Heynen & Kat (1994), reproduced in Haug, "Option Pricing
Formulas" (1997 / 2007).

The C++ implementation has the following structure:

* ``calculate()`` dispatches on (option_type, barrier_type) and
  combines call/put parity with the ``A(eta, phi)`` helper:

    Call/UpOut:      A(+1, +1)
    Call/DownOut:    A(+1, -1)
    Call/UpIn:       call - A(+1, +1)
    Call/DownIn:     call - A(+1, -1)
    Put/UpOut:       A(-1, +1)
    Put/DownOut:     A(-1, -1)
    Put/UpIn:        put  - A(-1, +1)
    Put/DownIn:      put  - A(-1, -1)

  (The C++ ``B(eta, phi)`` helper is hard-coded to return 0 because
  the rebate functionality of Heynen-Kat is not exposed by this
  engine — matches v1.42.1.)
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.instruments.barrier_option import BarrierType
from pquantlib.instruments.instrument import InstrumentResults
from pquantlib.instruments.two_asset_barrier_option import (
    TwoAssetBarrierOptionArguments,
)
from pquantlib.math.distributions.bivariate_normal_distribution import (
    BivariateCumulativeNormalDistributionDr78,
)
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.quote import Quote
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency


class AnalyticTwoAssetBarrierEngine(
    GenericEngine[TwoAssetBarrierOptionArguments, InstrumentResults]
):
    """Heynen-Kat closed-form engine for ``TwoAssetBarrierOption``.

    # C++ parity: ``AnalyticTwoAssetBarrierEngine``.

    Args:
        process1: GBSM process for the first (payoff) asset.
        process2: GBSM process for the second (barrier-monitoring) asset.
        rho: Correlation between the two Brownian drivers as a ``Quote``.
    """

    def __init__(
        self,
        process1: GeneralizedBlackScholesProcess,
        process2: GeneralizedBlackScholesProcess,
        rho: Quote,
    ) -> None:
        super().__init__(TwoAssetBarrierOptionArguments(), InstrumentResults())
        self._process1: GeneralizedBlackScholesProcess = process1
        self._process2: GeneralizedBlackScholesProcess = process2
        self._rho: Quote = rho
        process1.register_with(self)
        process2.register_with(self)
        rho.register_with(self)

    # --- helpers ---------------------------------------------------------

    def _triggered(self, underlying: float) -> bool:
        """Has the barrier been hit at current ``underlying`` (S2)?

        # C++ parity: ``TwoAssetBarrierOption::engine::triggered``.
        """
        bt = self._arguments.barrier_type
        assert self._arguments.barrier is not None
        b = self._arguments.barrier
        if bt in (BarrierType.DownIn, BarrierType.DownOut):
            return underlying < b
        if bt in (BarrierType.UpIn, BarrierType.UpOut):
            return underlying > b
        # qassert.require here would be safer but the validate() upstream
        # already rejected anything else.
        raise AssertionError(f"unknown barrier type {bt}")

    def _residual_time(self) -> float:
        assert self._arguments.exercise is not None
        return self._process1.time(self._arguments.exercise.last_date())

    def _strike(self) -> float:
        po = self._arguments.payoff
        qassert.require(isinstance(po, PlainVanillaPayoff), "non-plain payoff given")
        assert isinstance(po, PlainVanillaPayoff)
        return po.strike()

    def _vol1(self) -> float:
        # C++ uses ``black_vol(t, strike)`` with the *option*-strike.
        return self._process1.black_volatility().black_vol_at_time(
            self._residual_time(), self._strike(), extrapolate=True
        )

    def _vol2(self) -> float:
        return self._process2.black_volatility().black_vol_at_time(
            self._residual_time(), self._strike(), extrapolate=True
        )

    def _risk_free_rate(self) -> float:
        return self._process1.risk_free_rate().zero_rate(
            self._residual_time(), Compounding.Continuous, Frequency.NoFrequency
        ).rate()

    def _div_yield_1(self) -> float:
        return self._process1.dividend_yield().zero_rate(
            self._residual_time(), Compounding.Continuous, Frequency.NoFrequency
        ).rate()

    def _div_yield_2(self) -> float:
        return self._process2.dividend_yield().zero_rate(
            self._residual_time(), Compounding.Continuous, Frequency.NoFrequency
        ).rate()

    @staticmethod
    def _bivariate_m(a: float, b: float, rho: float) -> float:
        f = BivariateCumulativeNormalDistributionDr78(rho)
        return f(a, b)

    def _vanilla_call(self) -> float:
        nd = CumulativeNormalDistribution()
        s1 = self._process1.state_variable().value()
        strike = self._strike()
        v1 = self._vol1()
        t = self._residual_time()
        r = self._risk_free_rate()
        b1 = r - self._div_yield_1()
        mu1 = b1 - 0.5 * v1 * v1
        d1 = (math.log(s1 / strike) + (mu1 + v1 * v1) * t) / (v1 * math.sqrt(t))
        d2 = d1 - v1 * math.sqrt(t)
        return s1 * nd(d1) - strike * math.exp(-r * t) * nd(d2)

    def _vanilla_put(self) -> float:
        nd = CumulativeNormalDistribution()
        s1 = self._process1.state_variable().value()
        strike = self._strike()
        v1 = self._vol1()
        t = self._residual_time()
        r = self._risk_free_rate()
        b1 = r - self._div_yield_1()
        mu1 = b1 - 0.5 * v1 * v1
        d1 = (math.log(s1 / strike) + (mu1 + v1 * v1) * t) / (v1 * math.sqrt(t))
        d2 = d1 - v1 * math.sqrt(t)
        return strike * math.exp(-r * t) * nd(-d2) - s1 * nd(-d1)

    def _a(self, eta: float, phi: float) -> float:
        """Heynen-Kat A helper.

        # C++ parity: ``AnalyticTwoAssetBarrierEngine::A``.
        """
        s1 = self._process1.state_variable().value()
        s2 = self._process2.state_variable().value()
        r = self._risk_free_rate()
        b1 = r - self._div_yield_1()
        b2 = r - self._div_yield_2()
        t = self._residual_time()
        assert self._arguments.barrier is not None
        h = self._arguments.barrier
        x = self._strike()
        sigma1 = self._vol1()
        sigma2 = self._vol2()
        rho = self._rho.value()

        mu1 = b1 - 0.5 * sigma1 * sigma1
        mu2 = b2 - 0.5 * sigma2 * sigma2

        sqrt_t = math.sqrt(t)
        d1 = (math.log(s1 / x) + (mu1 + sigma1 * sigma1) * t) / (sigma1 * sqrt_t)
        d2 = d1 - sigma1 * sqrt_t
        d3 = d1 + (2.0 * rho * math.log(h / s2)) / (sigma2 * sqrt_t)
        d4 = d2 + (2.0 * rho * math.log(h / s2)) / (sigma2 * sqrt_t)

        e1 = (math.log(h / s2) - (mu2 + rho * sigma1 * sigma2) * t) / (sigma2 * sqrt_t)
        e2 = e1 + rho * sigma1 * sqrt_t
        e3 = e1 - (2.0 * math.log(h / s2)) / (sigma2 * sqrt_t)
        e4 = e2 - (2.0 * math.log(h / s2)) / (sigma2 * sqrt_t)

        m = self._bivariate_m
        exp_drift_b = math.exp(
            (2.0 * (mu2 + rho * sigma1 * sigma2) * math.log(h / s2)) / (sigma2 * sigma2)
        )
        exp_drift_a = math.exp((2.0 * mu2 * math.log(h / s2)) / (sigma2 * sigma2))

        w = (
            eta * s1 * math.exp((b1 - r) * t)
            * (
                m(eta * d1, phi * e1, -eta * phi * rho)
                - exp_drift_b * m(eta * d3, phi * e3, -eta * phi * rho)
            )
            - eta * x * math.exp(-r * t)
            * (
                m(eta * d2, phi * e2, -eta * phi * rho)
                - exp_drift_a * m(eta * d4, phi * e4, -eta * phi * rho)
            )
        )
        return w

    # --- engine entry-point ----------------------------------------------

    def calculate(self) -> None:
        """Dispatch on (option type, barrier type).

        # C++ parity: ``AnalyticTwoAssetBarrierEngine::calculate``.
        """
        po = self._arguments.payoff
        qassert.require(isinstance(po, PlainVanillaPayoff), "non-plain payoff given")
        assert isinstance(po, PlainVanillaPayoff)
        qassert.require(po.strike() > 0.0, "strike must be positive")

        spot2 = self._process2.state_variable().value()
        qassert.require(spot2 > 0.0, "negative or null underlying given")
        qassert.require(not self._triggered(spot2), "barrier touched")

        barrier_type = self._arguments.barrier_type
        option_type = po.option_type()

        if option_type == OptionType.Call:
            if barrier_type == BarrierType.DownOut:
                value = self._a(+1.0, -1.0)
            elif barrier_type == BarrierType.UpOut:
                value = self._a(+1.0, +1.0)
            elif barrier_type == BarrierType.DownIn:
                value = self._vanilla_call() - self._a(+1.0, -1.0)
            elif barrier_type == BarrierType.UpIn:
                value = self._vanilla_call() - self._a(+1.0, +1.0)
            else:
                raise AssertionError(f"unknown barrier type {barrier_type}")
        elif barrier_type == BarrierType.DownOut:
            value = self._a(-1.0, -1.0)
        elif barrier_type == BarrierType.UpOut:
            value = self._a(-1.0, +1.0)
        elif barrier_type == BarrierType.DownIn:
            value = self._vanilla_put() - self._a(-1.0, -1.0)
        elif barrier_type == BarrierType.UpIn:
            value = self._vanilla_put() - self._a(-1.0, +1.0)
        else:
            raise AssertionError(f"unknown barrier type {barrier_type}")
        self._results.value = value


__all__ = ["AnalyticTwoAssetBarrierEngine"]
