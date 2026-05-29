"""AnalyticTwoAssetCorrelationEngine — Zhang closed form.

# C++ parity: ql/pricingengines/exotic/analytictwoassetcorrelationengine.{hpp,cpp}
# (v1.42.1).

Closed-form pricing for the ``TwoAssetCorrelationOption`` under
constant-parameter BSM dynamics on both assets, with constant
correlation between Brownian drivers.

Reference: Zhang's formulas as reproduced in Haug, "Option Pricing
Formulas". Implementation here mirrors the C++ exactly:

    Call: S2 * e^((b2-r)T) * M(y2+s2*sqrt(T), y1+rho*s2*sqrt(T))
          - X2 * e^(-rT) * M(y2, y1)

    Put:  X2 * e^(-rT) * M(-y2, -y1)
          - S2 * e^((b2-r)T) * M(-y2-s2*sqrt(T), -y1-rho*s2*sqrt(T))

where M(.,.) is the bivariate cumulative standard normal with
correlation ``rho``, ``s1`` / ``s2`` are vols, ``b1`` / ``b2`` are
cost-of-carries (r - q), and y1/y2 are the standardized log moneyness
quantiles.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.instruments.multi_asset_option import MultiAssetOptionResults
from pquantlib.instruments.two_asset_correlation_option import (
    TwoAssetCorrelationOptionArguments,
)
from pquantlib.math.distributions.bivariate_normal_distribution import (
    BivariateCumulativeNormalDistributionDr78,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.quote import Quote
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency


class AnalyticTwoAssetCorrelationEngine(
    GenericEngine[TwoAssetCorrelationOptionArguments, MultiAssetOptionResults]
):
    """Zhang closed-form engine for ``TwoAssetCorrelationOption``.

    # C++ parity: ``AnalyticTwoAssetCorrelationEngine``.

    Args:
        p1: GBSM process for the first (trigger) asset (S1, strike X1).
        p2: GBSM process for the second (payoff) asset (S2, strike X2).
        correlation: Correlation between Brownian drivers as a ``Quote``.
    """

    def __init__(
        self,
        p1: GeneralizedBlackScholesProcess,
        p2: GeneralizedBlackScholesProcess,
        correlation: Quote,
    ) -> None:
        super().__init__(
            TwoAssetCorrelationOptionArguments(), MultiAssetOptionResults()
        )
        self._p1: GeneralizedBlackScholesProcess = p1
        self._p2: GeneralizedBlackScholesProcess = p2
        self._correlation: Quote = correlation
        p1.register_with(self)
        p2.register_with(self)
        correlation.register_with(self)

    def calculate(self) -> None:
        """Apply the Zhang closed form.

        # C++ parity: ``AnalyticTwoAssetCorrelationEngine::calculate``.
        """
        rho = self._correlation.value()
        m = BivariateCumulativeNormalDistributionDr78(rho)

        po = self._arguments.payoff
        qassert.require(isinstance(po, PlainVanillaPayoff), "non-plain payoff given")
        assert isinstance(po, PlainVanillaPayoff)
        qassert.require(po.strike() > 0.0, "strike must be positive")

        ex = self._arguments.exercise
        assert ex is not None
        strike = po.strike()  # X1
        spot = self._p1.state_variable().value()
        qassert.require(spot > 0.0, "negative or null underlying given")

        last_date = ex.last_date()
        t = self._p2.time(last_date)

        sigma1 = self._p1.black_volatility().black_vol_at_time(
            self._p1.time(last_date), strike, extrapolate=True
        )
        sigma2 = self._p2.black_volatility().black_vol_at_time(
            self._p2.time(last_date), strike, extrapolate=True
        )

        s1 = self._p1.state_variable().value()
        s2 = self._p2.state_variable().value()
        q1 = self._p1.dividend_yield().zero_rate(
            t, Compounding.Continuous, Frequency.NoFrequency
        ).rate()
        q2 = self._p2.dividend_yield().zero_rate(
            t, Compounding.Continuous, Frequency.NoFrequency
        ).rate()
        r = self._p1.risk_free_rate().zero_rate(
            t, Compounding.Continuous, Frequency.NoFrequency
        ).rate()
        b1 = r - q1
        b2 = r - q2

        x2 = self._arguments.x2
        assert x2 is not None

        sqrt_t = math.sqrt(t)
        y1 = (math.log(s1 / strike) + (b1 - 0.5 * sigma1 * sigma1) * t) / (
            sigma1 * sqrt_t
        )
        y2 = (math.log(s2 / x2) + (b2 - 0.5 * sigma2 * sigma2) * t) / (
            sigma2 * sqrt_t
        )

        if po.option_type() == OptionType.Call:
            self._results.value = (
                s2 * math.exp((b2 - r) * t)
                * m(y2 + sigma2 * sqrt_t, y1 + rho * sigma2 * sqrt_t)
                - x2 * math.exp(-r * t) * m(y2, y1)
            )
        else:  # Put
            self._results.value = (
                x2 * math.exp(-r * t) * m(-y2, -y1)
                - s2 * math.exp((b2 - r) * t)
                * m(-y2 - sigma2 * sqrt_t, -y1 - rho * sigma2 * sqrt_t)
            )


__all__ = ["AnalyticTwoAssetCorrelationEngine"]
