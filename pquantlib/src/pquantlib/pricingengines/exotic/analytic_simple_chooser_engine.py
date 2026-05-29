"""AnalyticSimpleChooserEngine — closed-form for the simple chooser option.

# C++ parity:
# ql/pricingengines/exotic/analyticsimplechooserengine.{hpp,cpp}
# (v1.42.1).

Rubinstein 1991 closed-form. The chooser-option price is::

    spot * exp(-q*T) * N(d) - K * exp(-r*T) * N(d - sigma*sqrt(T))
        - spot * exp(-q*T) * N(-y) + K * exp(-r*T) * N(-y + sigma*sqrt(t_c))

with::

    d = (ln(spot/K) + ((r - q) + sigma^2/2) * T) / (sigma*sqrt(T))
    y = (ln(spot/K) + (r - q) * T + sigma^2 * t_c / 2) / (sigma*sqrt(t_c))

where ``T`` is time to exercise and ``t_c`` is time to the choosing
date.

The engine requires the three term-structures (risk-free rate,
dividend yield, black volatility) to share a day counter — same
defensive ``QL_REQUIRE`` as C++.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.instruments.simple_chooser_option import (
    SimpleChooserOptionArguments,
)
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency


class AnalyticSimpleChooserEngine(
    GenericEngine[SimpleChooserOptionArguments, OneAssetOptionResults]
):
    """Rubinstein 1991 closed-form for the simple chooser option.

    # C++ parity: ``AnalyticSimpleChooserEngine``.
    """

    def __init__(self, process: GeneralizedBlackScholesProcess) -> None:
        super().__init__(SimpleChooserOptionArguments(), OneAssetOptionResults())
        self._process: GeneralizedBlackScholesProcess = process
        process.register_with(self)

    def calculate(self) -> None:
        """Compute the chooser NPV.

        # C++ parity: ``AnalyticSimpleChooserEngine::calculate``.
        """
        args = self._arguments
        results = self._results

        rfdc = self._process.risk_free_rate().day_counter()
        divdc = self._process.dividend_yield().day_counter()
        voldc = self._process.black_volatility().day_counter()
        qassert.require(
            rfdc == divdc,
            "Risk-free rate and dividend yield must have the same day counter",
        )
        qassert.require(
            rfdc == voldc,
            "Risk-free rate and volatility must have the same day counter",
        )

        spot = self._process.state_variable().value()
        payoff = args.payoff
        qassert.require(
            isinstance(payoff, StrikedTypePayoff), "non-plain payoff given"
        )
        assert isinstance(payoff, StrikedTypePayoff)
        strike = payoff.strike()
        assert args.exercise is not None
        maturity = args.exercise.last_date()
        volatility = self._process.black_volatility().black_vol(
            maturity, strike, extrapolate=True
        )

        today = self._process.risk_free_rate().reference_date()
        time_to_maturity = rfdc.year_fraction(today, maturity)
        time_to_choosing = rfdc.year_fraction(today, args.choosing_date)

        dividend_rate = self._process.dividend_yield().zero_rate(
            maturity, Compounding.Continuous, Frequency.NoFrequency,
        ).rate()
        risk_free_rate = self._process.risk_free_rate().zero_rate(
            maturity, Compounding.Continuous, Frequency.NoFrequency,
        ).rate()

        qassert.require(spot > 0.0, "negative or null spot value")
        qassert.require(strike > 0.0, "negative or null strike value")
        qassert.require(volatility > 0.0, "negative or null volatility")
        qassert.require(
            time_to_choosing > 0.0,
            "choosing date earlier than or equal to evaluation date",
        )

        d = (
            math.log(spot / strike)
            + ((risk_free_rate - dividend_rate) + volatility * volatility * 0.5)
            * time_to_maturity
        ) / (volatility * math.sqrt(time_to_maturity))

        y = (
            math.log(spot / strike)
            + (risk_free_rate - dividend_rate) * time_to_maturity
            + (volatility * volatility * time_to_choosing / 2.0)
        ) / (volatility * math.sqrt(time_to_choosing))

        n = CumulativeNormalDistribution()

        results.reset()
        results.value = (
            spot * math.exp(-dividend_rate * time_to_maturity) * n(d)
            - strike
            * math.exp(-risk_free_rate * time_to_maturity)
            * n(d - volatility * math.sqrt(time_to_maturity))
            - spot * math.exp(-dividend_rate * time_to_maturity) * n(-y)
            + strike
            * math.exp(-risk_free_rate * time_to_maturity)
            * n(-y + volatility * math.sqrt(time_to_choosing))
        )

    def update(self) -> None:
        self.notify_observers()


__all__ = ["AnalyticSimpleChooserEngine"]
