"""AnalyticWriterExtensibleOptionEngine — Haug 2007 closed-form.

# C++ parity:
# ql/pricingengines/exotic/analyticwriterextensibleoptionengine.{hpp,cpp}
# (v1.42.1).

Haug 2007 closed-form for writer-extensible options under Black-Scholes:

  Result = BSM(payoff1, t1)
         +/- spot * exp((b-r)*t2) * biv(z1, -z2)
         -/+ payoff2.strike * exp(-r*t2)
              * biv(z1 - vol*sqrt(t2), -z2 + vol*sqrt(t1))

where ``biv`` is the bivariate-normal CDF with correlation ``-rho``
and ``rho = sqrt(t1/t2)``. The put case uses the symmetric variant.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.instruments.writer_extensible_option import (
    WriterExtensibleOptionArguments,
)
from pquantlib.math.distributions.bivariate_normal_distribution import (
    BivariateCumulativeNormalDistribution,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.black_formula import black_formula
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency


class AnalyticWriterExtensibleOptionEngine(
    GenericEngine[WriterExtensibleOptionArguments, OneAssetOptionResults]
):
    """Haug 2007 closed-form for writer-extensible options.

    # C++ parity: ``AnalyticWriterExtensibleOptionEngine``.
    """

    def __init__(self, process: GeneralizedBlackScholesProcess) -> None:
        super().__init__(
            WriterExtensibleOptionArguments(), OneAssetOptionResults()
        )
        self._process: GeneralizedBlackScholesProcess = process
        process.register_with(self)

    def calculate(self) -> None:
        """Closed-form NPV.

        # C++ parity: ``AnalyticWriterExtensibleOptionEngine::calculate``.
        """
        args = self._arguments
        results = self._results

        payoff1 = args.payoff
        qassert.require(
            isinstance(payoff1, PlainVanillaPayoff), "not a plain vanilla payoff"
        )
        assert isinstance(payoff1, PlainVanillaPayoff)

        payoff2 = args.payoff2
        qassert.require(
            isinstance(payoff2, PlainVanillaPayoff), "not a plain vanilla payoff"
        )
        assert isinstance(payoff2, PlainVanillaPayoff)

        assert args.exercise is not None
        assert args.exercise2 is not None

        option_type = payoff1.option_type()

        # Step 1: spot + curves at t1.
        spot = self._process.state_variable().value()
        dividend_dc = self._process.dividend_yield().day_counter()
        dividend = self._process.dividend_yield().zero_rate(
            args.exercise.last_date(),
            Compounding.Continuous,
            Frequency.NoFrequency,
            result_day_counter=dividend_dc,
        ).rate()
        rfdc = self._process.risk_free_rate().day_counter()
        risk_free = self._process.risk_free_rate().zero_rate(
            args.exercise.last_date(),
            Compounding.Continuous,
            Frequency.NoFrequency,
            result_day_counter=rfdc,
        ).rate()
        t1 = rfdc.year_fraction(
            self._process.risk_free_rate().reference_date(),
            args.exercise.last_date(),
        )
        t2 = rfdc.year_fraction(
            self._process.risk_free_rate().reference_date(),
            args.exercise2.last_date(),
        )

        b = risk_free - dividend
        forward_price = spot * math.exp(b * t1)

        volatility = self._process.black_volatility().black_vol(
            args.exercise.last_date(), payoff1.strike(), extrapolate=True
        )
        std_dev = volatility * math.sqrt(t1)
        discount = math.exp(-risk_free * t1)

        black = black_formula(
            option_type, payoff1.strike(), forward_price, std_dev, discount
        )

        # Step 2: bivariate-normal arguments.
        ro = math.sqrt(t1 / t2)
        z1 = (
            math.log(spot / payoff2.strike())
            + (b + volatility * volatility / 2.0) * t2
        ) / (volatility * math.sqrt(t2))
        z2 = (
            math.log(spot / payoff1.strike())
            + (b + volatility * volatility / 2.0) * t1
        ) / (volatility * math.sqrt(t1))

        biv = BivariateCumulativeNormalDistribution(-ro)

        # Step 3: combine.
        if option_type == OptionType.Call:
            bivariate1 = biv(z1, -z2)
            bivariate2 = biv(
                z1 - volatility * math.sqrt(t2), -z2 + volatility * math.sqrt(t1)
            )
            result = (
                black
                + spot * math.exp((b - risk_free) * t2) * bivariate1
                - payoff2.strike() * math.exp(-risk_free * t2) * bivariate2
            )
        else:
            bivariate1 = biv(-z1, z2)
            bivariate2 = biv(
                -z1 + volatility * math.sqrt(t2), z2 - volatility * math.sqrt(t1)
            )
            result = (
                black
                - spot * math.exp((b - risk_free) * t2) * bivariate1
                + payoff2.strike() * math.exp(-risk_free * t2) * bivariate2
            )

        results.reset()
        results.value = result

    def update(self) -> None:
        self.notify_observers()


__all__ = ["AnalyticWriterExtensibleOptionEngine"]
