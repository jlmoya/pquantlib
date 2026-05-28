"""AnalyticContinuousGeometricAveragePriceAsianEngine — Kemna-Vorst closed form.

# C++ parity: ql/pricingengines/asian/analytic_cont_geom_av_price.{hpp,cpp}
# (v1.42.1).

Closed-form pricing for a European continuous geometric average price
Asian option under Black-Scholes-Merton dynamics. Formula from Haug
"Option Pricing Formulas" pp. 96-97 (Kemna-Vorst 1990): the geometric
average of a GBM is itself log-normal with halved variance and shifted
drift, so the option price reduces to a Black-1976 calculation with
adjusted parameters.

Only the unseasoned case is supported (start_date defaulted): the
seasoned case is deferred (C++ also raises ``QL_FAIL`` for non-null
start dates).
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import Exercise
from pquantlib.instruments.asian_option import (
    AverageType,
    ContinuousAveragingAsianOptionArguments,
)
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.payoffs import PlainVanillaPayoff
from pquantlib.pricingengines.black_calculator import BlackCalculator
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency


class AnalyticContinuousGeometricAveragePriceAsianEngine(
    GenericEngine[ContinuousAveragingAsianOptionArguments, OneAssetOptionResults]
):
    """Kemna-Vorst closed-form engine for continuous geometric Asian.

    # C++ parity: ``AnalyticContinuousGeometricAveragePriceAsianEngine``.
    """

    def __init__(self, process: GeneralizedBlackScholesProcess) -> None:
        super().__init__(
            ContinuousAveragingAsianOptionArguments(), OneAssetOptionResults()
        )
        self._process: GeneralizedBlackScholesProcess = process
        process.register_with(self)

    def calculate(self) -> None:
        """Kemna-Vorst geometric-Asian closed form.

        # C++ parity:
        # ``AnalyticContinuousGeometricAveragePriceAsianEngine::calculate``.
        """
        args = self._arguments
        results = self._results

        qassert.require(
            args.average_type == AverageType.Geometric,
            "not a geometric average option",
        )
        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        qassert.require(
            args.exercise.type() == Exercise.Type.European,
            "not an European Option",
        )

        # Seasoned options deferred — same QL_FAIL as C++.
        qassert.require(
            args.start_date == Date(),
            "seasoned continuous geometric Asian options not yet supported - "
            "requires adjustment of forward price and variance based on "
            "accumulated geometric average",
        )

        payoff = args.payoff
        qassert.require(
            isinstance(payoff, PlainVanillaPayoff), "non-plain payoff given"
        )
        assert isinstance(payoff, PlainVanillaPayoff)

        process = self._process
        exercise = args.exercise.last_date()

        # Black volatility + variance + risk-free discount at maturity.
        volatility = process.black_volatility().black_vol(
            exercise, payoff.strike(), extrapolate=True
        )
        variance = process.black_volatility().black_variance(
            exercise, payoff.strike(), extrapolate=True
        )
        risk_free_discount = process.risk_free_rate().discount(exercise)

        rfdc = process.risk_free_rate().day_counter()
        divdc = process.dividend_yield().day_counter()
        voldc = process.black_volatility().day_counter()

        # Adjusted dividend yield: 0.5 * (r + q + sigma^2 / 6).
        rf_zero = process.risk_free_rate().zero_rate(
            exercise, Compounding.Continuous, Frequency.NoFrequency
        ).rate()
        div_zero = process.dividend_yield().zero_rate(
            exercise, Compounding.Continuous, Frequency.NoFrequency
        ).rate()
        dividend_yield = 0.5 * (
            rf_zero + div_zero + volatility * volatility / 6.0
        )

        t_q = divdc.year_fraction(
            process.dividend_yield().reference_date(), exercise
        )
        dividend_discount = math.exp(-dividend_yield * t_q)

        spot = process.state_variable().value()
        qassert.require(spot > 0.0, "negative or null underlying")

        forward = spot * dividend_discount / risk_free_discount

        # variance is divided by 3 (Kemna-Vorst).
        black = BlackCalculator(
            payoff, forward, math.sqrt(variance / 3.0), risk_free_discount
        )

        results.value = black.value()
        results.delta = black.delta(spot)
        results.gamma = black.gamma(spot)

        # dividend_rho is per the adjusted q; halved per Kemna-Vorst.
        results.dividend_rho = black.dividend_rho(t_q) / 2.0

        t_r = rfdc.year_fraction(
            process.risk_free_rate().reference_date(),
            args.exercise.last_date(),
        )
        results.rho = black.rho(t_r) + 0.5 * black.dividend_rho(t_q)

        t_v = voldc.year_fraction(
            process.black_volatility().reference_date(),
            args.exercise.last_date(),
        )
        results.vega = (
            black.vega(t_v) / math.sqrt(3.0)
            + black.dividend_rho(t_q) * volatility / 6.0
        )
        try:
            results.theta = black.theta(spot, t_v)
        except LibraryException:
            results.theta = None


__all__ = ["AnalyticContinuousGeometricAveragePriceAsianEngine"]
