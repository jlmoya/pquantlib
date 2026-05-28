"""AnalyticDiscreteGeometricAveragePriceAsianEngine — closed-form pricing.

# C++ parity:
# ql/pricingengines/asian/analytic_discr_geom_av_price.{hpp,cpp}
# (v1.42.1).

Closed-form pricing for a European discrete geometric average price
Asian option under Black-Scholes-Merton dynamics. Formula from Levy
(1997) "Asian Option" in Clewlow & Strickland's "Exotic Options: The
State of the Art" pp. 65-97.

The engine can also be used as a control variate for the arithmetic
average MC engine, so it does NOT enforce ``average_type == Geometric``
(per C++ comment).
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import Exercise
from pquantlib.instruments.asian_option import (
    AverageType,
    DiscreteAveragingAsianOptionArguments,
)
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.distributions.normal_distribution import NormalDistribution
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.black_calculator import BlackCalculator
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency

_QL_EPSILON = 1.0e-16


class AnalyticDiscreteGeometricAveragePriceAsianEngine(
    GenericEngine[DiscreteAveragingAsianOptionArguments, OneAssetOptionResults]
):
    """Levy 1997 closed-form engine for discrete geometric Asian.

    # C++ parity: ``AnalyticDiscreteGeometricAveragePriceAsianEngine``.
    """

    def __init__(self, process: GeneralizedBlackScholesProcess) -> None:
        super().__init__(
            DiscreteAveragingAsianOptionArguments(), OneAssetOptionResults()
        )
        self._process: GeneralizedBlackScholesProcess = process
        process.register_with(self)

    def calculate(self) -> None:  # noqa: PLR0915
        """Discrete geometric Asian closed form.

        # C++ parity:
        # ``AnalyticDiscreteGeometricAveragePriceAsianEngine::calculate``.
        # (Note: C++ doesn't enforce Geometric average_type because
        # the engine doubles as a control variate.)
        """
        args = self._arguments
        results = self._results

        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        qassert.require(
            args.exercise.type() == Exercise.Type.European,
            "not an European Option",
        )

        if args.average_type == AverageType.Geometric:
            qassert.require(
                args.running_accumulator is not None
                and args.running_accumulator > 0.0,
                f"positive running product required: "
                f"{args.running_accumulator} not allowed",
            )
            assert args.running_accumulator is not None
            running_log = math.log(args.running_accumulator)
            past_fixings = args.past_fixings if args.past_fixings is not None else 0
        else:
            # Used as control variate — running_log is harmless.
            running_log = 1.0
            past_fixings = 0

        payoff = args.payoff
        qassert.require(
            isinstance(payoff, PlainVanillaPayoff), "non-plain payoff given"
        )
        assert isinstance(payoff, PlainVanillaPayoff)

        process = self._process
        reference_date = process.risk_free_rate().reference_date()
        rfdc = process.risk_free_rate().day_counter()
        divdc = process.dividend_yield().day_counter()
        voldc = process.black_volatility().day_counter()

        fixing_times: list[float] = []
        for d in args.fixing_dates:
            if d >= reference_date:
                fixing_times.append(voldc.year_fraction(reference_date, d))

        remaining_fixings = len(fixing_times)
        number_of_fixings = past_fixings + remaining_fixings
        n_real = float(number_of_fixings)

        past_weight = past_fixings / n_real
        future_weight = 1.0 - past_weight

        time_sum = sum(fixing_times)

        vola = process.black_volatility().black_vol(
            args.exercise.last_date(), payoff.strike(), extrapolate=True
        )

        # temp = sum_{i = pastFixings+1..number_of_fixings-1} t[i] * (N - i)
        # (C++ uses 0-based: i from pastFixings+1 to numberOfFixings-1 in
        # the OUTER index; the INNER access is fixingTimes[i-pastFixings-1].)
        # Translating literally to Python.
        temp = 0.0
        for i in range(past_fixings + 1, number_of_fixings):
            temp += fixing_times[i - past_fixings - 1] * (n_real - float(i))

        variance = vola * vola / (n_real * n_real) * (time_sum + 2.0 * temp)
        dsig_g_dsig = math.sqrt(time_sum + 2.0 * temp) / n_real
        sig_g = vola * dsig_g_dsig
        dmu_g_dsig = -(vola * time_sum) / n_real

        ex_date = args.exercise.last_date()
        dividend_rate = process.dividend_yield().zero_rate(
            ex_date, Compounding.Continuous, Frequency.NoFrequency
        ).rate()
        risk_free_rate = process.risk_free_rate().zero_rate(
            ex_date, Compounding.Continuous, Frequency.NoFrequency
        ).rate()
        nu = risk_free_rate - dividend_rate - 0.5 * vola * vola

        s = process.state_variable().value()
        qassert.require(s > 0.0, "positive underlying value required")

        m = past_fixings if past_fixings > 0 else 1
        mu_g = (
            past_weight * running_log / float(m)
            + future_weight * math.log(s)
            + nu * time_sum / n_real
        )
        forward_price = math.exp(mu_g + variance / 2.0)

        risk_free_discount = process.risk_free_rate().discount(ex_date)

        black = BlackCalculator(
            payoff, forward_price, math.sqrt(variance), risk_free_discount
        )

        results.value = black.value()
        results.delta = (
            future_weight * black.delta(forward_price) * forward_price / s
        )
        results.gamma = (
            forward_price * future_weight / (s * s)
        ) * (
            black.gamma(forward_price) * future_weight * forward_price
            - past_weight * black.delta(forward_price)
        )

        cnd = CumulativeNormalDistribution()
        nd = NormalDistribution()
        if sig_g > _QL_EPSILON:
            x_1 = (mu_g - math.log(payoff.strike()) + variance) / sig_g
            nx_1 = cnd(x_1)
            nx_1_pdf = nd(x_1)
        else:
            nx_1 = 1.0 if mu_g > math.log(payoff.strike()) else 0.0
            nx_1_pdf = 0.0
        results.vega = (
            forward_price
            * risk_free_discount
            * ((dmu_g_dsig + sig_g * dsig_g_dsig) * nx_1 + nx_1_pdf * dsig_g_dsig)
        )
        if payoff.option_type() == OptionType.Put:
            results.vega -= (
                risk_free_discount
                * forward_price
                * (dmu_g_dsig + sig_g * dsig_g_dsig)
            )

        t_rho = rfdc.year_fraction(
            process.risk_free_rate().reference_date(), args.exercise.last_date()
        )
        results.rho = (
            black.rho(t_rho) * time_sum / (n_real * t_rho)
            - (t_rho - time_sum / n_real) * results.value
        )

        t_div = divdc.year_fraction(
            process.dividend_yield().reference_date(),
            args.exercise.last_date(),
        )
        results.dividend_rho = (
            black.dividend_rho(t_div) * time_sum / (n_real * t_div)
        )

        results.strike_sensitivity = black.strike_sensitivity()

        # ``blackScholesTheta`` shortcut from ``pricingengines/greeks.hpp``:
        # theta = r*value - (r-q)*u*delta - 0.5*v^2*u^2*gamma, where
        # u = spot, v = local vol(0, u). C++:
        #   Real blackScholesTheta(p, value, delta, gamma) {
        #     Real u = p->stateVariable()->value();
        #     Rate r = p->riskFreeRate()->zeroRate(0.0, Continuous);
        #     Rate q = p->dividendYield()->zeroRate(0.0, Continuous);
        #     Volatility v = p->localVolatility()->localVol(0.0, u);
        #     return r*value - (r-q)*u*delta - 0.5*v*v*u*u*gamma;
        #   }
        try:
            u = s
            r = process.risk_free_rate().zero_rate(
                0.0, Compounding.Continuous, Frequency.NoFrequency
            ).rate()
            q = process.dividend_yield().zero_rate(
                0.0, Compounding.Continuous, Frequency.NoFrequency
            ).rate()
            v = process.local_volatility().local_vol_at_time(
                0.0, u, extrapolate=True
            )
            assert results.delta is not None
            assert results.gamma is not None
            assert results.value is not None
            results.theta = (
                r * results.value
                - (r - q) * u * results.delta
                - 0.5 * v * v * u * u * results.gamma
            )
        except LibraryException:
            results.theta = None


__all__ = ["AnalyticDiscreteGeometricAveragePriceAsianEngine"]
