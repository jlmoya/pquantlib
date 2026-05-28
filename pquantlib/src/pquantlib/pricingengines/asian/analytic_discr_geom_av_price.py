"""AnalyticDiscreteGeometricAveragePriceAsianEngine.

# C++ parity: ql/pricingengines/asian/analytic_discr_geom_av_price.{hpp,cpp}
# (v1.42.1).

Closed-form pricer for a European discrete-geometric-average price
Asian option under BSM (constant vol).  Uses E. Levy's 1997 formula
(*Asian Option* in *Exotic Options: The State of the Art*, Clewlow
& Strickland, pp. 65-97).

The engine is also used as the analytic *control-variate* anchor for
the arithmetic MC engine (``MCDiscreteArithmeticAveragePriceEngine``)
— the geometric-average payoff is a tractable function of the same
underlying paths, so subtracting its (deterministic) closed-form mean
removes the bulk of the variance from the arithmetic MC payoff.

Greeks: NPV + delta + gamma + vega + rho + dividend_rho + theta +
strike_sensitivity, all per the C++ formula.  ``theta_per_day`` is
left to the C++-mirror ``BlackScholesTheta`` helper — *not* ported in
this cluster because the helper carve-out (see L3 carve-outs) requires
a few additional pieces; we set ``theta = theta_per_day = None``.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.asian_option import DiscreteAveragingAsianOptionArguments
from pquantlib.instruments.average_type import AverageType
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

# C++ uses QL_EPSILON = std::numeric_limits<Real>::epsilon() ≈ 2.22e-16.
_QL_EPSILON = 2.220446049250313e-16


class AnalyticDiscreteGeometricAveragePriceAsianEngine(
    GenericEngine[DiscreteAveragingAsianOptionArguments, OneAssetOptionResults]
):
    """E. Levy (1997) closed-form pricer for discrete-geometric-average Asian.

    # C++ parity: ``AnalyticDiscreteGeometricAveragePriceAsianEngine``
    # (analytic_discr_geom_av_price.{hpp,cpp}).
    """

    def __init__(self, process: GeneralizedBlackScholesProcess) -> None:
        super().__init__(
            DiscreteAveragingAsianOptionArguments(), OneAssetOptionResults()
        )
        self._process: GeneralizedBlackScholesProcess = process
        process.register_with(self)

    def calculate(self) -> None:  # noqa: PLR0915
        """Compute NPV + Greeks via Levy 1997.

        # C++ parity: ``AnalyticDiscreteGeometricAveragePriceAsianEngine::calculate``
        # (analytic_discr_geom_av_price.cpp:39-165).

        Linear flow that mirrors the C++ method literally; suppressing
        PLR0915 because splitting would obscure the parity reading.
        """
        args = self._arguments
        results = self._results
        process = self._process

        # NB: cannot QL_REQUIRE(averageType == Geometric) — C++ comment
        # notes this engine is also used as a CV for the arithmetic
        # MC engine.
        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        qassert.require(
            args.exercise.type() == Exercise.Type.European,
            "not an European Option",
        )

        if args.average_type == AverageType.Geometric:
            qassert.require(args.running_accumulator is not None, "null running product")
            assert args.running_accumulator is not None
            qassert.require(
                args.running_accumulator > 0.0,
                f"positive running product required: {args.running_accumulator} not allowed",
            )
            running_log = math.log(args.running_accumulator)
            qassert.require(args.past_fixings is not None, "null past-fixing count")
            assert args.past_fixings is not None
            past_fixings = args.past_fixings
        else:
            # CV use — use placeholder running_log = 1.0 (C++ same value).
            running_log = 1.0
            past_fixings = 0

        payoff = args.payoff
        qassert.require(isinstance(payoff, PlainVanillaPayoff), "non-plain payoff given")
        assert isinstance(payoff, PlainVanillaPayoff)

        rfdc = process.risk_free_rate().day_counter()
        divdc = process.dividend_yield().day_counter()
        voldc = process.black_volatility().day_counter()
        reference_date = process.risk_free_rate().reference_date()

        fixing_times: list[float] = []
        for fd in args.fixing_dates:
            if fd >= reference_date:
                fixing_times.append(voldc.year_fraction(reference_date, fd))

        remaining_fixings = len(fixing_times)
        number_of_fixings = past_fixings + remaining_fixings
        n = float(number_of_fixings)

        past_weight = past_fixings / n
        future_weight = 1.0 - past_weight

        time_sum = sum(fixing_times)

        vola = process.black_volatility().black_vol(
            args.exercise.last_date(), payoff.strike()
        )
        temp = 0.0
        for i in range(past_fixings + 1, number_of_fixings):
            temp += fixing_times[i - past_fixings - 1] * (n - i)
        variance = vola * vola / n / n * (time_sum + 2.0 * temp)
        dsig_g_dsig = math.sqrt(time_sum + 2.0 * temp) / n
        sig_g = vola * dsig_g_dsig
        dmu_g_dsig = -(vola * time_sum) / n

        ex_date = args.exercise.last_date()
        dividend_rate = process.dividend_yield().zero_rate(
            ex_date,
            Compounding.Continuous,
            Frequency.NoFrequency,
            False,
            result_day_counter=divdc,
        ).rate()
        risk_free_rate = process.risk_free_rate().zero_rate(
            ex_date,
            Compounding.Continuous,
            Frequency.NoFrequency,
            False,
            result_day_counter=rfdc,
        ).rate()
        nu = risk_free_rate - dividend_rate - 0.5 * vola * vola

        s = process.state_variable().value()
        qassert.require(s > 0.0, "positive underlying value required")

        m = 1 if past_fixings == 0 else past_fixings
        mu_g = past_weight * running_log / m + future_weight * math.log(s) + nu * time_sum / n
        forward_price = math.exp(mu_g + variance / 2.0)

        risk_free_discount = process.risk_free_rate().discount(args.exercise.last_date())

        black = BlackCalculator(payoff, forward_price, math.sqrt(variance), risk_free_discount)

        results.value = black.value()
        results.delta = future_weight * black.delta(forward_price) * forward_price / s
        results.gamma = (
            forward_price
            * future_weight
            / (s * s)
            * (black.gamma(forward_price) * future_weight * forward_price - past_weight * black.delta(forward_price))
        )

        if sig_g > _QL_EPSILON:
            x_1 = (mu_g - math.log(payoff.strike()) + variance) / sig_g
            cnd = CumulativeNormalDistribution()
            nd = NormalDistribution()
            nx_1_upper = cnd(x_1)
            nx_1_lower = nd(x_1)
        else:
            nx_1_upper = 1.0 if mu_g > math.log(payoff.strike()) else 0.0
            nx_1_lower = 0.0

        vega = forward_price * risk_free_discount * (
            (dmu_g_dsig + sig_g * dsig_g_dsig) * nx_1_upper + nx_1_lower * dsig_g_dsig
        )
        if payoff.option_type() == OptionType.Put:
            vega -= risk_free_discount * forward_price * (dmu_g_dsig + sig_g * dsig_g_dsig)
        results.vega = vega

        t_rho = rfdc.year_fraction(
            process.risk_free_rate().reference_date(), args.exercise.last_date()
        )
        results.rho = black.rho(t_rho) * time_sum / (n * t_rho) - (t_rho - time_sum / n) * results.value

        t_div = divdc.year_fraction(
            process.dividend_yield().reference_date(), args.exercise.last_date()
        )
        results.dividend_rho = black.dividend_rho(t_div) * time_sum / (n * t_div)

        results.strike_sensitivity = black.strike_sensitivity()

        # blackScholesTheta carve-out: not ported in L5-C; consumers that
        # need it can call ``results.theta = blackScholesTheta(...)``
        # externally once that helper lands. C++ sets it via the helper.
        results.theta = None
        results.theta_per_day = None


__all__ = ["AnalyticDiscreteGeometricAveragePriceAsianEngine"]
