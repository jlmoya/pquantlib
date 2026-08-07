"""AnalyticDiscreteGeometricAverageStrikeAsianEngine — Levy (1997) closed form.

# C++ parity: ql/pricingengines/asian/analytic_discr_geom_av_strike.{hpp,cpp}
# (v1.43) — ``class AnalyticDiscreteGeometricAverageStrikeAsianEngine :
# public DiscreteAveragingAsianOption::engine``.

Closed form for a European discrete geometric **average-strike** Asian:
the terminal spot is exchanged against the geometric mean of the fixings,
so the payoff is ``max(eta * (S_T - G_n), 0)``. Formula from E. Levy,
"Asian Option", in Clewlow & Strickland (eds.), *Exotic Options: The State
of the Art*, pp. 65-97.

Sibling of
:class:`~pquantlib.pricingengines.asian.analytic_discrete_geometric_average_price_engine.AnalyticDiscreteGeometricAveragePriceAsianEngine`
(the average-**price** variant), but with four differences a port that
copies the sibling will get wrong:

* Fixing times are measured from ``fixing_dates[0]`` — **not** from the
  curve reference date — using the vol surface's day counter, while
  ``residual_time`` runs from ``fixing_dates[past_fixings]`` to expiry on
  the risk-free curve's day counter.
* The volatility is read at strike = the **underlying**, not at the
  payoff's strike.
* The payoff **strike is never used at all**; only its option type is.
  That is the point of an average-strike option, and it means two options
  differing only in strike price identically.
* Only ``value`` is assigned — the sibling fills the whole greek block.

``past_fixings != 0`` is rejected outright ("past fixings currently not
managed") even though the surrounding arithmetic carries the past-weight
terms.

The engine can divide by zero: with a single fixing that coincides with
expiry, ``sigma_sum_2`` is exactly zero and C++ returns NaN. This port
reproduces that (see :func:`_cpp_divide`) rather than raising, so callers
see the same degenerate answer as C++.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.asian_option import (
    AverageType,
    DiscreteAveragingAsianOptionArguments,
)
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency


def _cpp_divide(numerator: float, denominator: float) -> float:
    """IEEE-754 division with C++ semantics.

    C++ evaluates ``x / 0.0`` to +-infinity, and ``0.0 / 0.0`` to NaN,
    without raising; Python raises ``ZeroDivisionError`` instead. This
    engine's ``sqrt(sigma_sum_2)`` denominator is exactly zero for a single
    fixing landing on the expiry date, and C++ returns NaN there — a port
    that raised, or that special-cased the degenerate input into some
    "nice" number, would not agree with the reference.
    """
    if denominator != 0.0:
        return numerator / denominator
    if numerator == 0.0:
        return math.nan
    return math.copysign(math.inf, numerator) * math.copysign(1.0, denominator)


class AnalyticDiscreteGeometricAverageStrikeAsianEngine(
    GenericEngine[DiscreteAveragingAsianOptionArguments, OneAssetOptionResults]
):
    """Levy (1997) closed-form discrete geometric average-strike engine.

    # C++ parity: ``AnalyticDiscreteGeometricAverageStrikeAsianEngine(process)``.
    """

    def __init__(self, process: GeneralizedBlackScholesProcess) -> None:
        super().__init__(
            DiscreteAveragingAsianOptionArguments(), OneAssetOptionResults()
        )
        self._process: GeneralizedBlackScholesProcess = process
        process.register_with(self)

    def calculate(self) -> None:  # noqa: PLR0915
        """# C++ parity: ``...AverageStrikeAsianEngine::calculate``.

        Kept as one straight-line method: it follows the C++ body
        operation-for-operation, and splitting it would hide the ordering the
        cross-validation depends on.
        """
        args = self._arguments
        results = self._results

        qassert.require(
            args.average_type == AverageType.Geometric, "not a geometric average option"
        )
        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        qassert.require(
            args.exercise.type() == Exercise.Type.European, "not an European option"
        )

        qassert.require(
            args.running_accumulator is not None and args.running_accumulator > 0.0,
            f"positive running product required: {args.running_accumulator}not allowed",
        )
        assert args.running_accumulator is not None
        running_log = math.log(args.running_accumulator)
        past_fixings = args.past_fixings if args.past_fixings is not None else 0
        qassert.require(past_fixings == 0, "past fixings currently not managed")

        qassert.require(
            isinstance(args.payoff, PlainVanillaPayoff), "non-plain payoff given"
        )
        assert isinstance(args.payoff, PlainVanillaPayoff)
        payoff: PlainVanillaPayoff = args.payoff

        process = self._process
        r_ts = process.risk_free_rate()
        q_ts = process.dividend_yield()
        vol_ts = process.black_volatility()

        rfdc = r_ts.day_counter()
        divdc = q_ts.day_counter()
        voldc = vol_ts.day_counter()

        # C++ indexes ``arguments_.fixingDates[0]`` without checking — an empty
        # schedule is undefined behaviour there. Raising is the only sane
        # translation, and it cannot mask a divergence: no reachable C++ input
        # reaches this line with a defined result.
        qassert.require(len(args.fixing_dates) > 0, "no fixing dates given")
        first_fixing = args.fixing_dates[0]
        # Times run from the FIRST FIXING, not from the curve reference date.
        fixing_times = [
            voldc.year_fraction(first_fixing, d)
            for d in args.fixing_dates
            if d >= first_fixing
        ]

        remaining_fixings = len(fixing_times)
        number_of_fixings = past_fixings + remaining_fixings
        n_real = float(number_of_fixings)

        past_weight = past_fixings / n_real
        future_weight = 1.0 - past_weight

        # C++ std::accumulate with a Real(0.0) seed: a sequential left fold, so
        # the rounding matches sum(..., 0.0) rather than math.fsum.
        time_sum = sum(fixing_times, 0.0)

        ex_date = args.exercise.last_date()
        residual_time = rfdc.year_fraction(args.fixing_dates[past_fixings], ex_date)

        underlying = process.state_variable().value()
        qassert.require(underlying > 0.0, "positive underlying value required")

        # The vol is read at strike = the UNDERLYING, not at the payoff strike.
        volatility = vol_ts.black_vol(ex_date, underlying)

        dividend_rate = q_ts.zero_rate(
            ex_date, Compounding.Continuous, Frequency.NoFrequency, False, divdc
        ).rate()
        risk_free_rate = r_ts.zero_rate(
            ex_date, Compounding.Continuous, Frequency.NoFrequency, False, rfdc
        ).rate()

        nu = risk_free_rate - dividend_rate - 0.5 * volatility * volatility

        temp = 0.0
        for i in range(past_fixings + 1, number_of_fixings):
            temp += fixing_times[i - past_fixings - 1] * (n_real - float(i))
        variance = volatility * volatility / n_real / n_real * (time_sum + 2.0 * temp)
        covariance_term = volatility * volatility / n_real * time_sum
        sigma_sum_2 = (
            variance + volatility * volatility * residual_time - 2.0 * covariance_term
        )

        m = past_fixings if past_fixings != 0 else 1
        running_log_average = running_log / m

        mu_g = (
            past_weight * running_log_average
            + future_weight * math.log(underlying)
            + nu * time_sum / n_real
        )

        f = CumulativeNormalDistribution()

        sqrt_sigma_sum_2 = math.sqrt(sigma_sum_2)
        y1 = _cpp_divide(
            math.log(underlying)
            + (risk_free_rate - dividend_rate) * residual_time
            - mu_g
            - variance / 2.0
            + sigma_sum_2 / 2.0,
            sqrt_sigma_sum_2,
        )
        y2 = y1 - sqrt_sigma_sum_2

        discounted_geo = math.exp(
            mu_g + variance / 2.0 - risk_free_rate * residual_time
        )
        discounted_spot = underlying * math.exp(-dividend_rate * residual_time)

        if payoff.option_type() == OptionType.Call:
            results.value = discounted_spot * f(y1) - discounted_geo * f(y2)
        elif payoff.option_type() == OptionType.Put:
            results.value = -discounted_spot * f(-y1) + discounted_geo * f(-y2)
        else:
            qassert.fail("invalid option type")


__all__ = ["AnalyticDiscreteGeometricAverageStrikeAsianEngine"]
