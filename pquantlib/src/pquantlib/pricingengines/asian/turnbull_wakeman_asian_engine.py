"""TurnbullWakemanAsianEngine — two-moment matching for arithmetic Asians.

# C++ parity: ql/pricingengines/asian/turnbullwakemanasianengine.{hpp,cpp}
# (v1.43) — ``class TurnbullWakemanAsianEngine :
# public DiscreteAveragingAsianOption::engine``.

References: Iain Clark, *Commodity Option Pricing*, §2.7.4; E.G. Haug,
*Option Pricing Formulas*, 2nd ed., pp. 192-202. Parts of the C++
implementation follow the ``CommodityAveragePriceOptionAnalyticalEngine``
of Open Source Risk Engine.

The arithmetic average of the remaining fixings is replaced by a lognormal
with the same first two moments:

* ``E[A] = (1/m) * sum_i F_i`` over the *future* fixings, where ``m`` is
  the total (past + future) fixing count;
* ``E[A^2] = (1/m^2) * sum_i (F_i^2 e^{v_i} + 2 F_i sum_{j<i} F_j e^{v_j})``
  with ``v_i = blackVariance(t_i, effective_strike)``;
* ``sigma = sqrt(log(E[A^2]/E[A]^2) / t_n)``, where ``t_n`` is the time of
  the **last fixing date** — not of the exercise date.

Three things the cross-validation pins:

* ``accrued = running_accumulator / (past_fixings + future_fixings)`` —
  the running sum is divided by the **total** fixing count, not by the
  number of past fixings.
* Each fixing gets its **own** variance off the surface, so a sloping vol
  term structure moves the price; an engine that read a single vol at
  expiry reproduces the flat-vol cases and fails the sloping ones.
* When ``effective_strike = strike - accrued <= 0`` the option is already
  guaranteed in resp. out of the money and a separate closed form applies:
  calls get ``discount * (S_A_hat - strike)`` plus a delta, puts get hard
  zeros, both get ``gamma = 0``, and the moment-matching additional
  results (``forward``, ``exp_A_2``, ``tte``, ``sigma``, ``times``,
  ``spotVols``, ``forwards``) are **not** published.

``vega`` / ``rho`` / ``theta`` / ``dividend_rho`` are never assigned on any
path.
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
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.black_calculator import BlackCalculator
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)


class TurnbullWakemanAsianEngine(
    GenericEngine[DiscreteAveragingAsianOptionArguments, OneAssetOptionResults]
):
    """Turnbull-Wakeman two-moment matching engine.

    # C++ parity: ``TurnbullWakemanAsianEngine(process)``.
    """

    def __init__(self, process: GeneralizedBlackScholesProcess) -> None:
        super().__init__(
            DiscreteAveragingAsianOptionArguments(), OneAssetOptionResults()
        )
        self._process: GeneralizedBlackScholesProcess = process
        process.register_with(self)

    def calculate(self) -> None:  # noqa: PLR0915
        """# C++ parity: ``TurnbullWakemanAsianEngine::calculate``."""
        args = self._arguments
        results = self._results
        process = self._process

        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        qassert.require(
            args.exercise.type() == Exercise.Type.European, "not a European Option"
        )
        qassert.require(
            args.average_type == AverageType.Arithmetic,
            "must be Arithmetic Average::Type",
        )

        past_fixings = args.past_fixings if args.past_fixings is not None else 0
        future_fixings = len(args.fixing_dates)
        accrued_average = 0.0
        if past_fixings != 0:
            assert args.running_accumulator is not None
            # Divided by the TOTAL fixing count, not by past_fixings.
            accrued_average = args.running_accumulator / (past_fixings + future_fixings)
        results.additional_results["accrued"] = accrued_average

        discount = process.risk_free_rate().discount(args.exercise.last_date())
        results.additional_results["discount"] = discount

        qassert.require(
            isinstance(args.payoff, PlainVanillaPayoff), "non-plain payoff given"
        )
        assert isinstance(args.payoff, PlainVanillaPayoff)
        payoff: PlainVanillaPayoff = args.payoff

        # The surface is read at the effective strike, i.e. net of the accrued
        # part of the average.
        effective_strike = payoff.strike() - accrued_average
        results.additional_results["strike"] = payoff.strike()
        results.additional_results["effective_strike"] = effective_strike

        m = future_fixings + past_fixings

        if effective_strike <= 0.0:
            # Guaranteed exercise (call) or permanent OTM (put): Haug 2nd ed.,
            # p. 193. No moment matching, and no further additional results.
            if payoff.option_type() == OptionType.Call:
                spot = process.state_variable().value()
                s_a_hat = accrued_average
                for fd in args.fixing_dates:
                    s_a_hat += (
                        spot
                        * process.dividend_yield().discount(fd)
                        / process.risk_free_rate().discount(fd)
                    ) / m
                results.value = discount * (s_a_hat - payoff.strike())
                results.delta = discount * (s_a_hat - accrued_average) / spot
            elif payoff.option_type() == OptionType.Put:
                results.value = 0.0
                results.delta = 0.0
            results.gamma = 0.0
            return

        qassert.require(
            effective_strike > 0.0, "expected effectiveStrike to be positive"
        )

        expected_average = 0.0
        forwards: list[float] = []
        times: list[float] = []
        spot_vars: list[float] = []
        spot_vols: list[float] = []
        spot = process.state_variable().value()

        for fd in args.fixing_dates:
            dividend_discount = process.dividend_yield().discount(fd)
            risk_free_discount_for_fwd = process.risk_free_rate().discount(fd)

            forwards.append(spot * dividend_discount / risk_free_discount_for_fwd)
            times.append(process.black_volatility().time_from_reference(fd))

            spot_vars.append(
                process.black_volatility().black_variance_at_time(
                    times[-1], effective_strike
                )
            )
            spot_vols.append(math.sqrt(spot_vars[-1] / times[-1]))

            expected_average += forwards[-1]

        expected_average /= m

        expected_average_2 = 0.0
        n = len(forwards)
        for i in range(n):
            expected_average_2 += forwards[i] * forwards[i] * math.exp(spot_vars[i])
            for j in range(i):
                expected_average_2 += (
                    2 * forwards[i] * forwards[j] * math.exp(spot_vars[j])
                )
        expected_average_2 /= m * m

        # tn is the time of the LAST FIXING, not of the exercise date.
        tn = times[-1]
        sigma = math.sqrt(
            math.log(expected_average_2 / (expected_average * expected_average)) / tn
        )

        black = BlackCalculator.from_type_strike(
            payoff.option_type(),
            effective_strike,
            expected_average,
            sigma * math.sqrt(tn),
            discount,
        )

        results.value = black.value()
        results.delta = black.delta(spot)
        results.gamma = black.gamma(spot)

        results.additional_results["forward"] = expected_average
        results.additional_results["exp_A_2"] = expected_average_2
        results.additional_results["tte"] = tn
        results.additional_results["sigma"] = sigma
        results.additional_results["times"] = times
        results.additional_results["spotVols"] = spot_vols
        results.additional_results["forwards"] = forwards


__all__ = ["TurnbullWakemanAsianEngine"]
