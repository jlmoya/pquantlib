"""AnalyticCliquetEngine — closed-form cliquet (ratchet) option engine.

# C++ parity: ql/pricingengines/cliquet/analyticcliquetengine.{hpp,cpp}
# (v1.43) — ``class AnalyticCliquetEngine : public CliquetOption::engine``.

A cliquet is a strip of forward-starting options: on each reset date the
strike is re-struck at ``moneyness * S(reset)``. Under Black-Scholes the
strip decomposes into a sum of independent forward-start Blacks, one per
consecutive pair of dates in ``reset_dates + [expiry]``.

Per period ``[t_{i-1}, t_i]`` the engine prices a plain Black on

* forward ``S * qDF(t_{i-1}, t_i) / rDF(t_{i-1}, t_i)`` — both *forward*
  discount factors, i.e. ratios across the period;
* strike ``S * moneyness`` — the level the strike would be re-struck at if
  the spot never moved;
* variance ``blackForwardVariance(t_{i-1}, t_i, strike)``;
* discount ``rDF(t_{i-1}, t_i)`` — again the forward risk-free factor,

and weights the result by ``qDF(0, t_{i-1})`` — a *dividend* discount from
today to the period start, not a risk-free one. That asymmetry (dividend
weight outside, risk-free discount inside) is the thing a port most often
gets backwards; the sibling
:class:`~pquantlib.pricingengines.cliquet.analytic_performance_engine.AnalyticPerformanceEngine`
has it exactly the other way round.

The engine assigns ``gamma`` a hard ``0.0`` rather than leaving it unset,
and reads three separate day counters (risk-free for rho, dividend for
dividend-rho, vol for vega). It refuses started (accrued coupon / last
fixing) and capped/floored options.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.cliquet_option import CliquetOptionArguments
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.payoffs import PercentageStrikePayoff, PlainVanillaPayoff
from pquantlib.pricingengines.black_calculator import BlackCalculator
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency


def check_unsupported_cliquet_features(args: CliquetOptionArguments) -> None:
    """Reject started and capped/floored cliquets.

    # C++ parity: the two ``QL_REQUIRE`` blocks shared verbatim by
    # ``AnalyticCliquetEngine::calculate`` and
    # ``AnalyticPerformanceEngine::calculate``.
    """
    qassert.require(
        args.accrued_coupon is None and args.last_fixing is None,
        "this engine cannot price options already started",
    )
    qassert.require(
        args.local_cap is None
        and args.local_floor is None
        and args.global_cap is None
        and args.global_floor is None,
        "this engine cannot price capped/floored options",
    )


def cliquet_reset_grid(args: CliquetOptionArguments) -> list[Date]:
    """``reset_dates`` with the exercise date appended.

    # C++ parity: ``resetDates.push_back(arguments_.exercise->lastDate())``.
    A one-reset cliquet therefore has TWO periods, not one.
    """
    assert args.exercise is not None
    return [*args.reset_dates, args.exercise.last_date()]


class AnalyticCliquetEngine(
    GenericEngine[CliquetOptionArguments, OneAssetOptionResults]
):
    """Closed-form cliquet engine.

    # C++ parity: ``AnalyticCliquetEngine(process)``.
    """

    def __init__(self, process: GeneralizedBlackScholesProcess) -> None:
        super().__init__(CliquetOptionArguments(), OneAssetOptionResults())
        self._process: GeneralizedBlackScholesProcess = process
        process.register_with(self)

    def calculate(self) -> None:
        """# C++ parity: ``AnalyticCliquetEngine::calculate``."""
        args = self._arguments
        results = self._results

        check_unsupported_cliquet_features(args)

        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        qassert.require(
            args.exercise.type() == Exercise.Type.European, "not an European option"
        )
        qassert.require(
            isinstance(args.payoff, PercentageStrikePayoff), "wrong payoff given"
        )
        assert isinstance(args.payoff, PercentageStrikePayoff)
        moneyness: PercentageStrikePayoff = args.payoff

        reset_dates = cliquet_reset_grid(args)

        process = self._process
        r_ts = process.risk_free_rate()
        q_ts = process.dividend_yield()
        vol_ts = process.black_volatility()

        underlying = process.state_variable().value()
        qassert.require(underlying > 0.0, "negative or null underlying")
        strike = underlying * moneyness.strike()
        payoff = PlainVanillaPayoff(moneyness.option_type(), strike)

        results.value = 0.0
        results.delta = 0.0
        results.gamma = 0.0
        results.theta = 0.0
        results.rho = 0.0
        results.dividend_rho = 0.0
        results.vega = 0.0

        rfdc = r_ts.day_counter()
        divdc = q_ts.day_counter()
        voldc = vol_ts.day_counter()

        for i in range(1, len(reset_dates)):
            start, end = reset_dates[i - 1], reset_dates[i]

            # The outer weight is a DIVIDEND discount from today to the period
            # start; the Black discount is the FORWARD risk-free factor.
            weight = q_ts.discount(start)
            discount = r_ts.discount(end) / r_ts.discount(start)
            q_discount = q_ts.discount(end) / q_ts.discount(start)
            forward = underlying * q_discount / discount
            variance = vol_ts.black_forward_variance(start, end, strike)

            black = BlackCalculator(payoff, forward, math.sqrt(variance), discount)

            results.value += weight * black.value()
            # The strike re-strikes with the spot, so the delta picks up the
            # strike sensitivity (BlackCalculator.beta) scaled by the moneyness.
            results.delta += weight * (
                black.delta(underlying) + moneyness.strike() * discount * black.beta()
            )
            results.gamma += 0.0
            # Cliquet theta uses the DIVIDEND curve's forward rate (the
            # performance engine uses the risk-free one).
            results.theta += (
                q_ts.forward_rate(
                    start, end, Compounding.Continuous, Frequency.NoFrequency, False, rfdc
                ).rate()
                * weight
                * black.value()
            )

            dt = rfdc.year_fraction(start, end)
            results.rho += weight * black.rho(dt)

            t = divdc.year_fraction(q_ts.reference_date(), start)
            dt = divdc.year_fraction(start, end)
            results.dividend_rho += weight * (black.dividend_rho(dt) - t * black.value())

            dt = voldc.year_fraction(start, end)
            results.vega += weight * black.vega(dt)


__all__ = [
    "AnalyticCliquetEngine",
    "check_unsupported_cliquet_features",
    "cliquet_reset_grid",
]
