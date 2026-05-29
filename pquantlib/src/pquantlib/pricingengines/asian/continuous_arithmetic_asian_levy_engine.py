"""ContinuousArithmeticAsianLevyEngine — Levy 1992 closed-form.

# C++ parity:
# ql/pricingengines/asian/continuousarithmeticasianlevyengine.{hpp,cpp}
# (v1.42.1).

Levy 1992 closed-form approximation for continuously-averaged
arithmetic Asian options under Black-Scholes. The approximation
matches the first two moments of the (log) arithmetic average and
applies a Black-Scholes call/put on the effective forward.

Formula highlights:

* ``Se``  — effective forward, distinguishing the b → 0 limit.
* ``X``   — strike adjusted for the seasoned-average accrual.
* ``m``   — first moment of the integrated forward.
* ``M``   — second moment (variance of the integrated forward).
* ``D``   — ``M / T^2`` (squared running average forward).
* ``V``   — log-variance of the arithmetic average.

The engine accepts a ``Handle<Quote>`` for the current running average
(used in the seasoned branch when the option's ``start_date`` lies
before the reference date). When the option is unseasoned the running
average is ignored.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.asian_option import (
    AverageType,
    ContinuousAveragingAsianOptionArguments,
)
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.constants import QL_EPSILON
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.payoffs import OptionType, StrikedTypePayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.quote import Quote
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency


class ContinuousArithmeticAsianLevyEngine(
    GenericEngine[ContinuousAveragingAsianOptionArguments, OneAssetOptionResults]
):
    """Levy 1992 closed-form for continuously-averaged arithmetic Asians.

    # C++ parity: ``ContinuousArithmeticAsianLevyEngine``.
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        current_average: Quote,
        start_date: Date | None = None,
    ) -> None:
        super().__init__(
            ContinuousAveragingAsianOptionArguments(), OneAssetOptionResults()
        )
        self._process: GeneralizedBlackScholesProcess = process
        self._current_average: Quote = current_average
        # ``start_date`` is the (deprecated) constructor parameter; the
        # primary path is to read it off the option arguments.
        self._start_date: Date = start_date if start_date is not None else Date()
        process.register_with(self)
        current_average.register_with(self)

    def calculate(self) -> None:
        """Compute the Levy 1992 NPV.

        # C++ parity:
        # ``ContinuousArithmeticAsianLevyEngine::calculate``.
        """
        args = self._arguments
        results = self._results

        qassert.require(
            args.average_type == AverageType.Arithmetic,
            "not an Arithmetic average option",
        )
        assert args.exercise is not None
        qassert.require(
            args.exercise.type() == Exercise.Type.European,
            "not an European Option",
        )

        # Prefer the option's start date; fall back to the ctor parameter.
        start_date = args.start_date if args.start_date != Date() else self._start_date
        qassert.require(start_date != Date(), "start date not provided")
        qassert.require(
            start_date <= self._process.risk_free_rate().reference_date(),
            "start date must be earlier than or equal to reference date",
        )

        rfdc = self._process.risk_free_rate().day_counter()
        spot = self._process.state_variable().value()

        payoff = args.payoff
        qassert.require(
            isinstance(payoff, StrikedTypePayoff), "non-plain payoff given"
        )
        assert isinstance(payoff, StrikedTypePayoff)

        # Original time to maturity (start → expiry).
        maturity = args.exercise.last_date()
        t_full = rfdc.year_fraction(start_date, maturity)
        # Remaining time to maturity (today → expiry).
        t_remaining = rfdc.year_fraction(
            self._process.risk_free_rate().reference_date(), maturity
        )

        strike = payoff.strike()
        volatility = self._process.black_volatility().black_vol(
            maturity, strike, extrapolate=True
        )

        risk_free_rate = self._process.risk_free_rate().zero_rate(
            maturity, Compounding.Continuous, Frequency.NoFrequency
        ).rate()
        dividend_yield = self._process.dividend_yield().zero_rate(
            maturity, Compounding.Continuous, Frequency.NoFrequency
        ).rate()
        b = risk_free_rate - dividend_yield

        # Effective forward.
        if abs(b) > 1000 * QL_EPSILON:
            se = (spot / (t_full * b)) * (
                math.exp((b - risk_free_rate) * t_remaining)
                - math.exp(-risk_free_rate * t_remaining)
            )
        else:
            se = spot * t_remaining / t_full * math.exp(-risk_free_rate * t_remaining)

        # Strike adjusted for the seasoned accrual.
        if t_remaining < t_full:
            qassert.require(
                self._current_average.is_valid(),
                "current average required for seasoned option",
            )
            x = strike - ((t_full - t_remaining) / t_full) * self._current_average.value()
        else:
            x = strike

        m = (
            (math.exp(b * t_remaining) - 1.0) / b
            if abs(b) > 1000 * QL_EPSILON
            else t_remaining
        )

        big_m = (2.0 * spot * spot / (b + volatility * volatility)) * (
            (
                (math.exp((2.0 * b + volatility * volatility) * t_remaining) - 1.0)
                / (2.0 * b + volatility * volatility)
            )
            - m
        )

        d_var = big_m / (t_full * t_full)
        v_log_var = math.log(d_var) - 2.0 * (
            risk_free_rate * t_remaining + math.log(se)
        )

        d1 = (1.0 / math.sqrt(v_log_var)) * ((math.log(d_var) / 2.0) - math.log(x))
        d2 = d1 - math.sqrt(v_log_var)

        n = CumulativeNormalDistribution()

        results.reset()
        if payoff.option_type() == OptionType.Call:
            results.value = se * n(d1) - x * math.exp(-risk_free_rate * t_remaining) * n(d2)
        else:
            results.value = (
                se * n(d1)
                - x * math.exp(-risk_free_rate * t_remaining) * n(d2)
                - se
                + x * math.exp(-risk_free_rate * t_remaining)
            )

    def update(self) -> None:
        self.notify_observers()


__all__ = ["ContinuousArithmeticAsianLevyEngine"]
