"""AnalyticEuropeanMargrabeEngine — Margrabe 1978 closed form.

# C++ parity:
# ql/pricingengines/exotic/analyticeuropeanmargrabeengine.{hpp,cpp} (v1.43).

Reference: W. Margrabe, "The Value of an Option to Exchange One Asset
for Another", Journal of Finance 33 (March 1978), 177-186.

With ``F_i`` the forward of asset ``i``, ``sigma^2 = var1 + var2 -
2*rho*sd1*sd2`` the variance of the exchange ratio and ``D`` the
risk-free discount factor::

    d1    = (log(q1*F1 / (q2*F2)) + sigma^2/2) / sigma
    d2    = d1 - sigma
    value = D * (q1*F1*N(d1) - q2*F2*N(d2))

The engine fills ``value``, ``delta1``, ``delta2``, ``gamma1``,
``gamma2``, ``theta`` and ``rho`` (which is identically 0.0 — the
exchange ratio is rate-independent).  It deliberately leaves ``delta``,
``gamma``, ``vega`` and ``dividend_rho`` unset, so those accessors raise
— same as C++, where they stay ``Null<Real>``.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.exercise import EuropeanExercise, Exercise
from pquantlib.instruments.margrabe_option import (
    MargrabeOptionArguments,
    MargrabeOptionResults,
)
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.distributions.normal_distribution import NormalDistribution
from pquantlib.payoffs import NullPayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)


class AnalyticEuropeanMargrabeEngine(GenericEngine[MargrabeOptionArguments, MargrabeOptionResults]):
    """Margrabe 1978 closed form for the European exchange option.

    # C++ parity: ``AnalyticEuropeanMargrabeEngine``.

    Args:
        process1: GBSM process of the first (received) asset.
        process2: GBSM process of the second (delivered) asset.
        correlation: Correlation between the two Brownian drivers.
    """

    def __init__(
        self,
        process1: GeneralizedBlackScholesProcess,
        process2: GeneralizedBlackScholesProcess,
        correlation: float,
    ) -> None:
        super().__init__(MargrabeOptionArguments(), MargrabeOptionResults())
        self._process1: GeneralizedBlackScholesProcess = process1
        self._process2: GeneralizedBlackScholesProcess = process2
        self._rho: float = correlation
        process1.register_with(self)
        process2.register_with(self)

    def calculate(self) -> None:
        """# C++ parity: ``AnalyticEuropeanMargrabeEngine::calculate``."""
        args = self._arguments
        results = self._results

        exercise = args.exercise
        assert exercise is not None
        qassert.require(exercise.type() == Exercise.Type.European, "not an European Option")
        qassert.require(isinstance(exercise, EuropeanExercise), "not an European Option")
        qassert.require(isinstance(args.payoff, NullPayoff), "non a Null Payoff type")

        quantity1 = args.q1
        quantity2 = args.q2
        assert quantity1 is not None
        assert quantity2 is not None

        last_date = exercise.last_date()

        s1 = self._process1.state_variable().value()
        s2 = self._process2.state_variable().value()

        variance1 = self._process1.black_volatility().black_variance(last_date, s1)
        variance2 = self._process2.black_volatility().black_variance(last_date, s2)

        risk_free_discount = self._process1.risk_free_rate().discount(last_date)
        dividend_discount1 = self._process1.dividend_yield().discount(last_date)
        dividend_discount2 = self._process2.dividend_yield().discount(last_date)

        forward1 = s1 * dividend_discount1 / risk_free_discount
        forward2 = s2 * dividend_discount2 / risk_free_discount

        std_dev1 = math.sqrt(variance1)
        std_dev2 = math.sqrt(variance2)
        variance = variance1 + variance2 - 2.0 * self._rho * std_dev1 * std_dev2
        std_dev = math.sqrt(variance)

        d1 = (math.log((quantity1 * forward1) / (quantity2 * forward2)) + 0.5 * variance) / std_dev
        d2 = d1 - std_dev

        cum = CumulativeNormalDistribution()
        norm = NormalDistribution()
        n_d1 = cum(d1)
        n_d2 = cum(d2)
        nd1 = norm(d1)
        nd2 = norm(d2)

        rfdc = self._process1.risk_free_rate().day_counter()
        t = rfdc.year_fraction(self._process1.risk_free_rate().reference_date(), last_date)
        sqt = math.sqrt(t)
        # C++ divides by ``sqt*sqt`` rather than ``t`` — kept verbatim so
        # the floating-point result is identical.
        q1 = -math.log(dividend_discount1) / (sqt * sqt)
        q2 = -math.log(dividend_discount2) / (sqt * sqt)

        results.value = risk_free_discount * (quantity1 * forward1 * n_d1 - quantity2 * forward2 * n_d2)

        results.delta1 = risk_free_discount * (quantity1 * forward1 * n_d1) / s1
        results.delta2 = -risk_free_discount * (quantity2 * forward2 * n_d2) / s2
        results.gamma1 = (risk_free_discount * (quantity1 * forward1 * nd1) / s1) / (quantity1 * s1 * std_dev)
        # The double negation is C++ verbatim (it cancels, but the
        # rounding of the intermediate does not).
        results.gamma2 = (-risk_free_discount * (quantity2 * forward2 * nd2) / s2) / (
            -quantity2 * s2 * std_dev
        )
        vega = risk_free_discount * (quantity1 * forward1 * nd1) * sqt
        results.theta = -(
            (std_dev * vega / sqt) / (2.0 * t)
            - (q1 * quantity1 * s1 * results.delta1)
            - (q2 * quantity2 * s2 * results.delta2)
        )
        results.rho = 0.0


__all__ = ["AnalyticEuropeanMargrabeEngine"]
