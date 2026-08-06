"""AnalyticAmericanMargrabeEngine — American exchange option.

# C++ parity:
# ql/pricingengines/exotic/analyticamericanmargrabeengine.{hpp,cpp} (v1.43).

Reference: W. Margrabe, "The Value of an American Option to Exchange One
Asset for Another", Journal of Finance 33, 177-186.

An American exchange option is an American *single-asset* call on the
ratio: spot ``q1*S1``, strike ``q2*S2``, dividend yield ``q_1``,
risk-free rate ``q_2``, volatility ``sqrt(var/t)`` with
``var = var1 + var2 - 2*rho*sd1*sd2``.  C++ builds exactly that synthetic
``BlackScholesMertonProcess`` and prices it with
``BjerksundStenslandApproximationEngine``, taking ``NPV()`` only.

This engine fills ``value`` and nothing else, so every Greek accessor on
the instrument raises — same as C++.

PORTING NOTE — ``BjerksundStenslandApproximationEngine`` is not (yet)
ported to pquantlib.  Only the *value* branch of that engine is needed
here (C++ reads ``option.NPV()`` and discards the Greeks), so the value
recursion is reproduced below as module-private functions whose
signatures mirror the C++ private members
(``europeanCallResults`` / ``immediateExercise`` /
``americanCallApproximation`` / the anonymous-namespace ``phi``).  When
the full engine is ported, these should be deleted and this engine
should route through it.

PORTING NOTE — C++ takes ``today`` from
``Settings::instance().evaluationDate()`` to anchor the synthetic curves.
pquantlib has no global evaluation date, so the reference date of the
first process's risk-free curve is used instead; in C++ the two coincide
whenever the curves are anchored at the evaluation date.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.exercise import AmericanExercise, Exercise
from pquantlib.instruments.margrabe_option import (
    MargrabeOptionArguments,
    MargrabeOptionResults,
)
from pquantlib.math.constants import QL_EPSILON
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.payoffs import NullPayoff, OptionType
from pquantlib.pricingengines.black_calculator import BlackCalculator
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.calendars.null_calendar import NullCalendar

_CUM = CumulativeNormalDistribution()


def _phi(s: float, gamma: float, h: float, i: float, r_t: float, b_t: float, variance: float) -> float:
    """# C++ parity: anonymous-namespace ``phi`` in bjerksundstenslandengine.cpp."""
    lambda_ = -r_t + gamma * b_t + 0.5 * gamma * (gamma - 1.0) * variance
    d = -(math.log(s / h) + (b_t + (gamma - 0.5) * variance)) / math.sqrt(variance)
    kappa = 2.0 * b_t / variance + (2.0 * gamma - 1.0)
    return math.exp(lambda_) * (
        _CUM(d) - math.pow(i / s, kappa) * _CUM(d - 2.0 * math.log(i / s) / math.sqrt(variance))
    )


def _european_call_value(s: float, x: float, rf_d: float, d_d: float, variance: float) -> float:
    """# C++ parity: value branch of ``europeanCallResults``."""
    forward_price = s * d_d / rf_d
    black = BlackCalculator.from_type_strike(OptionType.Call, x, forward_price, math.sqrt(variance), rf_d)
    return black.value()


def _american_call_approximation_value(s: float, x: float, rf_d: float, d_d: float, variance: float) -> float:
    """# C++ parity: value branch of ``americanCallApproximation``."""
    european = _european_call_value(s, x, rf_d, d_d, variance)

    b_t = math.log(d_d / rf_d)
    r_t = math.log(1.0 / rf_d)

    beta = (0.5 - b_t / variance) + math.sqrt((b_t / variance - 0.5) ** 2 + 2.0 * r_t / variance)
    b_infinity = beta / (beta - 1.0) * x
    b0 = x if b_t == r_t else max(x, r_t / (r_t - b_t) * x)
    ht = -(b_t + 2.0 * math.sqrt(variance)) * b0 / (b_infinity - b0)
    i = b0 + (b_infinity - b0) * (1.0 - math.exp(ht))

    fwd = s * d_d / rf_d

    # PORTING NOTE: C++ evaluates ``q = log(I/fwd)/sqrt(variance)`` *before*
    # this branch. A large negative ``ht`` exponent can drive ``I`` negative
    # (it does, for a high asset-1 dividend yield), and C++ then relies on
    # ``std::log`` of a non-positive argument returning NaN, which silently
    # fails the ``q > 12.5`` test. ``math.log`` raises instead, so ``q`` is
    # computed lazily in the branch below; reaching it implies ``i > s > 0``,
    # where the logarithm is always well defined, so no case is lost.
    if s >= i:
        value = max(0.0, s - x)
    elif math.log(i / fwd) / math.sqrt(variance) > 12.5:
        # Run-away exercise boundary: C++ falls back to the European
        # results wholesale.
        value = european
    else:
        value = (
            (i - x) * math.pow(s / i, beta) * (1.0 - _phi(s, beta, i, i, r_t, b_t, variance))
            + s * _phi(s, 1.0, i, i, r_t, b_t, variance)
            - s * _phi(s, 1.0, x, i, r_t, b_t, variance)
            - x * _phi(s, 0.0, i, i, r_t, b_t, variance)
            + x * _phi(s, 0.0, x, i, r_t, b_t, variance)
        )

    # C++: "check if European engine gives higher NPV".
    return european if value < european else value


def _bjerksund_stensland_value(s: float, x: float, rf_d: float, d_d: float, variance: float) -> float:
    """# C++ parity: value branch of
    # ``BjerksundStenslandApproximationEngine::calculate`` for a Call.
    """
    qassert.require(s > 0.0, "negative or null underlying given")
    qassert.require(not (d_d > 1.0 and rf_d > d_d), "double-boundary case r<q<0 for a call given")

    if d_d >= 1.0 and d_d >= rf_d:
        value = _european_call_value(s, x, rf_d, d_d, variance)
    else:
        value = _american_call_approximation_value(s, x, rf_d, d_d, variance)

    # C++: "check if immediate exercise gives higher NPV".
    if value < (s - x) * (1.0 + 10.0 * QL_EPSILON):
        value = max(0.0, s - x)
    return value


class AnalyticAmericanMargrabeEngine(GenericEngine[MargrabeOptionArguments, MargrabeOptionResults]):
    """Bjerksund-Stensland approximation for the American exchange option.

    # C++ parity: ``AnalyticAmericanMargrabeEngine``.

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
        """# C++ parity: ``AnalyticAmericanMargrabeEngine::calculate``."""
        args = self._arguments
        results = self._results

        exercise = args.exercise
        assert exercise is not None
        qassert.require(exercise.type() == Exercise.Type.American, "not an American option")
        qassert.require(isinstance(exercise, AmericanExercise), "not an American option")
        qassert.require(isinstance(args.payoff, NullPayoff), "not a null payoff")

        quantity1 = args.q1
        quantity2 = args.q2
        assert quantity1 is not None
        assert quantity2 is not None

        last_date = exercise.last_date()
        rf_ts = self._process1.risk_free_rate()
        rfdc = rf_ts.day_counter()
        today = rf_ts.reference_date()
        t = rfdc.year_fraction(today, last_date)

        s1 = self._process1.state_variable().value()
        s2 = self._process2.state_variable().value()

        spot = quantity1 * s1
        strike = quantity2 * s2

        dividend_discount1 = self._process1.dividend_yield().discount(last_date)
        q1 = -math.log(dividend_discount1) / t
        dividend_discount2 = self._process2.dividend_yield().discount(last_date)
        q2 = -math.log(dividend_discount2) / t

        # The synthetic single-asset process: asset-1 yield becomes the
        # dividend yield, asset-2 yield becomes the risk-free rate.
        q_ts = FlatForward.from_rate(today, q1, rfdc)
        r_ts = FlatForward.from_rate(today, q2, rfdc)

        variance1 = self._process1.black_volatility().black_variance(last_date, s1)
        variance2 = self._process2.black_volatility().black_variance(last_date, s2)
        variance = variance1 + variance2 - 2.0 * self._rho * math.sqrt(variance1) * math.sqrt(variance2)
        volatility = math.sqrt(variance / t)
        vol_ts = BlackConstantVol(
            reference_date=today,
            calendar=NullCalendar(),
            day_counter=rfdc,
            volatility=volatility,
        )

        # Values the Bjerksund-Stensland engine would read off the
        # synthetic process (C++ round-trips through the term structures,
        # so we do too).
        bs_variance = vol_ts.black_variance(last_date, strike)
        dividend_discount = q_ts.discount(last_date)
        risk_free_discount = r_ts.discount(last_date)

        results.value = _bjerksund_stensland_value(
            spot, strike, risk_free_discount, dividend_discount, bs_variance
        )


__all__ = ["AnalyticAmericanMargrabeEngine"]
