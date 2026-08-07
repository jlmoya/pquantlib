"""BaroneAdesiWhaleyApproximationEngine — quadratic approximation (1987).

# C++ parity: ql/pricingengines/vanilla/baroneadesiwhaleyengine.{hpp,cpp}
# (v1.43) — ``class BaroneAdesiWhaleyApproximationEngine :
# public VanillaOption::engine``.

The engine has exactly two arms:

``dividend_discount >= 1.0 and option_type == Call``
    Early exercise is never optimal, so the answer *is* the European price
    and the **full** greek set is filled from a :class:`BlackCalculator`
    (value, delta, delta_forward, elasticity, gamma, rho, dividend_rho,
    vega, theta, theta_per_day, strike_sensitivity, itm_cash_probability).
    Note the comparison is ``>=``: ``q == 0`` exactly takes this arm.

otherwise
    The Barone-Adesi/Whaley quadratic approximation, which fills **only**
    ``value``.  Every greek is left unset — the instrument's accessors
    raise.  That asymmetry between the two arms is deliberate C++ behaviour
    and is cross-validated.

:meth:`BaroneAdesiWhaleyApproximationEngine.critical_price` is a public
static in C++ and is kept as a ``@staticmethod`` here, with the same
signature including the ``tolerance = 1e-6`` default:
:class:`~pquantlib.pricingengines.vanilla.ju_quadratic_engine.JuQuadraticApproximationEngine`
calls it directly, so hiding it inside the engine would make Ju
unimplementable.  It solves ``S* - K = c(S*) + (1 - qDF N(d1)) S*/Q`` by
Newton-Raphson, stopping when ``|LHS - RHS| / strike <= tolerance`` — a
*relative-to-strike* criterion, not an absolute one.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.exercise import AmericanExercise, Exercise
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.closeness import close
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.option import OptionArguments
from pquantlib.payoffs import OptionType, StrikedTypePayoff
from pquantlib.pricingengines._ieee_arithmetic import div as _div
from pquantlib.pricingengines.black_calculator import BlackCalculator
from pquantlib.pricingengines.black_formula import black_formula
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)


class BaroneAdesiWhaleyApproximationEngine(
    GenericEngine[OptionArguments, OneAssetOptionResults]
):
    """Barone-Adesi and Whaley (1987) American option engine.

    # C++ parity: ``class BaroneAdesiWhaleyApproximationEngine``.
    """

    def __init__(self, process: GeneralizedBlackScholesProcess) -> None:
        super().__init__(OptionArguments(), OneAssetOptionResults())
        self._process: GeneralizedBlackScholesProcess = process
        process.register_with(self)

    @staticmethod
    def critical_price(
        payoff: StrikedTypePayoff,
        risk_free_discount: float,
        dividend_discount: float,
        variance: float,
        tolerance: float = 1e-6,
    ) -> float:
        """Critical commodity price S* above/below which exercise is optimal.

        # C++ parity: ``static Real
        # BaroneAdesiWhaleyApproximationEngine::criticalPrice(
        #     const ext::shared_ptr<StrikedTypePayoff>&, DiscountFactor,
        #     DiscountFactor, Real, Real tolerance = 1e-6)``.
        """
        qassert.require(
            risk_free_discount <= 1.0,
            "the Barone-Adesi-Whaley approximation is not applicable "
            "with negative interest rates "
            f"(risk-free discount factor: {risk_free_discount})",
        )

        strike = payoff.strike()

        # Seed value Si.
        # IEEE semantics: at zero variance C++ propagates +-inf/NaN through to
        # blackFormula, which is what raises. See ``_ieee_arithmetic``.
        n = _div(2.0 * math.log(dividend_discount / risk_free_discount), variance)
        m = _div(-2.0 * math.log(risk_free_discount), variance)
        b_t = math.log(dividend_discount / risk_free_discount)

        if payoff.option_type() == OptionType.Call:
            qu = (-(n - 1.0) + math.sqrt((n - 1.0) * (n - 1.0) + 4.0 * m)) / 2.0
            su = _div(strike, 1.0 - _div(1.0, qu))
            h = _div(-(b_t + 2.0 * math.sqrt(variance)) * strike, su - strike)
            si = strike + (su - strike) * (1.0 - math.exp(h))
        else:
            qu = (-(n - 1.0) - math.sqrt((n - 1.0) * (n - 1.0) + 4.0 * m)) / 2.0
            su = _div(strike, 1.0 - _div(1.0, qu))
            h = _div((b_t - 2.0 * math.sqrt(variance)) * strike, strike - su)
            si = su + (strike - su) * math.exp(h)

        # Newton-Raphson on Si.
        forward_si = si * dividend_discount / risk_free_discount
        d1 = _div(math.log(forward_si / strike) + 0.5 * variance, math.sqrt(variance))
        cum_normal_dist = CumulativeNormalDistribution()
        # C++ ``close(riskFreeDiscount, 1.0, 1000)`` — the 1000-ulp overload.
        k = (
            _div(-2.0 * math.log(risk_free_discount), variance * (1.0 - risk_free_discount))
            if not close(risk_free_discount, 1.0, 1000)
            else _div(2.0, variance)
        )
        temp = (
            black_formula(payoff.option_type(), strike, forward_si, math.sqrt(variance))
            * risk_free_discount
        )

        if payoff.option_type() == OptionType.Call:
            q = (-(n - 1.0) + math.sqrt((n - 1.0) * (n - 1.0) + 4 * k)) / 2
            lhs = si - strike
            rhs = temp + (1 - dividend_discount * cum_normal_dist(d1)) * si / q
            bi = dividend_discount * cum_normal_dist(d1) * (1 - 1 / q) + (
                1 - dividend_discount * cum_normal_dist.derivative(d1) / math.sqrt(variance)
            ) / q
            while abs(lhs - rhs) / strike > tolerance:
                si = (strike + rhs - bi * si) / (1 - bi)
                forward_si = si * dividend_discount / risk_free_discount
                d1 = _div(math.log(forward_si / strike) + 0.5 * variance, math.sqrt(variance))
                lhs = si - strike
                temp2 = (
                    black_formula(payoff.option_type(), strike, forward_si, math.sqrt(variance))
                    * risk_free_discount
                )
                rhs = temp2 + (1 - dividend_discount * cum_normal_dist(d1)) * si / q
                bi = dividend_discount * cum_normal_dist(d1) * (1 - 1 / q) + (
                    1 - dividend_discount * cum_normal_dist.derivative(d1) / math.sqrt(variance)
                ) / q
        else:
            q = (-(n - 1.0) - math.sqrt((n - 1.0) * (n - 1.0) + 4 * k)) / 2
            lhs = strike - si
            rhs = temp - (1 - dividend_discount * cum_normal_dist(-d1)) * si / q
            bi = -dividend_discount * cum_normal_dist(-d1) * (1 - 1 / q) - (
                1 + dividend_discount * cum_normal_dist.derivative(-d1) / math.sqrt(variance)
            ) / q
            while abs(lhs - rhs) / strike > tolerance:
                si = (strike - rhs + bi * si) / (1 + bi)
                forward_si = si * dividend_discount / risk_free_discount
                d1 = _div(math.log(forward_si / strike) + 0.5 * variance, math.sqrt(variance))
                lhs = strike - si
                temp2 = (
                    black_formula(payoff.option_type(), strike, forward_si, math.sqrt(variance))
                    * risk_free_discount
                )
                rhs = temp2 - (1 - dividend_discount * cum_normal_dist(-d1)) * si / q
                bi = -dividend_discount * cum_normal_dist(-d1) * (1 - 1 / q) - (
                    1 + dividend_discount * cum_normal_dist.derivative(-d1) / math.sqrt(variance)
                ) / q

        return si

    def calculate(self) -> None:  # noqa: PLR0915 - one-to-one with the C++ body
        """Price the American option.

        # C++ parity: ``BaroneAdesiWhaleyApproximationEngine::calculate``.
        """
        args = self._arguments
        results = self._results

        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        qassert.require(
            args.exercise.type() == Exercise.Type.American,
            "not an American Option",
        )
        qassert.require(isinstance(args.exercise, AmericanExercise), "non-American exercise given")
        assert isinstance(args.exercise, AmericanExercise)
        ex: AmericanExercise = args.exercise
        qassert.require(not ex.payoff_at_expiry(), "payoff at expiry not handled")

        qassert.require(args.payoff is not None, "no payoff given")
        assert args.payoff is not None
        qassert.require(isinstance(args.payoff, StrikedTypePayoff), "non-striked payoff given")
        assert isinstance(args.payoff, StrikedTypePayoff)
        payoff: StrikedTypePayoff = args.payoff

        process = self._process
        last_date = ex.last_date()
        variance = process.black_volatility().black_variance(
            last_date, payoff.strike(), extrapolate=True
        )
        dividend_discount = process.dividend_yield().discount(last_date)
        risk_free_discount = process.risk_free_rate().discount(last_date)
        spot = process.state_variable().value()
        qassert.require(spot > 0.0, "negative or null underlying given")
        forward_price = spot * dividend_discount / risk_free_discount
        black = BlackCalculator(payoff, forward_price, math.sqrt(variance), risk_free_discount)

        if dividend_discount >= 1.0 and payoff.option_type() == OptionType.Call:
            # Early exercise never optimal: the European answer, full greeks.
            results.value = black.value()
            results.delta = black.delta(spot)
            results.delta_forward = black.delta_forward()
            results.elasticity = black.elasticity(spot)
            results.gamma = black.gamma(spot)

            rfdc = process.risk_free_rate().day_counter()
            divdc = process.dividend_yield().day_counter()
            voldc = process.black_volatility().day_counter()
            t = rfdc.year_fraction(process.risk_free_rate().reference_date(), last_date)
            results.rho = black.rho(t)

            t = divdc.year_fraction(process.dividend_yield().reference_date(), last_date)
            results.dividend_rho = black.dividend_rho(t)

            t = voldc.year_fraction(process.black_volatility().reference_date(), last_date)
            results.vega = black.vega(t)
            results.theta = black.theta(spot, t)
            results.theta_per_day = black.theta_per_day(spot, t)

            results.strike_sensitivity = black.strike_sensitivity()
            results.itm_cash_probability = black.itm_cash_probability()
            return

        # Early exercise can be optimal: the quadratic approximation. Only
        # ``value`` is filled — matching C++, which leaves every greek Null.
        cum_normal_dist = CumulativeNormalDistribution()
        tolerance = 1e-6
        sk = self.critical_price(
            payoff, risk_free_discount, dividend_discount, variance, tolerance
        )
        forward_sk = sk * dividend_discount / risk_free_discount
        d1 = _div(math.log(forward_sk / payoff.strike()) + 0.5 * variance, math.sqrt(variance))
        n = _div(2.0 * math.log(dividend_discount / risk_free_discount), variance)
        k = (
            _div(-2.0 * math.log(risk_free_discount), variance * (1.0 - risk_free_discount))
            if not close(risk_free_discount, 1.0, 1000)
            else _div(2.0, variance)
        )

        if payoff.option_type() == OptionType.Call:
            q = (-(n - 1.0) + math.sqrt((n - 1.0) * (n - 1.0) + 4.0 * k)) / 2.0
            a = (sk / q) * (1.0 - dividend_discount * cum_normal_dist(d1))
            if spot < sk:
                results.value = black.value() + a * math.pow(spot / sk, q)
            else:
                results.value = spot - payoff.strike()
        else:
            q = (-(n - 1.0) - math.sqrt((n - 1.0) * (n - 1.0) + 4.0 * k)) / 2.0
            a = -(sk / q) * (1.0 - dividend_discount * cum_normal_dist(-d1))
            if spot > sk:
                results.value = black.value() + a * math.pow(spot / sk, q)
            else:
                results.value = payoff.strike() - spot


__all__ = ["BaroneAdesiWhaleyApproximationEngine"]
