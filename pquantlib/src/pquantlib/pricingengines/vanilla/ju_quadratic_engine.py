"""JuQuadraticApproximationEngine — Ju (1999) quadratic approximation.

# C++ parity: ql/pricingengines/vanilla/juquadraticengine.{hpp,cpp} (v1.43) —
# ``class JuQuadraticApproximationEngine : public VanillaOption::engine``.

Reference: *An Approximate Formula for Pricing American Options*, Journal of
Derivatives, Winter 1999, N. Ju.

Structure mirrors :class:`BaroneAdesiWhaleyApproximationEngine`:

``dividend_discount >= 1.0 and option_type == Call``
    Early exercise never optimal -> the European price and the **full**
    greek set.

otherwise
    Ju's approximation, which fills ``value``, ``delta`` and ``gamma`` — and
    nothing else.  (BAW's approximation arm fills only ``value``, so the two
    engines have genuinely different "which greeks exist" signatures; the
    cross-validation pins both.)

The critical price ``Sk`` comes from
:meth:`BaroneAdesiWhaleyApproximationEngine.critical_price` — the C++ header
says so explicitly:

    \\warning Barone-Adesi-Whaley critical commodity price calculation is
             used, it has not been modified to see whether the method of Ju
             is faster.

What distinguishes Ju from BAW is the ``hA`` amplitude and the ``chi``
correction:

* ``hA  = phi * (Sk - K) - blackFormula(...) * rDF`` — the early-exercise
  premium *at* the boundary, and
* ``chi = ln(S/Sk) * (b * ln(S/Sk) + c)`` — a quadratic in ``ln(S/Sk)``
  which the premium is divided by, i.e. ``hA * (S/Sk)^lambda / (1 - chi)``.

Delta and gamma carry the correction that the C++ source flags:

    There is a typo in the original paper from Ju-Zhong; the first term is
    the Black-Scholes delta/gamma.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.exercise import AmericanExercise, Exercise
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.distributions.normal_distribution import NormalDistribution
from pquantlib.option import OptionArguments
from pquantlib.payoffs import OptionType, StrikedTypePayoff
from pquantlib.pricingengines.black_calculator import BlackCalculator
from pquantlib.pricingengines.black_formula import black_formula
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.vanilla.barone_adesi_whaley_engine import (
    BaroneAdesiWhaleyApproximationEngine,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)


class JuQuadraticApproximationEngine(GenericEngine[OptionArguments, OneAssetOptionResults]):
    """Ju (1999) quadratic approximation engine for American options.

    # C++ parity: ``class JuQuadraticApproximationEngine``.
    """

    def __init__(self, process: GeneralizedBlackScholesProcess) -> None:
        super().__init__(OptionArguments(), OneAssetOptionResults())
        self._process: GeneralizedBlackScholesProcess = process
        process.register_with(self)

    def calculate(self) -> None:  # noqa: PLR0915 - one-to-one with the C++ body
        """Price the American option.

        # C++ parity: ``JuQuadraticApproximationEngine::calculate``.
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
        strike = payoff.strike()

        process = self._process
        last_date = ex.last_date()
        variance = process.black_volatility().black_variance(last_date, strike, extrapolate=True)
        dividend_discount = process.dividend_yield().discount(last_date)
        risk_free_discount = process.risk_free_rate().discount(last_date)
        spot = process.state_variable().value()
        qassert.require(spot > 0.0, "negative or null underlying given")
        forward_price = spot * dividend_discount / risk_free_discount
        black = BlackCalculator(payoff, forward_price, math.sqrt(variance), risk_free_discount)

        if dividend_discount >= 1.0 and payoff.option_type() == OptionType.Call:
            # Early exercise never optimal.
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

        # Early exercise can be optimal.
        cum_normal_dist = CumulativeNormalDistribution()
        normal_dist = NormalDistribution()

        tolerance = 1e-6
        sk = BaroneAdesiWhaleyApproximationEngine.critical_price(
            payoff, risk_free_discount, dividend_discount, variance, tolerance
        )
        forward_sk = sk * dividend_discount / risk_free_discount

        alpha = -2.0 * math.log(risk_free_discount) / variance
        beta = 2.0 * math.log(dividend_discount / risk_free_discount) / variance
        h = 1 - risk_free_discount
        phi = 1.0 if payoff.option_type() == OptionType.Call else -1.0

        # C++ comment: "it can throw: to be fixed".
        temp_root = math.sqrt((beta - 1) * (beta - 1) + (4 * alpha) / h)
        lambda_ = (-(beta - 1) + phi * temp_root) / 2
        lambda_prime = -phi * alpha / (h * h * temp_root)

        black_sk = (
            black_formula(payoff.option_type(), strike, forward_sk, math.sqrt(variance))
            * risk_free_discount
        )
        h_a = phi * (sk - strike) - black_sk

        d1_sk = (math.log(forward_sk / strike) + 0.5 * variance) / math.sqrt(variance)
        d2_sk = d1_sk - math.sqrt(variance)
        part1 = forward_sk * normal_dist(d1_sk) / (alpha * math.sqrt(variance))
        part2 = (
            -phi
            * forward_sk
            * cum_normal_dist(phi * d1_sk)
            * math.log(dividend_discount)
            / math.log(risk_free_discount)
        )
        part3 = +phi * strike * cum_normal_dist(phi * d2_sk)
        v_e_h = part1 + part2 + part3

        b = (1 - h) * alpha * lambda_prime / (2 * (2 * lambda_ + beta - 1))
        c = -((1 - h) * alpha / (2 * lambda_ + beta - 1)) * (
            v_e_h / h_a + 1 / h + lambda_prime / (2 * lambda_ + beta - 1)
        )
        temp_spot_ratio = math.log(spot / sk)
        chi = temp_spot_ratio * (b * temp_spot_ratio + c)

        if phi * (sk - spot) > 0:
            results.value = black.value() + h_a * math.pow(spot / sk, lambda_) / (1 - chi)
            temp_chi_prime = (2 * b / spot) * math.log(spot / sk)
            chi_prime = temp_chi_prime + c / spot
            chi_double_prime = 2 * b / (spot * spot) - temp_chi_prime / spot - c / (spot * spot)
            d1_s = (math.log(forward_price / strike) + 0.5 * variance) / math.sqrt(variance)
            # NOTE: the leading term is the Black-Scholes delta/gamma — the
            # C++ source flags a typo in the original Ju-Zhong paper here.
            results.delta = phi * dividend_discount * cum_normal_dist(phi * d1_s) + (
                lambda_ / (spot * (1 - chi)) + chi_prime / ((1 - chi) * (1 - chi))
            ) * (phi * (sk - strike) - black_sk) * math.pow(spot / sk, lambda_)

            results.gamma = (
                dividend_discount * normal_dist(phi * d1_s) / (spot * math.sqrt(variance))
                + (
                    2 * lambda_ * chi_prime / (spot * (1 - chi) * (1 - chi))
                    + 2 * chi_prime * chi_prime / ((1 - chi) * (1 - chi) * (1 - chi))
                    + chi_double_prime / ((1 - chi) * (1 - chi))
                    + lambda_ * (lambda_ - 1) / (spot * spot * (1 - chi))
                )
                * (phi * (sk - strike) - black_sk)
                * math.pow(spot / sk, lambda_)
            )
        else:
            results.value = phi * (spot - strike)
            results.delta = phi
            results.gamma = 0.0


__all__ = ["JuQuadraticApproximationEngine"]
