"""StulzEngine — 2-asset basket option closed form (Stulz 1982).

# C++ parity: ql/pricingengines/basket/stulzengine.{hpp,cpp} (v1.42.1).

Closed-form pricing for 2D European min/max basket options under
Black-Scholes dynamics. Reference: R. Stulz, "Options on the Minimum
or the Maximum of Two Risky Assets", Journal of Financial Economics
(1982) 10, 161-185.

The engine prices 4 combos:
  * Min-basket Call:  ``E[(min(S1, S2) - K)+]``
  * Min-basket Put:   parity = ``K * df - min_call(K=0) + min_call(K=K)``
  * Max-basket Call:  ``max_call = vanilla_1 + vanilla_2 - min_call``
  * Max-basket Put:   parity = ``K * df - max_call(K=0) + max_call(K=K)``

The closed form uses ``BivariateCumulativeNormalDistribution`` to
evaluate the joint integral over the two log-normal underlyings.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.exercise import EuropeanExercise, Exercise
from pquantlib.instruments.basket_option import (
    BasketOptionResults,
    BasketPayoff,
    MaxBasketPayoff,
    MinBasketPayoff,
)
from pquantlib.math.distributions.bivariate_normal_distribution import (
    BivariateCumulativeNormalDistribution,
)
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.option import OptionArguments
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.black_formula import black_formula
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)


def _euro_two_asset_min_basket_call(
    forward1: float,
    forward2: float,
    strike: float,
    risk_free_discount: float,
    variance1: float,
    variance2: float,
    rho: float,
) -> float:
    """Stulz 1982 closed form for min(S1, S2) basket call.

    # C++ parity: ``euroTwoAssetMinBasketCall`` (anonymous-namespace
    # function in ``stulzengine.cpp``).
    """
    std_dev1 = math.sqrt(variance1)
    std_dev2 = math.sqrt(variance2)

    variance = variance1 + variance2 - 2.0 * rho * std_dev1 * std_dev2
    std_dev = math.sqrt(variance)

    mod_rho_1 = (rho * std_dev2 - std_dev1) / std_dev
    mod_rho_2 = (rho * std_dev1 - std_dev2) / std_dev

    d1_v = (math.log(forward1 / forward2) + 0.5 * variance) / std_dev

    if strike != 0.0:
        biv_c_norm = BivariateCumulativeNormalDistribution(rho)
        biv_c_norm_mod_2 = BivariateCumulativeNormalDistribution(mod_rho_2)
        biv_c_norm_mod_1 = BivariateCumulativeNormalDistribution(mod_rho_1)

        d1_1 = (math.log(forward1 / strike) + 0.5 * variance1) / std_dev1
        d1_2 = (math.log(forward2 / strike) + 0.5 * variance2) / std_dev2
        alfa = biv_c_norm_mod_1(d1_1, -d1_v)
        beta = biv_c_norm_mod_2(d1_2, d1_v - std_dev)
        gamma = biv_c_norm(d1_1 - std_dev1, d1_2 - std_dev2)
    else:
        cum = CumulativeNormalDistribution()
        alfa = cum(-d1_v)
        beta = cum(d1_v - std_dev)
        gamma = 1.0

    return risk_free_discount * (forward1 * alfa + forward2 * beta - strike * gamma)


def _euro_two_asset_max_basket_call(
    forward1: float,
    forward2: float,
    strike: float,
    risk_free_discount: float,
    variance1: float,
    variance2: float,
    rho: float,
) -> float:
    """Max-basket call = vanilla1 + vanilla2 - min-basket call.

    # C++ parity: ``euroTwoAssetMaxBasketCall``.
    """
    black1 = (
        black_formula(OptionType.Call, strike, forward1, math.sqrt(variance1))
        * risk_free_discount
    )
    black2 = (
        black_formula(OptionType.Call, strike, forward2, math.sqrt(variance2))
        * risk_free_discount
    )
    return (
        black1
        + black2
        - _euro_two_asset_min_basket_call(
            forward1, forward2, strike, risk_free_discount, variance1, variance2, rho
        )
    )


class StulzEngine(GenericEngine[OptionArguments, BasketOptionResults]):
    """Stulz 1982 2-asset basket option closed-form engine.

    # C++ parity: ``StulzEngine``.
    """

    def __init__(
        self,
        process1: GeneralizedBlackScholesProcess,
        process2: GeneralizedBlackScholesProcess,
        correlation: float,
    ) -> None:
        super().__init__(OptionArguments(), BasketOptionResults())
        self._process1: GeneralizedBlackScholesProcess = process1
        self._process2: GeneralizedBlackScholesProcess = process2
        self._rho: float = correlation
        process1.register_with(self)
        process2.register_with(self)

    def calculate(self) -> None:
        """Branch on (basket type, option type).

        # C++ parity: ``StulzEngine::calculate``.
        """
        args = self._arguments
        results = self._results

        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        qassert.require(
            args.exercise.type() == Exercise.Type.European,
            "not an European Option",
        )
        # C++ also casts to EuropeanExercise; the type-narrowing assert
        # surfaces the same constraint.
        assert isinstance(args.exercise, EuropeanExercise)

        basket_payoff = args.payoff
        qassert.require(
            isinstance(basket_payoff, BasketPayoff), "unknown basket type"
        )
        assert isinstance(basket_payoff, BasketPayoff)

        is_min_basket = isinstance(basket_payoff, MinBasketPayoff)
        is_max_basket = isinstance(basket_payoff, MaxBasketPayoff)
        qassert.require(
            is_min_basket or is_max_basket, "unknown basket type"
        )

        base_payoff = basket_payoff.base_payoff()
        qassert.require(
            isinstance(base_payoff, PlainVanillaPayoff), "non-plain payoff given"
        )
        assert isinstance(base_payoff, PlainVanillaPayoff)
        payoff: PlainVanillaPayoff = base_payoff

        strike = payoff.strike()
        exercise_date = args.exercise.last_date()

        variance1 = self._process1.black_volatility().black_variance(
            exercise_date, strike, extrapolate=True
        )
        variance2 = self._process2.black_volatility().black_variance(
            exercise_date, strike, extrapolate=True
        )

        risk_free_discount = self._process1.risk_free_rate().discount(exercise_date)
        dividend_discount1 = self._process1.dividend_yield().discount(exercise_date)
        dividend_discount2 = self._process2.dividend_yield().discount(exercise_date)

        forward1 = (
            self._process1.state_variable().value() * dividend_discount1 / risk_free_discount
        )
        forward2 = (
            self._process2.state_variable().value() * dividend_discount2 / risk_free_discount
        )

        if is_max_basket:
            if payoff.option_type() == OptionType.Call:
                results.value = _euro_two_asset_max_basket_call(
                    forward1, forward2, strike, risk_free_discount,
                    variance1, variance2, self._rho,
                )
            else:  # Put — parity
                results.value = (
                    strike * risk_free_discount
                    - _euro_two_asset_max_basket_call(
                        forward1, forward2, 0.0, risk_free_discount,
                        variance1, variance2, self._rho,
                    )
                    + _euro_two_asset_max_basket_call(
                        forward1, forward2, strike, risk_free_discount,
                        variance1, variance2, self._rho,
                    )
                )
        # Min basket.
        elif payoff.option_type() == OptionType.Call:
            results.value = _euro_two_asset_min_basket_call(
                forward1, forward2, strike, risk_free_discount,
                variance1, variance2, self._rho,
            )
        else:  # Put — parity
            results.value = (
                strike * risk_free_discount
                - _euro_two_asset_min_basket_call(
                    forward1, forward2, 0.0, risk_free_discount,
                    variance1, variance2, self._rho,
                )
                + _euro_two_asset_min_basket_call(
                    forward1, forward2, strike, risk_free_discount,
                    variance1, variance2, self._rho,
                )
            )


__all__ = ["StulzEngine"]
