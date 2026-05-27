"""AnalyticEuropeanEngine — closed-form pricing for European vanilla options.

# C++ parity: ql/pricingengines/vanilla/analyticeuropeanengine.{hpp,cpp}
# (v1.42.1) — ``class AnalyticEuropeanEngine : public VanillaOption::engine``.

The engine pulls market data from a ``GeneralizedBlackScholesProcess``
(spot via ``stateVariable``, risk-free + dividend curves, Black vol
surface), constructs a ``BlackCalculator`` at the forward price, and
fills the standard option result fields (NPV + Greeks + additional
results).

Two constructors in C++:

* ``AnalyticEuropeanEngine(process)`` — use the process's own
  risk-free curve for discounting.
* ``AnalyticEuropeanEngine(process, discountCurve)`` — use a separate
  discount curve (forecasting still uses the process's risk-free).

The Python port keeps both forms — the second curve is optional.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.option import OptionArguments
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.black_calculator import BlackCalculator
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.termstructures.yield_term_structure import YieldTermStructure


class AnalyticEuropeanEngine(GenericEngine[OptionArguments, OneAssetOptionResults]):
    """Analytic European vanilla option engine.

    # C++ parity: ``AnalyticEuropeanEngine``.
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        discount_curve: YieldTermStructure | None = None,
    ) -> None:
        super().__init__(OptionArguments(), OneAssetOptionResults())
        self._process: GeneralizedBlackScholesProcess = process
        self._discount_curve: YieldTermStructure | None = discount_curve
        process.register_with(self)
        if discount_curve is not None:
            discount_curve.register_with(self)

    def calculate(self) -> None:
        """Compute value + Greeks via ``BlackCalculator``.

        # C++ parity: ``AnalyticEuropeanEngine::calculate``.
        """
        args = self._arguments
        results = self._results

        qassert.require(args.exercise is not None, "no exercise given")
        qassert.require(args.payoff is not None, "no payoff given")
        assert args.exercise is not None
        assert args.payoff is not None

        qassert.require(
            args.exercise.type() == Exercise.Type.European,
            "not a European option",
        )

        qassert.require(
            isinstance(args.payoff, StrikedTypePayoff),
            "non-striked payoff given",
        )
        assert isinstance(args.payoff, StrikedTypePayoff)
        payoff: StrikedTypePayoff = args.payoff

        process = self._process
        discount_ts: YieldTermStructure = (
            self._discount_curve if self._discount_curve is not None else process.risk_free_rate()
        )

        last_date = args.exercise.last_date()

        variance = process.black_volatility().black_variance(
            last_date, payoff.strike(), extrapolate=True
        )
        dividend_discount = process.dividend_yield().discount(last_date)
        df = discount_ts.discount(last_date)
        risk_free_discount_for_fwd = process.risk_free_rate().discount(last_date)
        spot = process.state_variable().value()
        qassert.require(spot > 0.0, "negative or null underlying given")

        forward_price = spot * dividend_discount / risk_free_discount_for_fwd

        black = BlackCalculator(payoff, forward_price, math.sqrt(variance), df)

        # Maturity used by the BlackCalculator Greeks is computed
        # against the engine's *own* day-counter on each curve — C++
        # uses ``rfdc.yearFraction(refDate, last_date)`` for rho /
        # vega / theta etc. (each curve has its own daycounter).
        rfdc = discount_ts.day_counter()
        divdc = process.dividend_yield().day_counter()
        voldc = process.black_volatility().day_counter()
        t_rho = rfdc.year_fraction(discount_ts.reference_date(), last_date)
        t_dividend_rho = divdc.year_fraction(
            process.dividend_yield().reference_date(), last_date
        )
        t_vega = voldc.year_fraction(
            process.black_volatility().reference_date(), last_date
        )

        results.value = black.value()
        results.delta = black.delta(spot)
        results.delta_forward = black.delta_forward()
        results.elasticity = black.elasticity(spot)
        results.gamma = black.gamma(spot)
        results.rho = black.rho(t_rho)
        results.dividend_rho = black.dividend_rho(t_dividend_rho)
        results.vega = black.vega(t_vega)
        try:
            results.theta = black.theta(spot, t_vega)
            results.theta_per_day = black.theta_per_day(spot, t_vega)
        except LibraryException:
            results.theta = None
            results.theta_per_day = None
        results.strike_sensitivity = black.strike_sensitivity()
        results.itm_cash_probability = black.itm_cash_probability()

        # Additional results — mirror C++ keys exactly.
        tte = process.black_volatility().time_from_reference(last_date)
        results.additional_results = {
            "spot": spot,
            "dividendDiscount": dividend_discount,
            "riskFreeDiscount": risk_free_discount_for_fwd,
            "forward": forward_price,
            "strike": payoff.strike(),
            "volatility": math.sqrt(variance / tte) if tte > 0.0 else 0.0,
            "timeToExpiry": tte,
        }


__all__ = ["AnalyticEuropeanEngine"]
