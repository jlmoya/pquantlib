"""SpreadBlackScholesVanillaEngine — base class for 2D BSM spread engines.

# C++ parity:
# ql/pricingengines/basket/spreadblackscholesvanillaengine.{hpp,cpp}
# (v1.42.1).

Abstract base for spread-option closed-form engines under Black-Scholes
on each leg. The base computes the two forwards, variances, and risk-free
discount; subclasses implement the model-specific ``_calculate`` hook
that turns ``(F1, F2, K, type, var1, var2, df)`` into the option value.

Subclasses (currently): :class:`KirkEngine`. The 2024 C++ rewrite adds
``BjerksundStensland2014SpreadEngine``, ``DengLiZhouSpreadEngine``, etc.
— those go in follow-up clusters.
"""

from __future__ import annotations

from abc import abstractmethod

from pquantlib import qassert
from pquantlib.exercise import EuropeanExercise, Exercise
from pquantlib.instruments.basket_option import (
    BasketOptionResults,
    SpreadBasketPayoff,
)
from pquantlib.option import OptionArguments
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)


class SpreadBlackScholesVanillaEngine(
    GenericEngine[OptionArguments, BasketOptionResults]
):
    """Base class for 2-asset BSM spread-option engines.

    # C++ parity: ``SpreadBlackScholesVanillaEngine``.
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

    # --- subclass hook --------------------------------------------------

    @abstractmethod
    def _calculate(
        self,
        f1: float,
        f2: float,
        strike: float,
        option_type: OptionType,
        variance1: float,
        variance2: float,
        df: float,
    ) -> float:
        """Subclass: return the spread-option NPV.

        # C++ parity: pure virtual ``calculate(...)`` in
        # spreadblackscholesvanillaengine.hpp:41-44.
        """

    # --- driver ---------------------------------------------------------

    def calculate(self) -> None:
        """Build the inputs + dispatch to ``_calculate``.

        # C++ parity:
        # ``SpreadBlackScholesVanillaEngine::calculate`` (spreadbsve.cpp:36-71).
        """
        args = self._arguments
        results = self._results

        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        qassert.require(
            args.exercise.type() == Exercise.Type.European,
            "not an European exercise",
        )
        assert isinstance(args.exercise, EuropeanExercise)

        spread_payoff = args.payoff
        qassert.require(
            isinstance(spread_payoff, SpreadBasketPayoff),
            " spread payoff expected",
        )
        assert isinstance(spread_payoff, SpreadBasketPayoff)

        payoff = spread_payoff.base_payoff()
        qassert.require(
            isinstance(payoff, PlainVanillaPayoff), "non-plain payoff given"
        )
        assert isinstance(payoff, PlainVanillaPayoff)
        strike = payoff.strike()
        option_type = payoff.option_type()

        maturity_date = args.exercise.last_date()
        # C++: F = S/r * q  (= forward under each process).
        f1 = (
            self._process1.state_variable().value()
            / self._process1.risk_free_rate().discount(maturity_date)
            * self._process1.dividend_yield().discount(maturity_date)
        )
        f2 = (
            self._process2.state_variable().value()
            / self._process2.risk_free_rate().discount(maturity_date)
            * self._process2.dividend_yield().discount(maturity_date)
        )

        variance1 = self._process1.black_volatility().black_variance(
            maturity_date, f1, extrapolate=True
        )
        variance2 = self._process2.black_volatility().black_variance(
            maturity_date, f2, extrapolate=True
        )

        df = self._process1.risk_free_rate().discount(maturity_date)

        results.reset()
        results.value = self._calculate(
            f1, f2, strike, option_type, variance1, variance2, df
        )

    def update(self) -> None:
        self.notify_observers()


__all__ = ["SpreadBlackScholesVanillaEngine"]
