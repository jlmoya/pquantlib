"""QuantoEngine — quanto wrapper around any single-asset engine.

# C++ parity: ql/pricingengines/quanto/quantoengine.hpp (v1.43).

Ported as a dependency of ``QuantoDoubleBarrierOption``: without it the
quanto instrument has no engine and nothing to cross-validate against.

The engine reprices the *same* instrument under a quanto-adjusted dividend
curve (:class:`~pquantlib.termstructures.yield_.quanto_term_structure.QuantoTermStructure`)
and then rebuilds the greeks from the inner engine's:

    rho     = rho + dividendRho
    vega    = vega + corr * sigma_X * dividendRho
    qvega   = corr * sigma_S * dividendRho
    qrho    = -dividendRho
    qlambda = sigma_X * sigma_S * dividendRho

C++ warns the engine "will only work with simple Black-Scholes processes
(i.e., no Merton)"; the port carries the same caveat and no extra check.
"""

from __future__ import annotations

from collections.abc import Callable

from pquantlib import qassert
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.instruments.quanto_option_results import QuantoOptionResults
from pquantlib.option import OptionArguments
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVolTermStructure,
)
from pquantlib.termstructures.yield_.quanto_term_structure import QuantoTermStructure
from pquantlib.termstructures.yield_term_structure import YieldTermStructure

# # C++ parity: ``Real exchangeRateATMlevel = 1.0;`` (quantoengine.hpp:88).
_EXCHANGE_RATE_ATM_LEVEL: float = 1.0


class QuantoEngine[ArgsT: OptionArguments](
    GenericEngine[ArgsT, QuantoOptionResults]
):
    """Quanto pricing engine wrapping an inner single-asset engine.

    # C++ parity: ``template <class Instr, class Engine> class QuantoEngine``
    # (quantoengine.hpp:47-63 + 68-170).

    C++ names the wrapped engine as a template parameter and builds it with
    ``new Engine(quantoProcess)``. Python takes the same thing as a factory
    callable ``inner_engine_factory(process) -> GenericEngine`` plus the
    arguments object the inner engine expects, which is what
    ``Instr::arguments`` supplies in C++.
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        foreign_risk_free_rate: YieldTermStructure,
        exchange_rate_volatility: BlackVolTermStructure,
        correlation: Quote,
        arguments: ArgsT,
        inner_engine_factory: Callable[
            [GeneralizedBlackScholesProcess],
            GenericEngine[ArgsT, OneAssetOptionResults],
        ],
    ) -> None:
        super().__init__(arguments, QuantoOptionResults())
        self._process: GeneralizedBlackScholesProcess = process
        self._foreign_risk_free_rate: YieldTermStructure = foreign_risk_free_rate
        self._exchange_rate_volatility: BlackVolTermStructure = exchange_rate_volatility
        self._correlation: Quote = correlation
        self._inner_engine_factory = inner_engine_factory
        process.register_with(self)
        foreign_risk_free_rate.register_with(self)
        exchange_rate_volatility.register_with(self)
        correlation.register_with(self)

    def calculate(self) -> None:
        """# C++ parity: ``QuantoEngine<Instr,Engine>::calculate``
        # (quantoengine.hpp:85-169)."""
        payoff = self._arguments.payoff
        qassert.require(isinstance(payoff, StrikedTypePayoff), "non-striked payoff given")
        assert isinstance(payoff, StrikedTypePayoff)
        strike = payoff.strike()

        spot = self._process.state_variable()
        qassert.require(spot.value() > 0.0, "negative or null underlying")
        risk_free_rate = self._process.risk_free_rate()
        dividend_yield = QuantoTermStructure(
            self._process.dividend_yield(),
            self._process.risk_free_rate(),
            self._foreign_risk_free_rate,
            self._process.black_volatility(),
            strike,
            self._exchange_rate_volatility,
            _EXCHANGE_RATE_ATM_LEVEL,
            self._correlation.value(),
        )
        black_vol = self._process.black_volatility()

        quanto_process = GeneralizedBlackScholesProcess(
            x0=spot,
            dividend_ts=dividend_yield,
            risk_free_ts=risk_free_rate,
            black_vol_ts=black_vol,
        )

        original_engine = self._inner_engine_factory(quanto_process)
        original_engine.reset()
        original_arguments = original_engine.get_arguments()
        # # C++ parity: ``dynamic_cast<typename Instr::arguments*>`` guarded by
        # # ``QL_REQUIRE(originalArguments, "wrong engine type")``
        # # (quantoengine.hpp:119-121). The inner engine's argument bundle has
        # # to *be* the instrument's, not merely a base of it: pairing a
        # # DoubleBarrierOption with, say, AnalyticEuropeanEngine leaves the
        # # barrier fields unset and the C++ downcast is what catches it.
        # # ``isinstance`` against the bundle this engine was built with is the
        # # Python spelling of that downcast.
        qassert.require(
            isinstance(original_arguments, type(self._arguments)), "wrong engine type"
        )
        # C++ copies the argument struct wholesale (``*originalArguments =
        # this->arguments_``); Python copies the public attribute bag, which
        # is the same set of fields.
        vars(original_arguments).update(vars(self._arguments))
        original_arguments.validate()
        original_engine.calculate()

        original_results = original_engine.get_results()
        # # C++ parity: ``dynamic_cast<const typename Instr::results*>`` guarded
        # # by a second ``QL_REQUIRE(originalResults, "wrong engine type")``
        # # (quantoengine.hpp:128-130). ``Instr::results`` is
        # # ``OneAssetOption::results`` for every instrument this engine wraps,
        # # which is exactly what ``inner_engine_factory`` is declared to
        # # produce — so this arm of the C++ downcast is carried by the
        # # signature rather than by a runtime check.

        self._results.value = original_results.value
        self._results.delta = original_results.delta
        self._results.gamma = original_results.gamma
        self._results.theta = original_results.theta
        orig_rho: float | None = original_results.rho
        orig_dividend_rho: float | None = original_results.dividend_rho
        orig_vega: float | None = original_results.vega

        if orig_rho is not None and orig_dividend_rho is not None:
            self._results.rho = orig_rho + orig_dividend_rho
            self._results.dividend_rho = orig_dividend_rho
        else:
            self._results.rho = None
            self._results.dividend_rho = None

        exercise = self._arguments.exercise
        assert exercise is not None
        exchange_rate_flat_vol = self._exchange_rate_volatility.black_vol(
            exercise.last_date(), _EXCHANGE_RATE_ATM_LEVEL
        )
        if orig_vega is not None and orig_dividend_rho is not None:
            self._results.vega = (
                orig_vega
                + self._correlation.value()
                * exchange_rate_flat_vol
                * orig_dividend_rho
            )
        else:
            self._results.vega = None

        if orig_dividend_rho is not None:
            volatility = self._process.black_volatility().black_vol(
                exercise.last_date(), self._process.state_variable().value()
            )
            # # C++ parity note: the C++ body re-evaluates ``blackVol(...)``
            # # inline for qvega instead of reusing the ``volatility`` local it
            # # just computed from the same arguments (quantoengine.hpp:155-159).
            # # Same value; reproduced as one call here.
            self._results.qvega = (
                self._correlation.value() * volatility * orig_dividend_rho
            )
            self._results.qrho = -orig_dividend_rho
            self._results.qlambda = (
                exchange_rate_flat_vol * volatility * orig_dividend_rho
            )
        else:
            self._results.qvega = None
            self._results.qrho = None
            self._results.qlambda = None


__all__ = ["QuantoEngine"]
