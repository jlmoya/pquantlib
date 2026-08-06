"""QuantoEngine — engine decorator adding the quanto adjustment + greeks.

# C++ parity: ql/pricingengines/quanto/quantoengine.hpp (v1.43).

The whole quanto effect is a change of dividend curve: the engine wraps
the underlying's dividend yield in a
:class:`~pquantlib.termstructures.yield_.quanto_term_structure.QuantoTermStructure`
(which folds in the foreign risk-free curve, the exchange-rate volatility
and the correlation), rebuilds the Black-Scholes process around it, and
delegates the actual pricing to an ordinary engine
(``QuantoEngine<Instr, Engine>::calculate``, quantoengine.hpp:87-172).

The delegated results are then re-expressed:

* ``rho`` absorbs the delegated ``dividendRho``, because in the quanto
  process the domestic rate enters the dividend curve too;
* ``vega`` picks up ``rho_corr * sigma_X * dividendRho``;
* the three quanto greeks are all proportional to the delegated
  ``dividendRho``, for the same reason.

Every one of those is ``None`` when the delegated engine leaves
``dividendRho`` unset — which is exactly what happens with
``AnalyticBarrierEngine``.

C++ carries two template parameters: ``Instr`` (which fixes the arguments
type) and ``Engine`` (the delegate, instantiated internally over the
quanto process). Python takes them as constructor arguments —
``arguments``, a fresh carrier of the instrument's argument type, and
``engine_factory``, a callable from the quanto process to a delegate
instance. The three instantiations the library uses are::

    QuantoEngine(process, fx_rf, fx_vol, corr,
                 arguments=OptionArguments(),
                 engine_factory=AnalyticEuropeanEngine)

    QuantoEngine(process, fx_rf, fx_vol, corr,
                 arguments=ForwardOptionArguments(),
                 engine_factory=lambda p: ForwardVanillaEngine(p, AnalyticEuropeanEngine))

    QuantoEngine(process, fx_rf, fx_vol, corr,
                 arguments=BarrierOptionArguments(),
                 engine_factory=AnalyticBarrierEngine)

C++ warning, carried over: this engine only works with plain
Black-Scholes processes (no Merton jumps).
"""

from __future__ import annotations

from collections.abc import Callable

from pquantlib import qassert
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.instruments.quanto_vanilla_option import QuantoOptionResults
from pquantlib.option import OptionArguments
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVolTermStructure,
)
from pquantlib.termstructures.yield_.quanto_term_structure import QuantoTermStructure
from pquantlib.termstructures.yield_term_structure import YieldTermStructure

# C++ uses an ATM exchange-rate level of 1.0 (quantoengine.hpp:89-90).
_EXCHANGE_RATE_ATM_LEVEL: float = 1.0


def _copy_arguments(source: PricingEngineArguments, target: PricingEngineArguments) -> None:
    """Whole-carrier field copy.

    # C++ parity: ``*originalArguments = this->arguments_;``
    # (quantoengine.hpp:124) — struct assignment. Both carriers are of the
    # same type here (the delegate is chosen for the instrument), so
    # copying every field is exactly the C++ semantics.
    """
    vars(target).update(vars(source))


class QuantoEngine[ArgsT: OptionArguments](GenericEngine[ArgsT, QuantoOptionResults]):
    """Quanto decorator around an ordinary option engine.

    # C++ parity: ``template <class Instr, class Engine> class QuantoEngine``.

    Args:
        process: the Black-Scholes process of the underlying.
        foreign_risk_free_rate: risk-free curve of the underlying's own
            currency.
        exchange_rate_volatility: Black vol surface of the exchange rate.
        correlation: correlation between underlying and exchange rate.
        arguments: a fresh arguments carrier of the instrument's argument
            type — the stand-in for the C++ ``Instr`` template parameter.
        engine_factory: builds the delegate engine from the quanto-adjusted
            process — the stand-in for the C++ ``Engine`` template
            parameter.
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        foreign_risk_free_rate: YieldTermStructure,
        exchange_rate_volatility: BlackVolTermStructure,
        correlation: Quote,
        *,
        arguments: ArgsT,
        engine_factory: Callable[
            [GeneralizedBlackScholesProcess], GenericEngine[ArgsT, OneAssetOptionResults]
        ],
    ) -> None:
        super().__init__(arguments, QuantoOptionResults())
        self._process: GeneralizedBlackScholesProcess = process
        self._foreign_risk_free_rate: YieldTermStructure = foreign_risk_free_rate
        self._exchange_rate_volatility: BlackVolTermStructure = exchange_rate_volatility
        self._correlation: Quote = correlation
        self._engine_factory: Callable[
            [GeneralizedBlackScholesProcess], GenericEngine[ArgsT, OneAssetOptionResults]
        ] = engine_factory
        process.register_with(self)
        foreign_risk_free_rate.register_with(self)
        exchange_rate_volatility.register_with(self)
        correlation.register_with(self)

    def calculate(self) -> None:
        """# C++ parity: ``QuantoEngine<Instr, Engine>::calculate``."""
        args = self._arguments
        results = self._results

        qassert.require(isinstance(args.payoff, StrikedTypePayoff), "non-striked payoff given")
        assert isinstance(args.payoff, StrikedTypePayoff)
        assert args.exercise is not None
        strike = args.payoff.strike()

        spot = self._process.state_variable()
        qassert.require(spot.value() > 0.0, "negative or null underlying")

        correlation = self._correlation.value()
        # The dividend curve is where the whole quanto adjustment lives.
        dividend_yield = QuantoTermStructure(
            self._process.dividend_yield(),
            self._process.risk_free_rate(),
            self._foreign_risk_free_rate,
            self._process.black_volatility(),
            strike,
            self._exchange_rate_volatility,
            _EXCHANGE_RATE_ATM_LEVEL,
            correlation,
        )
        quanto_process = GeneralizedBlackScholesProcess(
            x0=spot,
            dividend_ts=dividend_yield,
            risk_free_ts=self._process.risk_free_rate(),
            black_vol_ts=self._process.black_volatility(),
        )

        original_engine = self._engine_factory(quanto_process)
        original_engine.reset()
        original_arguments = original_engine.get_arguments()
        _copy_arguments(args, original_arguments)
        original_arguments.validate()
        original_engine.calculate()
        original_results = original_engine.get_results()

        results.value = original_results.value
        results.delta = original_results.delta
        results.gamma = original_results.gamma
        results.theta = original_results.theta

        dividend_rho = original_results.dividend_rho
        if original_results.rho is not None and dividend_rho is not None:
            results.rho = original_results.rho + dividend_rho
            results.dividend_rho = dividend_rho
        else:
            results.rho = None
            results.dividend_rho = None

        last_date = args.exercise.last_date()
        exchange_rate_flat_vol = self._exchange_rate_volatility.black_vol(last_date, _EXCHANGE_RATE_ATM_LEVEL)
        if original_results.vega is not None and dividend_rho is not None:
            results.vega = original_results.vega + correlation * exchange_rate_flat_vol * dividend_rho
        else:
            results.vega = None

        if dividend_rho is not None:
            volatility = self._process.black_volatility().black_vol(last_date, spot.value())
            results.qvega = correlation * volatility * dividend_rho
            results.qrho = -dividend_rho
            results.qlambda = exchange_rate_flat_vol * volatility * dividend_rho
        else:
            results.qvega = None
            results.qrho = None
            results.qlambda = None


__all__ = ["QuantoEngine"]
