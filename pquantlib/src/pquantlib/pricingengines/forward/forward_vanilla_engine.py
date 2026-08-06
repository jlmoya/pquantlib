"""ForwardVanillaEngine — engine decorator for strike-resetting vanillas.

# C++ parity: ql/pricingengines/forward/forwardengine.hpp (v1.43).
# (In v1.42.1 and earlier the same class lived in
# ql/pricingengines/forward/forwardvanillaengine.hpp; v1.43 merged the two
# headers into forwardengine.hpp. The class name is unchanged.)

A forward-starting option fixes its strike at ``moneyness * S(reset)``.
The engine prices it by building the process *as seen from the reset
date* — dividend and risk-free curves implied forward
(``ImpliedTermStructure``), Black vol implied forward
(``ImpliedVolTermStructure``) — pricing an ordinary vanilla with strike
``moneyness * S(0)`` on that process, then discounting and rescaling the
results back to today (``getOriginalResults``, forwardengine.hpp:145-172).

C++ takes the wrapped engine as a template parameter ``Engine`` and
instantiates it internally over the forward-started process. Python has no
template parameter, so the wrapped engine arrives as ``engine_factory``:
a callable from the forward-started process to a fresh engine instance.
Passing the engine *class* itself is the direct translation of the C++
template argument::

    ForwardVanillaEngine(process, AnalyticEuropeanEngine)
"""

from __future__ import annotations

from collections.abc import Callable

from pquantlib import qassert
from pquantlib.instruments.forward_vanilla_option import ForwardOptionArguments
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.option import OptionArguments
from pquantlib.payoffs import PlainVanillaPayoff, StrikedTypePayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.termstructures.volatility.equity_fx.implied_vol_term_structure import (
    ImpliedVolTermStructure,
)
from pquantlib.termstructures.yield_.implied_term_structure import ImpliedTermStructure
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency

type VanillaEngineFactory = Callable[
    [GeneralizedBlackScholesProcess], GenericEngine[OptionArguments, OneAssetOptionResults]
]


class ForwardVanillaEngine(GenericEngine[ForwardOptionArguments, OneAssetOptionResults]):
    """Forward (strike-resetting) engine for vanilla options.

    # C++ parity: ``template <class Engine> class ForwardVanillaEngine``.

    Args:
        process: the Black-Scholes process of the underlying.
        engine_factory: builds the wrapped vanilla engine from the
            forward-started process — the Python stand-in for the C++
            ``Engine`` template parameter.
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        engine_factory: VanillaEngineFactory,
    ) -> None:
        super().__init__(ForwardOptionArguments(), OneAssetOptionResults())
        self._process: GeneralizedBlackScholesProcess = process
        self._engine_factory: VanillaEngineFactory = engine_factory
        process.register_with(self)

    # --- helpers --------------------------------------------------------------

    def _setup(self) -> GenericEngine[OptionArguments, OneAssetOptionResults]:
        """Build the forward-started process + wrapped engine.

        # C++ parity: ``ForwardVanillaEngine<Engine>::setup``
        # (forwardengine.hpp:76-134).
        """
        args = self._arguments
        qassert.require(isinstance(args.payoff, StrikedTypePayoff), "wrong payoff given")
        assert isinstance(args.payoff, StrikedTypePayoff)
        assert args.moneyness is not None
        assert args.reset_date is not None
        assert args.exercise is not None

        payoff = PlainVanillaPayoff(args.payoff.option_type(), args.moneyness * self._process.x0())

        # The vol is interpolated at the *right level*, so the spot quote is
        # shared with the original process rather than re-quoted.
        spot = self._process.state_variable()
        qassert.require(spot.value() > 0.0, "negative or null underlying given")

        dividend_yield = ImpliedTermStructure(self._process.dividend_yield(), args.reset_date)
        risk_free_rate = ImpliedTermStructure(self._process.risk_free_rate(), args.reset_date)
        # C++ note, carried over: implying the vol forward is fine while the
        # vol is at most time-dependent, and plain wrong if it is
        # asset-dependent (that would need stochastic or local volatility).
        black_volatility = ImpliedVolTermStructure(self._process.black_volatility(), args.reset_date)

        fwd_process = GeneralizedBlackScholesProcess(
            x0=spot,
            dividend_ts=dividend_yield,
            risk_free_ts=risk_free_rate,
            black_vol_ts=black_volatility,
        )

        original_engine = self._engine_factory(fwd_process)
        original_engine.reset()
        original_arguments = original_engine.get_arguments()
        original_arguments.payoff = payoff
        original_arguments.exercise = args.exercise
        original_arguments.validate()
        return original_engine

    def _get_original_results(self, original_results: OneAssetOptionResults) -> None:
        """Discount + rescale the reset-date results back to today.

        # C++ parity: ``ForwardVanillaEngine<Engine>::getOriginalResults``
        # (forwardengine.hpp:143-172).
        """
        args = self._arguments
        results = self._results
        assert args.moneyness is not None
        assert args.reset_date is not None

        rfdc = self._process.risk_free_rate().day_counter()
        divdc = self._process.dividend_yield().day_counter()
        reset_time = rfdc.year_fraction(self._process.risk_free_rate().reference_date(), args.reset_date)
        disc_q = self._process.dividend_yield().discount(args.reset_date)

        assert original_results.value is not None
        results.value = disc_q * original_results.value
        # The strike is proportional to the spot at reset, so the delta picks
        # up the strike sensitivity scaled by the moneyness.
        if original_results.delta is not None and original_results.strike_sensitivity is not None:
            results.delta = disc_q * (
                original_results.delta + args.moneyness * original_results.strike_sensitivity
            )
        results.gamma = 0.0
        results.theta = (
            self._process.dividend_yield()
            .zero_rate(
                args.reset_date,
                Compounding.Continuous,
                Frequency.NoFrequency,
                False,
                divdc,
            )
            .rate()
            * results.value
        )
        if original_results.vega is not None:
            results.vega = disc_q * original_results.vega
        if original_results.rho is not None:
            results.rho = disc_q * original_results.rho
        if original_results.dividend_rho is not None:
            results.dividend_rho = -reset_time * results.value + disc_q * original_results.dividend_rho

    # --- main entry point -----------------------------------------------------

    def calculate(self) -> None:
        """# C++ parity: ``ForwardVanillaEngine<Engine>::calculate``."""
        original_engine = self._setup()
        original_engine.calculate()
        self._get_original_results(original_engine.get_results())


__all__ = ["ForwardVanillaEngine", "VanillaEngineFactory"]
