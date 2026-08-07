"""ForwardPerformanceVanillaEngine — forward-starting *performance* vanilla.

# C++ parity: ql/pricingengines/forward/forwardperformanceengine.hpp (v1.43)
# — ``template <class Engine> class ForwardPerformanceVanillaEngine :
# public ForwardVanillaEngine<Engine>``.

A performance option pays the forward-start payoff *normalised by the
reset spot*: ``max(eta * (S_T / S_reset - moneyness), 0)`` rather than
``max(eta * (S_T - moneyness * S_reset), 0)``. Since the strike-resetting
machinery is identical, C++ derives from
:class:`~pquantlib.pricingengines.forward.forward_vanilla_engine.ForwardVanillaEngine`
and overrides only the result mapping.

The difference is entirely in ``get_original_results``:

=============  =================================  =============================
               ForwardVanillaEngine               ForwardPerformanceVanillaEngine
=============  =================================  =============================
discount       ``q_df(reset)``                    ``r_df(reset) / S(0)``
delta          ``discQ * (delta + m * dK)``       hard ``0.0``
gamma          ``0.0``                            ``0.0``
theta          dividend zero rate * value         risk-free zero rate * value
rho            ``discQ * rho``                    ``-resetTime * value + disc * rho``
dividend rho   ``-resetTime * value + discQ * .`` ``disc * dividendRho``
Null guards    each greek guarded                 none
=============  =================================  =============================

C++ takes the wrapped engine as a template parameter; Python has no
template parameter, so — exactly as in the base class — it arrives as
``engine_factory``, a callable from the forward-started process to a fresh
engine. Passing the engine class itself is the direct translation::

    ForwardPerformanceVanillaEngine(process, AnalyticEuropeanEngine)
"""

from __future__ import annotations

from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.pricingengines.forward.forward_vanilla_engine import (
    ForwardVanillaEngine,
    VanillaEngineFactory,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency


class ForwardPerformanceVanillaEngine(ForwardVanillaEngine):
    """Forward performance engine for vanilla options.

    # C++ parity: ``ForwardPerformanceVanillaEngine<Engine>``.

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
        super().__init__(process, engine_factory)

    def _get_original_results(self, original_results: OneAssetOptionResults) -> None:
        """Rescale the reset-date results by ``r_df(reset) / S(0)``.

        # C++ parity: ``ForwardPerformanceVanillaEngine<Engine>::
        # getOriginalResults`` (forwardperformanceengine.hpp:69-89).
        """
        args = self._arguments
        results = self._results
        assert args.reset_date is not None

        rfdc = self._process.risk_free_rate().day_counter()
        reset_time = rfdc.year_fraction(
            self._process.risk_free_rate().reference_date(), args.reset_date
        )
        # "it's a performance option": the risk-free discount is divided by the
        # spot, which is what turns the absolute payoff into a relative one.
        disc_r = self._process.risk_free_rate().discount(args.reset_date)
        disc_r /= self._process.state_variable().value()

        # C++ dereferences the inner results without the Null guards its base
        # class uses; the assertions below make that precondition explicit
        # instead of silently propagating a None.
        assert original_results.value is not None
        assert original_results.vega is not None
        assert original_results.rho is not None
        assert original_results.dividend_rho is not None

        results.value = disc_r * original_results.value
        results.delta = 0.0
        results.gamma = 0.0
        results.theta = (
            self._process.risk_free_rate()
            .zero_rate(
                args.reset_date,
                Compounding.Continuous,
                Frequency.NoFrequency,
                False,
                rfdc,
            )
            .rate()
            * results.value
        )
        results.vega = disc_r * original_results.vega
        results.rho = -reset_time * results.value + disc_r * original_results.rho
        results.dividend_rho = disc_r * original_results.dividend_rho


__all__ = ["ForwardPerformanceVanillaEngine"]
