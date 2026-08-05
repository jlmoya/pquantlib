"""ImpliedVolatilityHelper — shared machinery for ``impliedVolatility()``.

# C++ parity: ql/instruments/impliedvolatility.hpp + .cpp (v1.43),
#             namespace ``QuantLib::detail``.

Two static utilities used by option classes that expose an
``implied_volatility()`` method:

* :meth:`ImpliedVolatilityHelper.clone` — copy a
  ``GeneralizedBlackScholesProcess``, keeping its state variable, dividend
  curve and risk-free curve, and replacing **only** the volatility with a
  flat one driven by a mutable quote.
* :meth:`ImpliedVolatilityHelper.calculate` — Brent-solve that quote so the
  engine reproduces a target value.

C++ puts these in ``namespace detail``; Python has no such convention, so
the class is public in its own module. The private ``PriceError`` functor
(impliedvolatility.cpp:26-56) is a closure here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.instruments.instrument import InstrumentResults
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)

if TYPE_CHECKING:
    from pquantlib.instruments.instrument import Instrument
    from pquantlib.pricingengines.pricing_engine import PricingEngine


class ImpliedVolatilityHelper:
    """Helper for one-asset implied-volatility calculation.

    The passed engine must be linked to the passed quote — see
    :meth:`clone` for the standard way to arrange that.
    """

    @staticmethod
    def calculate(
        instrument: Instrument,
        engine: PricingEngine,
        vol_quote: SimpleQuote,
        target_value: float,
        accuracy: float,
        max_evaluations: int,
        min_vol: float,
        max_vol: float,
    ) -> float:
        """Solve for the volatility reproducing ``target_value``.

        C++ parity: impliedvolatility.cpp:61-81. Brent on
        ``f(x) = engine_value(x) - target_value``, bracketed on
        ``[min_vol, max_vol]``, starting from their midpoint.
        """
        engine.reset()
        args = engine.get_arguments()
        instrument.setup_arguments(args)
        args.validate()

        results = engine.get_results()
        qassert.require(
            isinstance(results, InstrumentResults),
            "pricing engine does not supply needed results",
        )
        assert isinstance(results, InstrumentResults)

        def price_error(x: float) -> float:
            # C++ parity: impliedvolatility.cpp:50-54 ``PriceError::operator()``.
            vol_quote.set_value(x)
            engine.calculate()
            value = engine.get_results()
            assert isinstance(value, InstrumentResults)
            assert value.value is not None
            return value.value - target_value

        solver = Brent()
        solver.set_max_evaluations(max_evaluations)
        guess = (min_vol + max_vol) / 2.0
        return solver.solve(price_error, accuracy, guess, min_vol, max_vol)

    @staticmethod
    def clone(
        process: GeneralizedBlackScholesProcess,
        vol_quote: SimpleQuote,
    ) -> GeneralizedBlackScholesProcess:
        """Copy ``process`` with a flat volatility driven by ``vol_quote``.

        C++ parity: impliedvolatility.cpp:83-102. The state variable, the
        dividend curve and the risk-free curve are carried over unchanged —
        only the volatility term structure is replaced. The replacement
        keeps the original vol structure's reference date, calendar and day
        counter, so a recovered volatility is expressed on the same clock.
        """
        black_vol = process.black_volatility()
        volatility = BlackConstantVol(
            reference_date=black_vol.reference_date(),
            calendar=black_vol.calendar(),
            day_counter=black_vol.day_counter(),
            volatility=vol_quote,
        )
        return GeneralizedBlackScholesProcess(
            x0=process.state_variable(),
            dividend_ts=process.dividend_yield(),
            risk_free_ts=process.risk_free_rate(),
            black_vol_ts=volatility,
        )


__all__ = ["ImpliedVolatilityHelper"]
